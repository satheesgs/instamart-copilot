"""Part D, tasks 1-4: chunk the policy documents, TF-IDF retrieval, grounded answers, evaluation.
The refund agent (part_d_agent.py) reuses retrieve() from here.   Run:  python src/part_d_rag.py"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import POLICIES, RAW, heading, md_table
import mock_llm

CHUNK_WORDS, OVERLAP = 40, 10

SYSTEM_PROMPT = ("You are the Instamart POLICY ASSISTANT. Answer the question using only the policy context "
                 "supplied in the user message. Do not use outside knowledge. If the context does not contain "
                 "the answer, say: I could not find that information in the policy documents.")


# ----------------------------------------------------------------- task 1
def chunk_words(words, size=CHUNK_WORDS, overlap=OVERLAP):
    """Windows of `size` words, each starting size-overlap words after the previous one; stop after the
    first window whose end reaches or passes the last word."""
    step, chunks, start = size - overlap, [], 0
    while True:
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            return chunks
        start += step


def load_chunks(overlap=OVERLAP):
    chunks = []
    for fn in sorted(os.listdir(POLICIES)):
        if fn.endswith(".txt"):
            words = open(os.path.join(POLICIES, fn), encoding="utf-8").read().split()
            chunks += [{"source": fn, "text": t} for t in chunk_words(words, overlap=overlap)]
    return chunks


# ----------------------------------------------------------------- task 2
class PolicyIndex:
    def __init__(self, overlap=OVERLAP):
        self.chunks = load_chunks(overlap)
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = scores.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"], "score": round(float(scores[i]), 4),
                 "text": self.chunks[i]["text"]} for i in order]


_DEFAULT = None


def default_index():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = PolicyIndex()
    return _DEFAULT


def retrieve(query, k=3, index=None):
    return (index or default_index()).retrieve(query, k)


# ----------------------------------------------------------------- task 3
def build_user_message(chunks, question):
    context = "\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    return f"CONTEXT:\n{context}\nQUESTION:\n{question}"


def answer(question, index=None):
    chunks = retrieve(question, k=3, index=index)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(chunks, question)}]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [c["source"] for c in chunks]


# ----------------------------------------------------------------- task 4
def evaluate(index):
    qa = pd.read_csv(os.path.join(RAW, "policy_eval.csv"), keep_default_na=False)
    rows = []
    for q in qa.itertuples(index=False):
        text, sources = answer(q.question, index=index)
        answerable = q.expected_doc != "NONE"
        rows.append({"question": q.question, "expected_doc": q.expected_doc, "answerable": answerable,
                     "doc_in_top3": answerable and q.expected_doc in sources,
                     "fact_in_answer": answerable and q.expected_fact in text,
                     "fallback": text.startswith("I could not find"), "answer": text})
    df = pd.DataFrame(rows)
    a, u = df[df["answerable"]], df[~df["answerable"]]
    summary = {"expected_doc in top 3 (of 12)": int(a["doc_in_top3"].sum()),
               "answer contains expected_fact (of 12)": int(a["fact_in_answer"].sum()),
               "unanswerable -> fallback (of 2)": int(u["fallback"].sum())}
    return df, summary


def main():
    heading("PART D - TASK 1: CHUNKING (40 words, overlap 10)")
    index = default_index()
    counts = pd.Series([c["source"] for c in index.chunks]).value_counts()
    print(f"total chunks: {len(index.chunks)}")
    print(md_table(counts.rename_axis("source").reset_index(name="chunks")))

    heading("PART D - TASK 2: TF-IDF RETRIEVAL")
    for r in retrieve("refund policy late delivery"):
        print(f"  {r['source']:<22} score {r['score']:.4f}  {r['text'][:70]}...")

    heading("PART D - TASK 3: GROUNDED ANSWER EXAMPLE")
    q = "Who must approve a refund above INR 1000?"
    print("user message sent to the model:\n" + build_user_message(retrieve(q), q))
    text, sources = answer(q)
    print(f"\nanswer: {text}\nsources: {sources}")

    heading("PART D - TASK 4: EVALUATION (overlap 10 vs overlap 0)")
    results = {}
    for ov in (10, 0):
        df, summary = evaluate(PolicyIndex(overlap=ov))
        results[ov] = df
        print(f"\noverlap {ov} ({len(PolicyIndex(overlap=ov).chunks)} chunks): {summary}")
    print("\nper-question comparison:")
    cmp_ = pd.DataFrame({"question": results[10]["question"],
                         "doc@3 ov10": results[10]["doc_in_top3"], "fact ov10": results[10]["fact_in_answer"],
                         "doc@3 ov0": results[0]["doc_in_top3"], "fact ov0": results[0]["fact_in_answer"],
                         "fallback ov10": results[10]["fallback"], "fallback ov0": results[0]["fallback"]})
    print(md_table(cmp_))
    broken = results[0][results[0]["answerable"] & ~(results[0]["doc_in_top3"] & results[0]["fact_in_answer"])]
    chunks0 = PolicyIndex(overlap=0).chunks
    for r in broken.itertuples(index=False):
        print(f"\nBREAKS with overlap 0: {r.question}\n  answer given: {' '.join(r.answer.split())}")
        for i, c in enumerate(chunks0[:-1]):            # show the chunk boundary that cut the sentence
            nxt = chunks0[i + 1]
            if c["source"] == r.expected_doc == nxt["source"] and \
                    any(w in (c["text"][-60:] + nxt["text"][:60]) for w in ("45 minutes", "48 hours")):
                print(f"  boundary: '...{c['text'][-45:]}' | '{nxt['text'][:45]}...'")


if __name__ == "__main__":
    main()
