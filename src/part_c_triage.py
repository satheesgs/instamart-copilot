"""Part C: support-ticket triage with guardrails around the supplied mock_llm.py.
Input guardrails -> PII redaction -> resilient (retried) model call -> output validation -> token budget -> routing.
Writes outputs/triage_budget_demo.csv and outputs/triage_results.csv.   Run:  python src/part_c_triage.py"""
import json
import os
import random
import re
import time

import pandas as pd

from common import RAW, OUTPUTS, ensure_dirs, heading, md_table
import mock_llm
from mock_llm import TransientLLMError, est_tokens

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "error_permanent"}

# --------------------------------------------------------------- PII redaction (applied in this order)
PII_PATTERNS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]


def redact(text):
    counts = {}
    for name, pattern, placeholder in PII_PATTERNS:
        text, counts[name] = pattern.subn(placeholder, text)
    return text, counts


def contains_pii(text):
    return any(p.search(text or "") for _, p, _ in PII_PATTERNS)


# --------------------------------------------------------------- prompts
ZERO_SHOT = """You are a support-ticket triage assistant for a quick-commerce grocery app. CLASSIFY the customer ticket.
Allowed categories (choose exactly one): billing, delivery, technical, fraud.
- billing: wrong amount, double charge, invoice or payment deduction problems
- delivery: late, missing or wrongly marked deliveries and rider issues
- technical: app crashes, login or OTP problems, errors in the app
- fraud: unauthorized transactions, hacked accounts, orders the customer never placed
Reply with JSON only, no other text, in exactly this form: {"category": "<one of the four>", "reason": "<short reason>"}"""

FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: Someone used my card for order [ORDER_ID] that I never placed. This is fraud, call [PHONE].
Answer: {"category": "fraud", "reason": "unauthorized transaction reported"}
Ticket: The app shows an error on the payment page and freezes when I apply a coupon.
Answer: {"category": "technical", "reason": "app error while paying"}
Ticket: I was charged twice for order [ORDER_ID], please refund the extra amount.
Answer: {"category": "billing", "reason": "double charge"}
Ticket: My order [ORDER_ID] is very late and the rider is not answering.
Answer: {"category": "delivery", "reason": "late delivery"}"""


# --------------------------------------------------------------- resilient call
def call_with_retry(messages, ticket_id, retry_log):
    """Call the model; on TransientLLMError wait BASE_DELAY * 2**n * jitter and retry (n = 0, 1, 2)."""
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as err:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            print(f"  retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s error='{err}'")
            retry_log.append({"ticket_id": ticket_id, "retry_number": n, "wait": wait})
            time.sleep(wait)


# --------------------------------------------------------------- output validation
def validate(reply):
    """Return (status, category). Never guess: anything ambiguous goes to human review."""
    try:
        data = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        words = set(re.findall(r"[a-z]+", (reply or "").lower()))
        found = [c for c in CATEGORIES if c in words]
        if len(found) == 1:
            return "classified_keyword", found[0]
        return "flagged_invalid_output", ""
    if isinstance(data, dict) and data.get("category") in CATEGORIES:
        return "classified_json", data["category"]
    return "flagged_invalid_output", ""


# --------------------------------------------------------------- one full pipeline run
def run_pipeline(tickets, system_prompt, token_budget=None, label=""):
    mock_llm.reset()                                    # fresh counters before every full run
    random.seed(42)                                     # reproducible jitter
    retry_log, rows, total_tokens, calls = [], [], 0, 0
    print(f"\n--- run: {label} (token budget: {token_budget}) ---")
    for t in tickets.itertuples(index=False):
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "true_category": t.true_category,
               "status": "", "category": "", "tokens_used": 0}
        if t.text.strip() == "":
            row["status"] = "skipped_empty"
        elif est_tokens(t.text) > MAX_INPUT_TOKENS:
            row["status"] = "flagged_long"
        elif token_budget is not None and total_tokens >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            clean_text, _ = redact(t.text)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": clean_text}]
            # privacy guardrail: nothing that looks like PII may leave the pipeline
            assert not any(contains_pii(m["content"]) for m in messages), f"PII in prompt for {t.ticket_id}"
            try:
                resp = call_with_retry(messages, t.ticket_id, retry_log)
                calls += 1
                used = resp.usage.total_tokens
                total_tokens += used
                row["tokens_used"] = used
                row["status"], row["category"] = validate(resp.choices[0].message.content)
            except TransientLLMError:
                row["status"] = "error_permanent"
        rows.append(row)
    df = pd.DataFrame(rows)
    for r in retry_log:                                 # every wait inside its jitter range
        lo, hi = BASE_DELAY * 2 ** r["retry_number"] * 0.9, BASE_DELAY * 2 ** r["retry_number"] * 1.15
        assert lo <= r["wait"] <= hi
    print(f"model calls answered: {calls}; TransientLLMErrors retried: {len(retry_log)}; "
          f"permanent errors: {int((df['status'] == 'error_permanent').sum())}; total tokens: {total_tokens}")
    print("status counts:", df["status"].value_counts().to_dict())
    return df, retry_log, total_tokens


def route(df, segments):
    seg = segments.set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out["customer_id"].map(seg["segment_name"]).fillna("")
    premium = out["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)
    out["route"], out["priority"] = "support_queue", "NORMAL"
    human = out["status"].isin(HUMAN_STATUSES)
    fraud = ~human & (out["category"] == "fraud")
    prem = ~human & ~fraud & premium
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    out.loc[human, "category"] = ""
    return out[["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]]


def summarise(df, tokens, name):
    cls = df[df["status"].isin(["classified_json", "classified_keyword"])]
    correct = int((cls["category"] == cls["true_category"]).sum())
    return {"prompt": name, "classified": len(cls), "correct": correct,
            "accuracy_pct": round(100 * correct / len(cls), 1), "total_tokens": tokens,
            "tokens_per_classified": round(tokens / len(cls), 1)}


def main():
    ensure_dirs()
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False, dtype=str)

    heading("PART C - TASK 1: INPUT GUARDRAILS")
    empty = tickets["text"].str.strip() == ""
    tok = tickets["text"].map(est_tokens)
    long_ = ~empty & (tok > MAX_INPUT_TOKENS)
    print(f"empty tickets: {int(empty.sum())} {list(tickets.loc[empty, 'ticket_id'])}")
    print(f"long tickets (> {MAX_INPUT_TOKENS} tokens): {int(long_.sum())} {list(tickets.loc[long_, 'ticket_id'])}; "
          f"estimated tokens: {sorted(set(tok[long_]))}")
    print(f"eligible for the model: {int((~empty & ~long_).sum())}")

    heading("PART C - TASK 2: PII REDACTION (all 164 tickets)")
    totals = {"email": 0, "phone": 0, "order_id": 0}
    redacted = []
    for text in tickets["text"]:
        r, counts = redact(text)
        redacted.append(r)
        for k in totals:
            totals[k] += counts[k]
    print(md_table(pd.DataFrame([(k, v) for k, v in totals.items()] + [("total", sum(totals.values()))],
                                columns=["type", "substitutions"])))
    leftover = sum(len(p.findall(r)) for r in redacted for _, p, _ in PII_PATTERNS)
    assert leftover == 0
    print(f"re-applying the three patterns to the redacted texts finds: {leftover}")
    print("example:", tickets["text"].iloc[0], "\n     ->", redacted[0])

    heading("PART C - TASK 3: PROMPTS")
    for name, p in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        assert "CLASSIFY" in p and all(c in p for c in CATEGORIES)
        print(f"{name}: {est_tokens(p)} estimated tokens; contains 'EXAMPLES:': {'EXAMPLES:' in p}; "
              f"contains PII: {contains_pii(p)}")
    assert "EXAMPLES:" not in ZERO_SHOT and "EXAMPLES:" in FEW_SHOT

    heading("PART C - TASKS 4-6: FULL RUNS (no token budget), RETRIES AND VALIDATION")
    zs, zs_retries, zs_tokens = run_pipeline(tickets, ZERO_SHOT, label="zero-shot")
    fs, fs_retries, fs_tokens = run_pipeline(tickets, FEW_SHOT, label="few-shot")
    assert len(zs_retries) == 19 and (zs["status"] == "error_permanent").sum() == 0
    comparison = pd.DataFrame([summarise(zs, zs_tokens, "zero-shot"), summarise(fs, fs_tokens, "few-shot")])
    print("\nzero-shot vs few-shot:")
    print(md_table(comparison, floatfmt="{:,.1f}"))
    cls = zs[zs["status"].isin(["classified_json", "classified_keyword"])]
    print("\nzero-shot confusion table (rows = true_category, columns = predicted):")
    ct = pd.crosstab(cls["true_category"], cls["category"]).reset_index()
    print(md_table(ct))

    heading("PART C - TASK 7a: TOKEN BUDGET DEMO (zero-shot, TOKEN_BUDGET = 6000)")
    segments = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
    demo, _, demo_tokens = run_pipeline(tickets, ZERO_SHOT, token_budget=6000, label="zero-shot budget demo")
    n_exh = int((demo["status"] == "budget_exhausted").sum())
    assert n_exh >= 1
    route(demo, segments).to_csv(os.path.join(OUTPUTS, "triage_budget_demo.csv"), index=False)
    print(f"budget_exhausted tickets: {n_exh} (tokens spent {demo_tokens}) -> outputs/triage_budget_demo.csv")

    heading("PART C - TASK 7b: FINAL TRIAGE (few-shot, TOKEN_BUDGET = 100000) + ROUTING")
    final, _, final_tokens = run_pipeline(tickets, FEW_SHOT, token_budget=100000, label="few-shot final")
    result = route(final, segments)
    result.to_csv(os.path.join(OUTPUTS, "triage_results.csv"), index=False)
    print(f"wrote outputs/triage_results.csv ({len(result)} rows, {final_tokens} tokens)")
    summary = (result.groupby(["route", "priority"]).size().reset_index(name="tickets"))
    print(md_table(summary))


if __name__ == "__main__":
    main()
