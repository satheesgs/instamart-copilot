"""Part D, tasks 5-8: refund agent with tools, a ReAct loop, safe stops and crash-and-resume checkpoints.
Writes outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl and state/*.json.
Run:  python src/part_d_agent.py"""
import json
import os
import sqlite3
import time
from datetime import datetime

import pandas as pd

from common import DB_PATH, OUTPUTS, RAW, STATE, ensure_dirs, heading, md_table
import mock_llm
from part_d_rag import retrieve

RUNS_PATH = os.path.join(OUTPUTS, "agent_runs.jsonl")
LEDGER_PATH = os.path.join(OUTPUTS, "refund_ledger.jsonl")
LATE_MINUTES, AUTO_LIMIT_INR = 45, 1000

SYSTEM_PROMPT = ("You are the Instamart refund AGENT. Decide refund requests by calling tools: look up the order, "
                 "check the refund policy, calculate the refund, then issue it (up to INR 1000) or escalate it to a "
                 "human (above INR 1000). Finish with one line that starts with REFUND, ESCALATE, DENIED or "
                 "NOT_FOUND followed by a colon.")


# ----------------------------------------------------------------- task 5: tools
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Full refund when a completed order took more than 45 minutes, otherwise 0."""
    if delivery_minutes is None or float(delivery_minutes) <= LATE_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(float(order_amount), 2)}


_CURRENT_RUN = {"run_id": None}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"), "tool": tool,
                            "run_id": _CURRENT_RUN["run_id"], "order_id": order_id,
                            "refund_inr": round(float(refund_inr), 2)}) + "\n")
    return {"refund_inr": round(float(refund_inr), 2)}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in the Instamart database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "e.g. ORD-100123"}},
                    "required": ["order_id"]}},
    {"name": "search_policy", "description": "Search the policy documents and return the three most relevant passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR for a completed order from its amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an approved refund of at most INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}


# ----------------------------------------------------------------- task 8: checkpoints
class SimulatedCrash(Exception):
    pass


def save_checkpoint(state):
    with open(os.path.join(STATE, f"{state['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def load_checkpoint(run_id):
    with open(os.path.join(STATE, f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- task 6-7: ReAct loop with safe stops
def _handoff(state, reason):
    tools = ", ".join(state["tools_executed"]) or "none"
    return (f"HANDOFF TO HUMAN: {reason}. Request: '{state['user_message']}'. "
            f"Completed {state['step']} tool step(s) ({tools}) in {state['llm_calls']} model call(s), "
            f"{state['tokens_used']} tokens. Please review and finish this case manually.")


def _result(state, status, final_answer):
    state.update(status=status, final_answer=final_answer)
    save_checkpoint(state)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status,
            "final_answer": final_answer, "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "tools_executed": state["tools_executed"]}


def _loop(state, crash_after_tools=None):
    _CURRENT_RUN["run_id"] = state["run_id"]
    start = time.monotonic()
    lim = state["limits"]
    while True:
        if time.monotonic() - start >= lim["time_limit_seconds"]:
            return _result(state, "FALLBACK_TIMEOUT", _handoff(state, "time limit reached"))
        if state["llm_calls"] >= lim["max_iterations"]:
            return _result(state, "FALLBACK_MAX_ITERATIONS",
                           _handoff(state, f"no final answer after {lim['max_iterations']} model calls"))
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += resp.usage.total_tokens
        msg = resp.choices[0].message
        if state["tokens_used"] > lim["token_limit"]:          # stop before executing the requested tool
            return _result(state, "FALLBACK_TOKEN_LIMIT",
                           _handoff(state, f"token limit {lim['token_limit']} exceeded"))
        if not msg.tool_calls:
            final = msg.content or ""
            return _result(state, final.split(":", 1)[0].strip(), final)
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                                                 for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            fn = TOOL_MAP.get(tc.name)
            observation = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
            state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                                      "content": json.dumps(observation)})
            state["step"] += 1
            state["tools_executed"].append(tc.name)
            state["steps"].append({"step": state["step"], "tool": tc.name, "arguments": tc.arguments,
                                   "observation": observation})
            save_checkpoint(state)                               # checkpoint after every observation
            if crash_after_tools is not None and state["step"] == crash_after_tools:
                raise SimulatedCrash(f"simulated crash after tool #{state['step']} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    state = {"run_id": run_id or f"{scenario_id}_{datetime.now():%Y%m%d%H%M%S%f}", "scenario_id": scenario_id,
             "user_message": user_message,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                        "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
             "status": "RUNNING", "final_answer": None}
    return _loop(state, crash_after_tools)


def resume_agent(run_id):
    state = load_checkpoint(run_id)
    print(f"  resuming {run_id} from step {state['step']} (tools already executed: {state['tools_executed']})")
    return _loop(state)


# ----------------------------------------------------------------- helpers / demo
def log_run(result, label):
    record = {"run_label": label, "run_id": result["run_id"], "scenario_id": result["scenario_id"],
              "status": result["status"], "final_answer": result["final_answer"],
              "llm_calls": result["llm_calls"], "tokens_used": result["tokens_used"],
              "steps": [{"tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]}
                        for s in result["steps"]]}
    with open(RUNS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def refund_amount(result):
    for s in result["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"):
            return s["observation"]["refund_inr"]
    return 0.0


def main():
    ensure_dirs()
    for path in (RUNS_PATH, LEDGER_PATH):                         # each full demo starts from empty logs
        if os.path.exists(path):
            os.remove(path)
    for fn in os.listdir(STATE):
        if fn.endswith(".json"):
            os.remove(os.path.join(STATE, fn))

    heading("PART D - TASK 5: TOOLS")
    for t in TOOLS_SPEC:
        print(f"  {t['name']:<18} {t['description']}")

    heading("PART D - TASK 6: SEVEN SCENARIOS (ReAct loop, max_iterations=6, token_limit=6000)")
    scen = pd.read_csv(os.path.join(RAW, "agent_scenarios.csv"), keep_default_na=False)
    rows = []
    for s in scen.itertuples(index=False):
        res = run_agent(s.message, s.scenario_id, run_id=f"{s.scenario_id}_scenario")
        log_run(res, "scenario")
        amt = refund_amount(res)
        ok = (res["status"] == s.expected_outcome or
              (s.expected_outcome == "FALLBACK" and res["status"].startswith("FALLBACK"))) and \
            abs(amt - float(s.expected_refund_inr)) <= 0.01
        rows.append({"scenario": s.scenario_id, "expected": s.expected_outcome, "status": res["status"],
                     "refund_inr": amt, "model_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                     "tools": " > ".join(res["tools_executed"]), "matches_expected": ok})
    table = pd.DataFrame(rows)
    print(md_table(table))
    assert table["matches_expected"].all()

    heading("PART D - TASK 7: STOP SAFELY")
    s1 = scen.iloc[0]
    tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_demo")
    log_run(tok, "demo_token_limit_200")
    print(f"S1 with token_limit=200      -> {tok['status']} after {tok['llm_calls']} calls, "
          f"{tok['tokens_used']} tokens, tools {tok['tools_executed']}")
    print(f"  final_answer: {tok['final_answer']}")
    tmo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_timeout_demo")
    log_run(tmo, "demo_time_limit_0")
    print(f"S1 with time_limit_seconds=0 -> {tmo['status']} after {tmo['llm_calls']} calls")
    print(f"  final_answer: {tmo['final_answer']}")
    s7 = table[table["scenario"] == "S7"].iloc[0]
    print(f"S7 with defaults             -> {s7['status']} after {s7['model_calls']} calls, {s7['tokens_used']} tokens")
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tmo["status"] == "FALLBACK_TIMEOUT"

    heading("PART D - TASK 8: CRASH AND RESUME (S1)")
    run_id = "S1_crash_resume_demo"
    try:
        run_agent(s1.message, "S1", run_id=run_id, crash_after_tools=2)
    except SimulatedCrash as err:
        print(f"  {err}; checkpoint saved at state/{run_id}.json")
    before = load_checkpoint(run_id)["tools_executed"]
    res = resume_agent(run_id)
    log_run(res, "demo_crash_resume")
    print(f"  tools before crash: {before}")
    print(f"  tools after resume: {res['tools_executed'][len(before):]}")
    print(f"  all tools executed, in order: {res['tools_executed']}")
    print(f"  final status: {res['status']} | refund INR {refund_amount(res):.2f} | answer: {res['final_answer']}")
    assert res["tools_executed"] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert res["status"] == "REFUND" and abs(refund_amount(res) - 128.25) <= 0.01

    with open(LEDGER_PATH, encoding="utf-8") as f:
        ledger = [json.loads(line) for line in f]
    print(f"\nrefund ledger entries: {len(ledger)}")
    for e in ledger:
        print(f"  {e['tool']:<18} {e['order_id']}  INR {e['refund_inr']:.2f}  (run {e['run_id']})")
    print(f"agent runs logged: {sum(1 for _ in open(RUNS_PATH, encoding='utf-8'))} lines in outputs/agent_runs.jsonl")


if __name__ == "__main__":
    main()
