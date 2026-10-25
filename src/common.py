"""Shared paths and helpers used by every Part of the Copilot."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CLEAN = os.path.join(ROOT, "data", "clean")
POLICIES = os.path.join(ROOT, "data", "policies")
DB_PATH = os.path.join(ROOT, "data", "instamart.db")
SQL_PATH = os.path.join(ROOT, "sql", "queries.sql")
OUTPUTS = os.path.join(ROOT, "outputs")
STATE = os.path.join(ROOT, "state")

# make the supplied mock_llm.py (repository root) importable from src/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def ensure_dirs():
    for d in (CLEAN, OUTPUTS, STATE):
        os.makedirs(d, exist_ok=True)


def md_table(df, floatfmt="{:,.2f}"):
    """Render a DataFrame as a GitHub markdown table (for pasting into README)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):  # itertuples keeps each column's own dtype
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def heading(title):
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)
