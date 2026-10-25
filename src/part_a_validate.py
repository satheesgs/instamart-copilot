"""Part A, task 5: assert the reference checkpoints on the clean data and the database.
Run:  python src/part_a_validate.py   -> prints ALL CHECKPOINTS PASSED"""
import os
import sqlite3

import pandas as pd

from common import CLEAN, DB_PATH

c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
o = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
completed = o[o["status"] == "completed"]
cancelled = o[o["status"] == "cancelled"]

# customers (task 3)
assert c["city"].value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197,
                                              "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}, "city counts"
assert c["gender"].value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30}, "gender counts"
assert c["age"].isna().sum() == 0, "missing ages"
assert c["age"].between(18, 60).all(), "age range 18-60"

# orders (task 4)
assert len(o) == 12309 and o["order_id"].is_unique, "one row per order_id"
assert len(completed) == 11110, "completed count"
assert len(cancelled) == 1199, "cancelled count"
assert (o["status"] == "pending").sum() == 0, "no pending rows"
assert o["order_amount"].isna().sum() == 0, "missing order_amount"
assert completed["delivery_minutes"].isna().sum() == 0, "missing delivery among completed"
assert cancelled["delivery_minutes"].isna().sum() == 1199, "cancelled delivery left missing"
assert (o["delivery_minutes"].dropna() >= 0).all(), "no negative delivery"
assert round(completed["order_amount"].sum(), 2) == 2998668.80, "completed revenue"

# database (task 5)
with sqlite3.connect(DB_PATH) as con:
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
    ts = con.execute("SELECT order_ts FROM orders LIMIT 1").fetchone()[0]
    db_rev = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status = 'completed'").fetchone()[0]
assert counts == {"customers": 1200, "products": 40, "orders": 12309}, "database row counts"
assert len(ts) == 19 and ts[10] == " ", "order_ts stored as YYYY-MM-DD HH:MM:SS text"
assert db_rev == 2998668.80, "database revenue"

print("ALL CHECKPOINTS PASSED")
