"""Part A, tasks 6-7: run the seven SQL queries and the pandas cross-check / exploratory analysis.
Run:  python src/part_a_analysis.py"""
import os
import re
import sqlite3

import pandas as pd

from common import CLEAN, DB_PATH, SQL_PATH, heading, md_table


def load_queries(path=SQL_PATH):
    text = open(path, encoding="utf-8").read()
    parts = re.split(r"^-- name: (Q\d+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_sql():
    heading("PART A - TASK 6: SQL ANALYSIS")
    queries = load_queries()
    con = sqlite3.connect(DB_PATH)
    res = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
    con.close()

    q1 = res["Q1"]
    print(f"\nQ1 WHERE + HAVING: {len(q1)} customers")

    q2 = res["Q2"]
    top = q2.iloc[0]
    print(f"Q2 city x category revenue: {len(q2)} rows, total INR {q2['revenue_inr'].sum():,.2f}, "
          f"largest {top['city']} / {top['category']} INR {top['revenue_inr']:,.2f}")

    q3 = res["Q3"]
    print(f"Q3 products above average price: {len(q3)} products")

    q4 = res["Q4"]
    print(f"Q4 customers with no completed order: {len(q4)} customers")

    q5 = res["Q5"]
    print(f"Q5 customers above average spend: {len(q5)} customers; average INR {q5['average_spend_inr'].iloc[0]:,.2f}")

    q6 = res["Q6"]
    tops = q6[q6["row_num"] == 1][["city", "customer_id", "total_spend"]]
    print("Q6 top customer per city:")
    print(md_table(tops))
    ties = q6[q6["rnk"] != q6["row_num"]]
    print(f"Q6 rows where RANK() != ROW_NUMBER(): {len(ties)}")
    print(md_table(ties))

    q7 = res["Q7"]
    print(f"Q7 running total: {len(q7)} daily rows; last cumulative INR {q7['cumulative_revenue_inr'].iloc[-1]:,.2f}")
    return res


def pandas_analysis(sql_q2_total):
    heading("PART A - TASK 7: PANDAS CROSS-CHECK AND EXPLORATORY ANALYSIS")
    c = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    p = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    o = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    comp = (o[o["status"] == "completed"]
            .merge(c[["customer_id", "city"]], on="customer_id")
            .merge(p[["product_id", "category"]], on="product_id"))

    # (a) pivot table with margins = SQL Q2 rebuilt
    pivot = pd.pivot_table(comp, values="order_amount", index="city", columns="category",
                           aggfunc="sum", margins=True, margins_name="Total")
    print(pivot.round(2).to_string())
    grand = pivot.loc["Total", "Total"]
    print(f"\npivot grand total INR {grand:,.2f} vs SQL Q2 total INR {sql_q2_total:,.2f} "
          f"-> equal: {round(grand, 2) == round(sql_q2_total, 2)}")

    by_city = (comp.groupby("city")
                   .agg(sum_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                        count=("order_amount", "count"))
                   .round(2).reset_index())
    print("\nper-city completed order_amount (named aggregation):")
    print(md_table(by_city))

    # (b) distribution of completed order_amount
    a = comp["order_amount"]
    q1, q3 = a.quantile(0.25), a.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    stats = [("mean", a.mean()), ("median", a.median()), ("skewness", a.skew()),
             ("Q1 (25%)", q1), ("Q2 (50%)", a.quantile(0.5)), ("Q3 (75%)", q3), ("IQR", iqr),
             ("lower fence (Q1 - 1.5 IQR)", lo), ("upper fence (Q3 + 1.5 IQR)", hi),
             ("outliers below lower fence", float((a < lo).sum())),
             ("outliers above upper fence", float((a > hi).sum()))]
    print("\ncompleted order_amount statistics:")
    print(md_table(pd.DataFrame(stats, columns=["statistic", "value"])))

    corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    print("\ncorrelation matrix (completed orders):")
    print(corr.round(3).to_string())
    pairs = corr.where(~pd.DataFrame([[i >= j for j in range(4)] for i in range(4)],
                                     index=corr.index, columns=corr.columns)).stack()
    best = pairs.abs().idxmax()
    print(f"largest |correlation|: {best[0]} vs {best[1]} = {pairs[best]:.3f}")
    print(f"order_amount vs quantity = {corr.loc['order_amount', 'quantity']:.3f}")


if __name__ == "__main__":
    results = run_sql()
    pandas_analysis(results["Q2"]["revenue_inr"].sum())
