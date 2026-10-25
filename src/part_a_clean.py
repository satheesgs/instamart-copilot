"""Part A, tasks 2-5: audit the raw exports, clean customers and orders, store CSV + SQLite.
Run from the repository root:  python src/part_a_clean.py"""
import os
import sqlite3

import pandas as pd

from common import RAW, CLEAN, DB_PATH, ensure_dirs, heading, md_table

pd.set_option("display.width", 140)


# ----------------------------------------------------------------- audit
def missing_report(name, df):
    print(f"\n### {name}: shape {df.shape[0]} rows x {df.shape[1]} columns")
    rep = pd.DataFrame({"column": df.columns,
                        "dtype": [str(t) for t in df.dtypes],
                        "missing": df.isna().sum().values,
                        "missing_pct": (df.isna().mean() * 100).round(2).values})
    print(md_table(rep))


def audit():
    heading("PART A - TASK 2: AUDIT OF THE RAW DATA")
    # standard pandas load for shape / dtypes / missing values
    customers = pd.read_csv(os.path.join(RAW, "customers.csv"))
    products = pd.read_csv(os.path.join(RAW, "products.csv"))
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
    tickets = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
    for name, df in [("customers.csv", customers), ("products.csv", products), ("orders.csv", orders)]:
        missing_report(name, df)
    print(f"\n(tickets.csv read with keep_default_na=False: {tickets.shape[0]} rows, "
          f"{(tickets['text'].str.strip() == '').sum()} empty texts kept as empty strings)")

    # string load so that "N/A" / "error" / blanks are seen exactly as written
    o = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype=str, keep_default_na=False)
    c = pd.read_csv(os.path.join(RAW, "customers.csv"), dtype=str, keep_default_na=False)
    amount_num = pd.to_numeric(o["order_amount"], errors="coerce")
    deliv = o["delivery_minutes"]
    deliv_num = pd.to_numeric(deliv, errors="coerce")
    age_num = pd.to_numeric(c["age"], errors="coerce")
    figures = [
        ("raw orders.csv rows", len(o)),
        ("exact duplicate rows", int(o.duplicated().sum())),
        ("rows with status pending", int((o["status"] == "pending").sum())),
        ("distinct order_id values", int(o["order_id"].nunique())),
        ("order_amount entries not convertible to a number", int((amount_num.isna() & (o["order_amount"] != "")).sum())),
        ("negative delivery_minutes", int((deliv_num < 0).sum())),
        ("blank delivery_minutes", int((deliv == "").sum())),
        ("customers: blank age", int((c["age"] == "").sum())),
        ("customers: age outside 10-100", int(((age_num < 10) | (age_num > 100)).sum())),
        ("customers: distinct raw city spellings", int(c["city"].nunique())),
        ("customers: distinct non-blank raw gender spellings", int(c.loc[c["gender"] != "", "gender"].nunique())),
        ("customers: blank gender", int((c["gender"] == "").sum())),
    ]
    print("\n### Audit figures")
    print(md_table(pd.DataFrame(figures, columns=["figure", "value"])))
    return dict(figures)


# ----------------------------------------------------------------- customers
def clean_customers():
    heading("PART A - TASK 3: CLEAN CUSTOMERS")
    c = pd.read_csv(os.path.join(RAW, "customers.csv"), dtype={"age": "object"}, keep_default_na=False)
    log = []

    # (a) city: strip spaces + title case
    before = c["city"].copy()
    c["city"] = c["city"].str.strip().str.title()
    log.append(("city stripped + title-cased", int((before != c["city"]).sum())))

    # (b) gender -> male / female / unknown
    g = c["gender"].str.strip().str.lower()
    mapping = {"male": "male", "m": "male", "female": "female", "f": "female"}
    new_gender = g.map(mapping).fillna("unknown")
    log.append(("gender spellings mapped to male/female/unknown", int((c["gender"] != new_gender).sum())))
    c["gender"] = new_gender

    # (c) impossible ages -> missing, then fill with city median (groupby + transform)
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    impossible = (c["age"] < 10) | (c["age"] > 100)
    log.append(("ages outside 10-100 set to missing", int(impossible.sum())))
    c.loc[impossible, "age"] = float("nan")
    n_missing = int(c["age"].isna().sum())
    c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median"))
    log.append(("missing ages filled with city median", n_missing))

    # (d) signup_date -> datetime
    c["signup_date"] = pd.to_datetime(c["signup_date"], format="%Y-%m-%d")
    log.append(("signup_date converted to datetime", len(c)))

    print(md_table(pd.DataFrame(log, columns=["rule", "rows/values changed"])))
    print("\ncity counts:", c["city"].value_counts().to_dict())
    print("gender counts:", c["gender"].value_counts().to_dict())
    print("missing ages:", int(c["age"].isna().sum()), "| age range:", c["age"].min(), "-", c["age"].max())
    print("city median ages used:", c.groupby("city")["age"].median().to_dict())
    return c, log


# ----------------------------------------------------------------- orders
def clean_orders(products):
    heading("PART A - TASK 4: CLEAN ORDERS")
    o = pd.read_csv(os.path.join(RAW, "orders.csv"), dtype={"order_amount": "object"}, keep_default_na=False)
    log = []

    # 1. order_ts -> datetime
    o["order_ts"] = pd.to_datetime(o["order_ts"], format="%Y-%m-%d %H:%M:%S")
    log.append(("1. order_ts converted to datetime", len(o)))

    # 2. remove exact duplicate rows
    n = len(o)
    o = o.drop_duplicates().reset_index(drop=True)
    log.append(("2. exact duplicate rows removed", n - len(o)))

    # 3. keep the latest record per order_id (drops the earlier 'pending' copies)
    n = len(o)
    o = (o.sort_values(["order_id", "order_ts"])
          .drop_duplicates("order_id", keep="last")
          .reset_index(drop=True))
    log.append(("3. older records of a repeated order_id removed (latest order_ts kept)", n - len(o)))

    # 4. order_amount -> numeric; refill unparseable values from the product price
    o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
    o = o.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
    miss = o["order_amount"].isna()
    o.loc[miss, "order_amount"] = (o.loc[miss, "price_inr"] * o.loc[miss, "quantity"]
                                   * (1 - o.loc[miss, "discount_pct"] / 100)).round(2)
    log.append(("4. unparseable order_amount recomputed from price x quantity x (1 - discount)", int(miss.sum())))

    # 5. negative delivery -> missing; fill completed orders with category median; cancelled stay missing
    o["delivery_minutes"] = pd.to_numeric(o["delivery_minutes"], errors="coerce")
    neg = o["delivery_minutes"] < 0
    o.loc[neg, "delivery_minutes"] = float("nan")
    log.append(("5a. negative delivery_minutes set to missing", int(neg.sum())))
    completed = o["status"] == "completed"
    cat_median = o[completed].groupby("category")["delivery_minutes"].transform("median")
    fill = completed & o["delivery_minutes"].isna()
    o.loc[fill, "delivery_minutes"] = cat_median[fill]
    log.append(("5b. missing delivery_minutes of completed orders filled with category median", int(fill.sum())))
    log.append(("5c. cancelled orders left with missing delivery_minutes",
                int((~completed & o["delivery_minutes"].isna()).sum())))

    print(md_table(pd.DataFrame(log, columns=["rule", "rows/values changed"])))
    print("\ncategory medians (completed):",
          o[completed].groupby("category")["delivery_minutes"].median().to_dict())

    o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
    print("rows:", len(o), "| distinct order_id:", o["order_id"].nunique())
    print("status:", o["status"].value_counts().to_dict())
    print("missing order_amount:", int(o["order_amount"].isna().sum()))
    print("missing delivery (completed):", int(o.loc[o.status == "completed", "delivery_minutes"].isna().sum()),
          "| missing delivery (cancelled):", int(o.loc[o.status == "cancelled", "delivery_minutes"].isna().sum()))
    print("sum of completed order_amount: INR {:,.2f}".format(o.loc[o.status == "completed", "order_amount"].sum()))
    return o, log


# ----------------------------------------------------------------- store
def store(customers, products, orders):
    heading("PART A - TASK 5: STORE")
    ensure_dirs()
    c = customers.copy()
    c["signup_date"] = c["signup_date"].dt.strftime("%Y-%m-%d")
    o = orders.copy()
    o["order_ts"] = o["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    c.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
    products.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
    o.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    with sqlite3.connect(DB_PATH) as con:
        c.to_sql("customers", con, index=False)
        products.to_sql("products", con, index=False)
        o.to_sql("orders", con, index=False)
        for t in ("customers", "products", "orders"):
            print(f"  table {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]} rows")
    print("wrote data/clean/*.csv and data/instamart.db")


if __name__ == "__main__":
    audit()
    products = pd.read_csv(os.path.join(RAW, "products.csv"))
    customers, _ = clean_customers()
    orders, _ = clean_orders(products)
    store(customers, products, orders)
