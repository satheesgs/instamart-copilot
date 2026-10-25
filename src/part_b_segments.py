"""Part B, tasks 1-3: RFM table, K-means (K=2..8), segment profile + names, PCA.
Writes outputs/customer_segments.csv.   Run:  python src/part_b_segments.py"""
import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from common import CLEAN, OUTPUTS, ensure_dirs, heading, md_table

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
FINAL_K = 4
RFM_COLS = ["recency", "frequency", "monetary"]


def build_rfm():
    heading("PART B - TASK 1: RFM TABLE (snapshot 2025-10-01 00:00:00)")
    o = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    comp = o[o["status"] == "completed"]
    rfm = (comp.groupby("customer_id")
               .agg(last_order_ts=("order_ts", "max"), frequency=("order_id", "nunique"),
                    monetary=("order_amount", "sum"))
               .reset_index())
    rfm["recency"] = (SNAPSHOT - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_id"] + RFM_COLS].sort_values("customer_id").reset_index(drop=True)
    print(f"customers with >= 1 completed order: {len(rfm)}")
    desc = rfm[RFM_COLS].describe().T[["mean", "50%", "min", "max"]].reset_index()
    desc.columns = ["feature", "mean", "median", "min", "max"]
    print(md_table(desc))
    return rfm


def choose_k(X):
    heading("PART B - TASK 2: K-MEANS, K = 2..8")
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        rows.append((k, round(km.inertia_, 1), round(silhouette_score(X, km.labels_), 3)))
    table = pd.DataFrame(rows, columns=["K", "inertia", "silhouette"])
    table["inertia_drop_vs_previous"] = (-table["inertia"].diff()).round(1)
    print(md_table(table.fillna(0), floatfmt="{:,.3f}"))
    return table


def name_segments(profile):
    """Give business names by RFM behaviour (cluster ids are arbitrary labels with no ordering)."""
    names = {}
    remaining = list(profile.index)
    premium = profile["monetary"].idxmax()
    names[premium] = "Premium Big Spenders"
    remaining.remove(premium)
    at_risk = profile.loc[remaining, "recency"].idxmax()
    names[at_risk] = "Dormant / At-Risk"
    remaining.remove(at_risk)
    loyal = profile.loc[remaining, "frequency"].idxmax()
    names[loyal] = "Loyal Regulars"
    remaining.remove(loyal)
    for i, sid in enumerate(remaining):
        names[sid] = "Occasional Shoppers" if i == 0 else f"Other Segment {sid}"
    return names, premium


ACTIONS = {
    "Premium Big Spenders": "VIP care: priority (1-hour) support, early access to electronics launches, no-cost EMI offers.",
    "Dormant / At-Risk": "Win-back: personalised 'we miss you' coupon on their favourite category, free delivery on the next order.",
    "Loyal Regulars": "Reward and retain: subscription / loyalty pass for daily grocery and dairy, streak rewards.",
    "Occasional Shoppers": "Grow frequency: basket-building bundles and time-limited offers to nudge a second and third order.",
}


def main():
    ensure_dirs()
    rfm = build_rfm()
    X = StandardScaler().fit_transform(rfm[RFM_COLS])
    choose_k(X)

    heading(f"PART B - TASK 3: PROFILE OF THE FINAL K = {FINAL_K}")
    km = KMeans(n_clusters=FINAL_K, n_init=10, random_state=42).fit(X)
    rfm["segment_id"] = km.labels_
    profile = (rfm.groupby("segment_id")
                  .agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                       mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"))
                  .rename(columns={"mean_recency": "recency", "mean_frequency": "frequency",
                                   "mean_monetary": "monetary"}))
    names, premium_id = name_segments(profile)
    rfm["segment_name"] = rfm["segment_id"].map(names)
    rfm["is_premium"] = rfm["segment_id"] == premium_id
    out = profile.reset_index()
    out["segment_name"] = out["segment_id"].map(names)
    out["is_premium"] = out["segment_id"] == premium_id
    out["recommended_action"] = out["segment_name"].map(ACTIONS)
    out = out[["segment_id", "segment_name", "size", "recency", "frequency", "monetary", "is_premium",
               "recommended_action"]]
    print(md_table(out))
    print("(cluster numbers are only labels - segment 0 is not 'better' or 'worse' than segment 3)")

    pca = PCA(n_components=2).fit(X)
    r = pca.explained_variance_ratio_
    print(f"\nPCA explained_variance_ratio_: PC1 {r[0]:.3f}, PC2 {r[1]:.3f}, total {r.sum():.3f}")

    path = os.path.join(OUTPUTS, "customer_segments.csv")
    rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]] \
        .to_csv(path, index=False)
    print(f"wrote {os.path.relpath(path)} ({len(rfm)} rows, {int(rfm['is_premium'].sum())} premium)")


if __name__ == "__main__":
    main()
