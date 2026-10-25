"""Part B, tasks 4-8: lapse cohort + features, leakage-free preprocessing, tuned models,
class-balance experiment, interpretation, model persistence and the Premium retention list.
Writes outputs/lapse_model.joblib and outputs/retention_list.csv.   Run:  python src/part_b_lapse.py"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import CLEAN, OUTPUTS, ensure_dirs, heading, md_table

CUTOFF = pd.Timestamp("2025-09-01")
LABEL_END = pd.Timestamp("2025-10-01")          # label window is 2025-09-01 .. 2025-09-30
NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancel_rate",
           "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]


# ----------------------------------------------------------------- task 4
def build_cohort():
    heading("PART B - TASK 4: LAPSE COHORT, LABEL AND FEATURES (cutoff 2025-09-01)")
    o = pd.read_csv(os.path.join(CLEAN, "orders.csv"), parse_dates=["order_ts"])
    c = pd.read_csv(os.path.join(CLEAN, "customers.csv"), parse_dates=["signup_date"])
    p = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    o = o.merge(p[["product_id", "category"]], on="product_id", how="left")

    before = o[o["order_ts"] < CUTOFF]                     # the ONLY orders features may use
    comp_before = before[before["status"] == "completed"]
    cohort_ids = sorted(comp_before["customer_id"].unique())

    sept = o[(o["order_ts"] >= CUTOFF) & (o["order_ts"] < LABEL_END) & (o["status"] == "completed")]
    active_in_sept = set(sept["customer_id"])

    g = comp_before.groupby("customer_id")
    feats = pd.DataFrame({
        "recency": (CUTOFF - g["order_ts"].max()).dt.days,
        "frequency": g["order_id"].nunique(),
        "monetary": g["order_amount"].sum(),
        "avg_discount_pct": g["discount_pct"].mean(),
        "avg_delivery_minutes": g["delivery_minutes"].mean(),
        # most frequent category; ties broken alphabetically so the result is deterministic
        "fav_category": g["category"].agg(lambda s: s.value_counts().sort_index().idxmax()),
    })
    status_counts = before.groupby("customer_id")["status"].value_counts().unstack(fill_value=0)
    feats["cancel_rate"] = (status_counts["cancelled"] / status_counts.sum(axis=1)).reindex(feats.index)

    feats = feats.loc[cohort_ids].reset_index().rename(columns={"index": "customer_id"})
    feats = feats.merge(c[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id", how="left")
    feats["tenure_days"] = (CUTOFF - feats["signup_date"]).dt.days
    feats["lapsed"] = (~feats["customer_id"].isin(active_in_sept)).astype(int)
    feats = feats.drop(columns="signup_date")

    n, lapsed = len(feats), int(feats["lapsed"].sum())
    print(f"cohort: {n} customers; lapsed: {lapsed} ({lapsed / n:.2%}); not lapsed: {n - lapsed}")
    print(f"latest order date used in features: {before['order_ts'].max()} (< cutoff {CUTOFF.date()})")
    return feats


# ----------------------------------------------------------------- task 5
def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL),
    ])


def lapsed_metrics(y_true, y_pred):
    return {"precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred)}


def report_test(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    m = lapsed_metrics(y_true, y_pred)
    print(f"\n{name} - confusion matrix on the test set (rows = actual, columns = predicted):")
    print("|  | pred not lapsed | pred lapsed |\n|---|---|---|")
    print(f"| actual not lapsed (0) | {tn} | {fp} |\n| actual lapsed (1) | {fn} | {tp} |")
    print(f"lapsed class: precision {m['precision']:.3f}, recall {m['recall']:.3f}, F1 {m['f1']:.3f}; "
          f"accuracy {m['accuracy']:.3f}")
    return m


def main():
    ensure_dirs()
    data = build_cohort()
    X, y = data[["customer_id"] + NUMERIC + NOMINAL], data["lapsed"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ids_test = X_test["customer_id"]
    X_train, X_test = X_train.drop(columns="customer_id"), X_test.drop(columns="customer_id")
    print(f"train rows: {len(X_train)}, test rows: {len(X_test)}; "
          f"lapsed in train {int(y_train.sum())}, in test {int(y_test.sum())}")

    base_pred = np.zeros(len(y_test), dtype=int)
    base = lapsed_metrics(y_test, base_pred)
    print(f"baseline 'always not lapsed': accuracy {base['accuracy']:.1%}, lapsed-class F1 {base['f1']:.2f}")

    heading("PART B - TASK 5: PREPROCESSING WITHOUT LEAKAGE")
    assert X.isna().sum().sum() == 0, "a feature column contains missing values"
    print("missing values per feature column:", X.isna().sum().to_dict(), "-> none")
    pre = make_preprocessor().fit(X_train)                 # fitted on the training rows only
    scaler = pre.named_transformers_["num"]
    print(f"scaler mean of frequency (train rows only): {scaler.mean_[NUMERIC.index('frequency')]:.4f} | "
          f"full-cohort mean of frequency: {X['frequency'].mean():.4f}")
    print("one-hot columns (k-1 per variable):", list(pre.named_transformers_["cat"].get_feature_names_out(NOMINAL)))

    heading("PART B - TASK 6: TUNE AND EVALUATE")
    lr_gs = GridSearchCV(Pipeline([("pre", make_preprocessor()), ("clf", LogisticRegression(max_iter=1000))]),
                         {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
    rf_gs = GridSearchCV(Pipeline([("pre", make_preprocessor()),
                                   ("clf", RandomForestClassifier(n_estimators=100, random_state=42))]),
                         {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                         cv=5, scoring="f1").fit(X_train, y_train)
    print(f"LogisticRegression best params {lr_gs.best_params_}, best CV F1 {lr_gs.best_score_:.3f} "
          f"({len(lr_gs.cv_results_['params']) * 5} fits)")
    print(f"RandomForest best params {rf_gs.best_params_}, best CV F1 {rf_gs.best_score_:.3f} "
          f"({len(rf_gs.cv_results_['params']) * 5} fits)")
    lr, rf = lr_gs.best_estimator_, rf_gs.best_estimator_
    lr_m = report_test("Logistic regression", y_test, lr.predict(X_test))
    rf_m = report_test("Random forest", y_test, rf.predict(X_test))
    selected_name, selected = ("Random forest", rf) if rf_m["f1"] >= lr_m["f1"] else ("Logistic regression", lr)
    print(f"\nselected model: {selected_name}")

    heading("PART B - TASK 7: CLASS-BALANCE EXPERIMENT (over-sample training rows only)")
    train = X_train.assign(lapsed=y_train.values)
    minority, majority = train[train["lapsed"] == 1], train[train["lapsed"] == 0]
    extra = minority.sample(n=len(majority) - len(minority), replace=True, random_state=42)
    balanced = pd.concat([train, extra])
    print(f"balanced training set: {int((balanced['lapsed'] == 0).sum())} not lapsed / "
          f"{int((balanced['lapsed'] == 1).sum())} lapsed")
    lr_bal = Pipeline([("pre", make_preprocessor()),
                       ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))])
    lr_bal.fit(balanced.drop(columns="lapsed"), balanced["lapsed"])
    bal_m = lapsed_metrics(y_test, lr_bal.predict(X_test))
    comp = pd.DataFrame([("logistic regression (original train)", lr_m["precision"], lr_m["recall"], lr_m["f1"]),
                         ("logistic regression (balanced train)", bal_m["precision"], bal_m["recall"], bal_m["f1"])],
                        columns=["model", "precision", "recall", "f1"])
    print(md_table(comp, floatfmt="{:.3f}"))

    heading("PART B - TASK 8: INTERPRET, PERSIST AND USE")
    rf_names = rf.named_steps["pre"].get_feature_names_out()
    imp = (pd.DataFrame({"feature": rf_names, "importance": rf.named_steps["clf"].feature_importances_})
             .sort_values("importance", ascending=False).head(5))
    print("random forest - top 5 feature importances:")
    print(md_table(imp, floatfmt="{:.4f}"))
    coefs = pd.DataFrame({"feature": lr.named_steps["pre"].get_feature_names_out(),
                          "coefficient": lr.named_steps["clf"].coef_[0]})
    coefs["sign"] = np.where(coefs["coefficient"] > 0, "+ (raises lapse risk)", "- (lowers lapse risk)")
    print("\nlogistic regression coefficients (standardised features):")
    print(md_table(coefs.reindex(coefs["coefficient"].abs().sort_values(ascending=False).index), floatfmt="{:.3f}"))

    path = os.path.join(OUTPUTS, "lapse_model.joblib")
    joblib.dump(selected, path)
    reloaded = joblib.load(path)
    same = np.array_equal(reloaded.predict(X_test), selected.predict(X_test)) and \
        np.allclose(reloaded.predict_proba(X_test), selected.predict_proba(X_test))
    print(f"\nsaved {os.path.relpath(path)}; reloaded model reproduces the test predictions exactly: {same}")
    assert same

    seg = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
    premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
    test_df = pd.DataFrame({"customer_id": ids_test.values,
                            "lapse_probability": reloaded.predict_proba(X_test)[:, 1].round(4),
                            "actually_lapsed": y_test.values})
    retention = (test_df[test_df["customer_id"].isin(premium_ids)]
                 .sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10))
    retention.to_csv(os.path.join(OUTPUTS, "retention_list.csv"), index=False)
    n_prem_test = int(test_df["customer_id"].isin(premium_ids).sum())
    print(f"\nPremium customers in the test set: {n_prem_test}; top 10 by lapse probability:")
    print(md_table(retention, floatfmt="{:.4f}"))
    print(f"truly lapsed among them: {int(retention['actually_lapsed'].sum())} of {len(retention)}")
    print("NOTE: the Premium flag is a post-hoc whole-window label (its RFM includes September), "
          "so this list is illustrative, not a leakage-free evaluation.")


if __name__ == "__main__":
    main()
