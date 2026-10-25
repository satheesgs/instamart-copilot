# Instamart Customer Intelligence & Support Copilot

## Overview

An internal Copilot for the (fictional) Swiggy Instamart Customer Experience team, built as one repository in four connected Parts. All data is synthetic, produced by the supplied `generate_data.py`; every language-model call goes through the supplied offline `mock_llm.py` (no API key, no network, no cost). All monetary values are in Indian Rupees (INR).

| Part | What it does | Consumes | Produces |
|---|---|---|---|
| A. Data Foundation | Audits the dirty exports, cleans them with documented rules, stores them in SQLite, answers 7 SQL questions and a pandas EDA | `data/raw/*.csv` | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B. Segments and Lapse Prediction | RFM + K-means segments (with a Premium flag) and a leakage-free lapse classifier | clean orders and customers | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C. Ticket Triage | Input guardrails, PII redaction, retries, output validation, token budget, routing by segment | `tickets.csv`, segment file from B | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D. Policy RAG + Refund Agent | TF-IDF retrieval over 5 policy docs, grounded answers, a ReAct refund agent with tools, safe stops and crash-resume | policies, `instamart.db` from A | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

Repository layout

```
generate_data.py, mock_llm.py   supplied files (unmodified)
requirements.txt, .gitignore, README.md
src/common.py                    shared paths + markdown-table helper
src/part_a_clean.py              Part A tasks 2-5 (audit, clean, store)
src/part_a_validate.py           Part A task 5 (prints ALL CHECKPOINTS PASSED)
src/part_a_analysis.py           Part A tasks 6-7 (SQL + pandas)
src/part_b_segments.py           Part B tasks 1-3
src/part_b_lapse.py              Part B tasks 4-8
src/part_c_triage.py             Part C tasks 1-7
src/part_d_rag.py                Part D tasks 1-4
src/part_d_agent.py              Part D tasks 5-8
sql/queries.sql                  the seven SQL queries
data/  outputs/  state/          generated artefacts (committed)
```

Environment used: Python 3.12.4, SQLite 3.45.3, pandas 3.1.0, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0 (exact pins in `requirements.txt`). Only the standard library, pandas, numpy, scikit-learn and joblib are imported.

## Reproduce

Run from a clean clone, in this order (Python 3.11 or newer):

1. `git clone https://github.com/satheesgs/instamart-copilot.git && cd instamart-copilot`
2. `python -m venv .venv`
3. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
4. `pip install -r requirements.txt`
5. `python generate_data.py` → last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`
6. `python src/part_a_clean.py`
7. `python src/part_a_validate.py` → `ALL CHECKPOINTS PASSED`
8. `python src/part_a_analysis.py`
9. `python src/part_b_segments.py`
10. `python src/part_b_lapse.py`
11. `python src/part_c_triage.py`
12. `python src/part_d_rag.py`
13. `python src/part_d_agent.py`

Each script prints every number reported below. `part_d_agent.py` clears `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl` and `state/*.json` at the start, so a re-run gives the same files (only ledger timestamps change).

## Part A

### Audit of the raw data

Shape, data types and missing values (standard `pd.read_csv`; note that pandas reads the literal `N/A` as missing, which is why `order_amount` shows 53 missing although no cell is blank). `tickets.csv` is always read with `keep_default_na=False` so its 4 empty texts stay empty strings.

customers.csv: 1,200 rows × 5 columns

| column | dtype | missing | missing % |
|---|---|---|---|
| customer_id | str | 0 | 0.00 |
| city | str | 0 | 0.00 |
| signup_date | str | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | str | 30 | 2.50 |

products.csv: 40 rows × 4 columns, no missing values (product_id, product_name, category: str; price_inr: int64).

orders.csv: 13,084 rows × 9 columns

| column | dtype | missing | missing % |
|---|---|---|---|
| order_id, customer_id, product_id, order_ts, status | str | 0 | 0.00 |
| quantity, discount_pct | int64 | 0 | 0.00 |
| order_amount | str | 53 | 0.41 |
| delivery_minutes | float64 | 2,541 | 19.42 |

Audit figures (raw files read as text so that every spelling is seen as written):

| figure | value |
|---|---|
| raw orders.csv rows | 13,084 |
| exact duplicate rows | 119 |
| rows with status pending | 656 |
| distinct order_id values | 12,309 |
| order_amount entries not convertible to a number | 120 |
| negative delivery_minutes | 106 |
| blank delivery_minutes | 2,541 |
| customers: blank age | 61 |
| customers: age outside 10–100 | 6 |
| customers: distinct raw city spellings | 24 |
| customers: distinct non-blank raw gender spellings | 10 |
| customers: blank gender | 30 |

### Cleaning rules (with the number of rows or values each rule changed)

Customers

| rule | rows/values changed |
|---|---|
| (a) city: strip leading/trailing spaces, then title case | 476 |
| (b) gender: male/m → male, female/f → female (case-insensitive), blank → unknown | 950 |
| (c1) ages outside 10–100 set to missing | 6 |
| (c2) all missing ages (61 blank + 6 impossible) filled with the city median age (`groupby("city").transform("median")`) | 67 |
| (d) signup_date converted to datetime | 1,200 |

City medians used: Bengaluru 42, Chennai 39, Delhi 43, Hyderabad 37.5, Mumbai 38.5, Pune 39. Result: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, all ages between 18 and 60.

Orders (applied in this order)

| rule | rows/values changed |
|---|---|
| 1. order_ts converted to datetime | 13,084 |
| 2. exact duplicate rows removed | 119 |
| 3. for each order_id keep only the record with the latest order_ts (drops the earlier `pending` copies) | 656 |
| 4. order_amount → numeric; unparseable values set missing then recomputed as round(price_inr × quantity × (1 − discount_pct/100), 2) | 112 (120 raw entries, 8 of which sat on rows already removed in steps 2–3) |
| 5a. negative delivery_minutes set to missing | 105 (106 raw, 1 was on a removed duplicate) |
| 5b. missing delivery_minutes of completed orders filled with the median of that product category among completed orders (`groupby("category").transform("median")`) | 774 |
| 5c. cancelled orders: delivery_minutes deliberately left missing (never delivered, imputing would invent data) | 1,199 left missing |

Category medians used: Dairy 25, Electronics 25, Grocery 25, Personal Care 25.5, Snacks 25 minutes.

Result after cleaning: 12,309 rows (one per order_id); 11,110 completed and 1,199 cancelled; no pending rows; 0 missing order_amount; 0 missing delivery_minutes among completed orders and 1,199 missing among cancelled orders; sum of completed order_amount INR 2,998,668.80.

### Store and validate

Clean tables are written to `data/clean/customers.csv`, `products.csv`, `orders.csv` and to `data/instamart.db` (tables `customers` 1,200 rows, `products` 40, `orders` 12,309; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). `src/part_a_validate.py` asserts 16 checkpoints (city counts, gender counts, no missing ages, ages 18–60, 12,309 unique orders, 11,110 completed, 1,199 cancelled, no pending, no missing amount, no missing completed delivery, 1,199 missing cancelled delivery, no negative delivery, revenue INR 2,998,668.80, database row counts, timestamp text format, database revenue) and prints `ALL CHECKPOINTS PASSED`.

### SQL analysis (`sql/queries.sql`, run with `pd.read_sql_query`)

| query | technique | result |
|---|---|---|
| Q1 | WHERE (completed, amount > 500) then HAVING (total > 8,000) | 30 customers |
| Q2 | three-table join, revenue by city × category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 |
| Q3 | scalar subquery, price above average catalogue price | 11 products |
| Q4 | LEFT JOIN, customers with no completed order | 25 customers |
| Q5 | two CTEs, spend above the average customer spend | 550 customers; average INR 2,552.06 |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER over (PARTITION BY city ORDER BY total_spend DESC) | top customers below; 8 rows with RANK() ≠ ROW_NUMBER() |
| Q7 | running total SUM(revenue) OVER (ORDER BY day) | 183 daily rows; last cumulative value INR 2,998,668.80 |

Q6 top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

Why ties cause the 8 differences: when two or more customers in the same city have exactly the same total spend, RANK() gives them all the same number while ROW_NUMBER() must keep counting 1, 2, 3, so every tied customer after the first gets a RANK lower than its ROW_NUMBER (the 8 rows are Bengaluru C0734, C0723; Mumbai C1053, C0608; Pune C1064, C0057, C0936, C0276).

### pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` of completed revenue by city × category gives a grand total of INR 2,998,668.80, identical to the SQL Q2 total.

Per city, completed order_amount (groupby + named aggregation):

| city | sum (INR) | mean (INR) | count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed order_amount distribution:

| statistic | value |
|---|---|
| mean | INR 269.91 |
| median | INR 135.00 |
| skewness | 4.61 |
| Q1 / Q2 / Q3 | INR 64.80 / 135.00 / 297.00 |
| IQR | INR 232.20 |
| lower fence / upper fence (1.5 × IQR) | INR −283.50 / INR 645.30 |
| outliers below lower fence / above upper fence | 0 / 897 |

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

**What is this data about?** Six months (April–September 2025) of synthetic quick-commerce orders: 12,309 orders by 1,200 customers in six cities over 40 products in five categories; 11,110 completed orders earned INR 2,998,668.80.

**What do the distributions look like?** Order value is strongly right-skewed: mean INR 269.91 against median INR 135.00, skewness 4.61; half of all orders lie between INR 64.80 and INR 297.00. The median delivery time is 25 minutes.

**Outliers or anomalies?** 897 orders exceed the upper fence of INR 645.30 and none fall below. They are genuine high-value baskets (635 Electronics, the rest multi-unit Ghee, Atta or lotion), so they are kept. The true anomalies (duplicates, pending copies, "N/A"/"error" amounts, negative times) were fixed during cleaning.

**Relationships?** All correlations are weak. The largest in absolute value is order_amount with discount_pct (−0.095), slightly ahead of order_amount with quantity (+0.087); delivery time is unrelated to the others. A correlation this small does not establish causation.

## Part B

### RFM table (snapshot 2025-10-01 00:00:00, completed orders only)

1,175 customers have at least one completed order.

| feature | mean | median | min | max |
|---|---|---|---|---|
| recency (days) | 33.56 | 19 | 0 | 180 |
| frequency (orders) | 9.46 | 5 | 1 | 33 |
| monetary (INR) | 2,552.06 | 2,286.50 | 12.00 | 14,766.55 |

### K-means on the standardised RFM matrix (`KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | inertia | silhouette | inertia drop vs K−1 |
|---|---|---|---|
| 2 | 1,820.8 | 0.465 | – |
| 3 | 1,268.2 | 0.445 | 552.6 |
| 4 | 743.1 | 0.526 | 525.1 |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Chosen K = 4.** The elbow is at K = 4: going from 3 to 4 still removes 525 units of inertia, but from 4 to 5 only 126. K = 4 also has the highest silhouette (0.526), and four groups map cleanly onto four distinct campaigns (premium care, loyalty, frequency growth, win-back).

Why inertia alone cannot pick K: inertia is the total squared distance of points to their own centroid, and adding a centroid can never increase it (the K = 8 solution can always place centroids at least as close as the K = 2 one), so it keeps falling until K equals the number of customers. Minimising inertia would therefore always choose the largest K; we need the elbow, the silhouette (which penalises splitting a natural group) and business usefulness to decide.

### Segment profile (K = 4). Cluster numbers are only labels with no ordering.

| segment_id | segment_name | size | mean recency (days) | mean frequency | mean monetary (INR) | is_premium | recommended action |
|---|---|---|---|---|---|---|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | False | Grow frequency: basket-building bundles and time-limited offers to nudge a second and third order |
| 1 | Premium Big Spenders | 138 | 24.12 | 6.77 | 7,129.70 | **True** | VIP care: priority 1-hour support, early access to electronics launches, no-cost EMI offers |
| 2 | Loyal Regulars | 419 | 8.77 | 19.40 | 3,410.02 | False | Reward and retain: subscription / loyalty pass for daily grocery and dairy, streak rewards |
| 3 | Dormant / At-Risk | 182 | 109.04 | 2.20 | 685.99 | False | Win-back: personalised coupon on their favourite category, free delivery on the next order |

Names are assigned by rule in code (highest mean monetary → Premium; highest remaining mean recency → Dormant; highest remaining frequency → Loyal; the rest → Occasional), so they stay correct even if cluster ids change.

PCA (2 components) on the scaled RFM matrix: explained_variance_ratio_ = 0.634 and 0.218, total 0.852.

`outputs/customer_segments.csv`: customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium (1,175 rows, 138 premium).

### Lapse problem

- Cutoff 2025-09-01. Cohort = customers with at least one completed order before the cutoff: **1,156 customers**.
- Label `lapsed` = 1 if no completed order from 2025-09-01 to 2025-09-30: **443 lapsed (38.32%)**, 713 not lapsed.
- Features (only orders dated before the cutoff, plus the clean customer table): recency, frequency, monetary, avg_discount_pct, cancel_rate (cancelled ÷ completed + cancelled), avg_delivery_minutes, fav_category (most frequent category, ties broken alphabetically), city, age, gender, tenure_days (cutoff − signup_date). The latest order used in any feature is 2025-08-31 22:35:45.
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`: **924 train / 232 test** rows (354 / 89 lapsed).
- Baseline "always not lapsed" on the test set: **accuracy 61.6%, lapsed-class F1 0.00**.

### Preprocessing without leakage

A `ColumnTransformer` inside a scikit-learn `Pipeline`: `StandardScaler` on the 8 numeric features and `OneHotEncoder(drop="first")` (k−1 columns: 5 city, 2 gender, 4 category dummies) on the nominal ones. The pipeline is fitted on training rows only and merely applied to test rows; inside GridSearchCV it is re-fitted on each training fold only.

- Scaler mean of frequency learned on training rows: **8.1147**; full-cohort mean of frequency: **8.0285**. They differ because the scaler only saw the 924 training customers, whose average differs slightly from that of all 1,156; that difference is exactly the proof that the test rows did not influence the scaler.
- An assertion in code confirms that no feature column contains a missing value (all 12 columns: 0 missing).

**Leakage statement:** no encoder, scaler or resampler saw a test row; no order dated 2025-09-01 or later entered any feature; and the segment labels from task 3 were not used as features.

### Tuning and evaluation

| model | grid | best parameters | best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100}, cv=5 (25 fits) | C = 100 | 0.651 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20}, cv=5 (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

Logistic regression CV F1 is flat from C = 0.1 to 100 (0.646, 0.645, 0.649, 0.651), so C = 100 rather than the reference C = 1 is a difference of 0.006 in F1 coming from our richer feature set, not a different model behaviour.

Test set (232 rows), lapsed class:

| model | TN | FP | FN | TP | precision | recall | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|
| baseline (always 0) | 143 | 0 | 89 | 0 | 0.000 | 0.000 | 0.000 | 0.616 |
| logistic regression (C=100) | 111 | 32 | 31 | 58 | 0.644 | 0.652 | 0.648 | 0.728 |
| random forest (depth 8, leaf 10) | 113 | 30 | 31 | 58 | 0.659 | 0.652 | **0.655** | 0.737 |

**Selected model: random forest.** Both tuned models catch the same 58 of the 89 customers who really lapsed (recall 0.652), so they are equally good at the thing that matters most: a missed lapser silently costs their future basket (the average customer spends about INR 2,552 over six months). The random forest raises two fewer false alarms (30 vs 32), and a false alarm only costs a coupon or a message sent to a loyal customer, which is cheap but not free. It also has the higher cross-validated F1 (0.681 vs 0.651) and the higher test F1 (0.655, above the required 0.55). Because missing a lapser is costlier than a wasted coupon, in production we would lower its decision threshold (or use the balancing below) to trade some precision for more recall.

### Class-balance experiment (training set only)

The training set was over-sampled by adding lapsed rows drawn with replacement (`random_state=42`) until both classes had **570** rows; the best logistic regression (C = 100) was refitted and evaluated on the original, unchanged test set.

| logistic regression | precision | recall | F1 |
|---|---|---|---|
| original training set | 0.644 | 0.652 | 0.648 |
| balanced training set (570 / 570) | 0.589 | **0.854** | 0.697 |

Recall moved most (0.65 → 0.85): the retention team now reaches about 76 of the 89 customers who will lapse instead of 58. Precision did not rise; it fell slightly to about 0.59, so roughly 4 of every 10 customers targeted would have stayed anyway, which is acceptable when a retention offer is cheap compared with a lost customer.

### Interpretation

Random forest, top 5 feature importances:

| feature | importance | business reading |
|---|---|---|
| frequency | 0.280 | How often someone ordered before September is the strongest signal of whether they keep ordering |
| monetary | 0.271 | Total spend separates habitual basket-fillers from one-off buyers |
| recency | 0.115 | Customers who have not ordered for weeks are already drifting away |
| avg_delivery_minutes | 0.061 | Slow experiences carry some weight in whether customers come back |
| cancel_rate | 0.052 | A history of cancellations adds a little risk information |

Logistic regression coefficient signs (standardised features):

| feature | coefficient | reading |
|---|---|---|
| frequency | −1.698 (negative) | More orders before the cutoff → much lower lapse risk; habit is the best retention |
| monetary | +0.285 (positive) | At the same frequency, higher spend → higher risk: big infrequent electronics baskets do not create a habit |
| recency | +0.040 (positive) | More days since the last order → slightly higher risk (most of its effect is already captured by frequency) |

Persistence: the selected random forest pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib`; after `joblib.load` it reproduces the original test predictions and probabilities exactly (checked with `np.array_equal`, `True`).

### Premium retention list (`outputs/retention_list.csv`)

There are 40 Premium-segment customers in the test set. The top 10 by predicted lapse probability:

| customer_id | lapse_probability | actually_lapsed |
|---|---|---|
| C0862 | 0.4910 | 0 |
| C1112 | 0.4841 | 0 |
| C0060 | 0.4798 | 0 |
| C0802 | 0.4796 | 0 |
| C1008 | 0.4778 | 0 |
| C1000 | 0.4624 | 0 |
| C0525 | 0.4597 | 1 |
| C0133 | 0.4562 | 0 |
| C0267 | 0.4525 | 0 |
| C0167 | 0.4522 | 0 |

1 of the 10 truly lapsed. The Premium flag comes from the whole-window RFM, which includes the September orders that define the lapse label, so it is a post-hoc whole-window label: this retention list is illustrative and not a leakage-free evaluation. (The model's own features still use only orders before 2025-09-01.)

## Part C

### Input guardrails

Checked before any model call, in file order: 4 empty tickets (T0157–T0160) get `skipped_empty`; 4 tickets with `est_tokens(text) > 400` (T0161–T0164, each estimated at **734 tokens**) get `flagged_long`. Neither kind is ever sent to the model; 156 tickets are eligible.

### PII redaction

Applied in this order: email `[\w.+-]+@[\w-]+\.[\w.]+` → `[EMAIL]`, phone `\b[6-9]\d{9}\b` → `[PHONE]`, order id `ORD-\d{6}` → `[ORDER_ID]`.

| type | substitutions (all 164 tickets) |
|---|---|
| email | 55 |
| phone | 91 |
| order id | 132 |
| **total** | **278** |

Re-applying the three patterns to the redacted texts finds 0 matches (asserted in code). Before every model call an assertion checks that no message contains any of the three patterns, and the user message content is exactly the redacted ticket text. Example: `...linked to order ORD-105035. I did not place it. Call 7766326255.` → `...linked to order [ORDER_ID]. I did not place it. Call [PHONE].`

### Prompts

Both system prompts contain `CLASSIFY`, list billing, delivery, technical and fraud with a one-line definition each, and demand a JSON-only reply `{"category": "...", "reason": "..."}`. The zero-shot prompt (115 estimated tokens) has no `EXAMPLES:`. The few-shot prompt (229 estimated tokens) adds an `EXAMPLES:` line followed by four labelled, already-redacted examples (fraud, technical, billing, delivery).

### Resilient calls

Every call goes through `call_with_retry`: at most 3 retries per ticket; the wait before retry n (n = 0, 1, 2) is `0.05 × 2**n × random.uniform(0.9, 1.15)` seconds, and each retry logs one line. `mock_llm.reset()` is called (and `random.seed(42)` set) immediately before each full run. A full run over the 156 eligible tickets raised exactly **19 TransientLLMErrors, all recovered on the first retry**, with 0 permanent errors. Every wait was between 0.045 and 0.0575 s (asserted). First lines of the log:

```
retry ticket=T0002 retry_number=0 wait=0.0530s error='503 server busy'
retry ticket=T0116 retry_number=0 wait=0.0453s error='503 server busy'
retry ticket=T0153 retry_number=0 wait=0.0484s error='503 server busy'
... (19 lines in total)
```

### Output validation

1. The reply parses as JSON and `category` is one of the four → `classified_json`.
2. Not valid JSON → words extracted with `re.findall(r"[a-z]+", text.lower())`; exactly one allowed category word present → `classified_keyword` (this is how `billing-related` becomes `billing`).
3. Anything else, including valid JSON with a category outside the list (`"shipping"`) → `flagged_invalid_output`, routed to human review. We never guess.

Full run over the 156 eligible tickets (identical for both prompts): **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output.**

### Zero-shot vs few-shot (no token budget)

| prompt | classified | correct | accuracy | total tokens | tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | **85.7%** | 22,348 | 152.0 |
| few-shot | 147 | 147 | **100.0%** | 40,132 | 273.0 |

Zero-shot confusion table (rows = true category, columns = predicted):

| true \ predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

When the few-shot tokens are worth paying for: the few-shot prompt costs 121 more tokens per classified ticket (273 vs 152, +80%) and 17,784 more tokens for the run, about 847 extra tokens for each of the 21 tickets it corrects. All 21 zero-shot errors are fraud or technical tickets mislabelled as billing, and 11 of them are fraud reports that would have gone to the normal billing queue instead of the security team within 2 hours. Because a misrouted fraud ticket costs far more than a few hundred tokens, few-shot is worth it for this mix. Zero-shot would only be acceptable for a cheap first pass on low-risk categories, with a human checking anything that mentions money.

### Token budget and routing

(a) Budget demo (`outputs/triage_budget_demo.csv`): zero-shot with **TOKEN_BUDGET = 6000**. 42 tickets were answered (6,034 tokens) before the running total reached the budget; **114 tickets ended `budget_exhausted`** and went to human review.

(b) Final triage (`outputs/triage_results.csv`, 164 rows): few-shot run with TOKEN_BUDGET = 100000 (40,132 tokens used). Routing rules, applied in this order:

1. Status skipped_empty, flagged_long, flagged_invalid_output or budget_exhausted → human_review, NORMAL.
2. Category fraud → security_team, HIGH.
3. Customer is Premium in `outputs/customer_segments.csv` → premium_desk, HIGH.
4. Otherwise → support_queue, NORMAL.

A customer with no row in the segment file counts as not premium.

| route | priority | tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

What was rule-validated and what went to humans: the 147 classified tickets were validated by rule, 135 by strict JSON-schema validation and 12 by the single-category keyword rule. The 17 human-review tickets are 4 empty, 4 too long and 9 whose model output failed validation. No category was guessed.

## Part D

### Chunking

40-word chunks with a 10-word overlap (each chunk starts 30 words after the previous one; stop after the first chunk that reaches the last word). The source file name is stored with each chunk.

| source | chunks |
|---|---|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **total** | **15** |

### Retrieval

`TfidfVectorizer()` is fitted on the 15 chunk texts; a query is transformed with the same vectorizer and chunks are ranked by `cosine_similarity`. `retrieve(query, k=3)` returns `[{source, score, text}]`. For `refund policy late delivery` the top three are all from `refund_policy.txt` (scores 0.272, 0.251, 0.245).

TF (term frequency) measures how often a word occurs in a chunk, so a chunk that repeats "refund" scores higher for refund questions. IDF (inverse document frequency) down-weights words that occur in many chunks (like "order") and up-weights rare, distinctive words (like "wallet" or "OTP"). A limitation is that TF-IDF only matches exact word forms: "money back" and "refund", or "how long until I am paid" and "within 5 working days", share no words and are not recognised as having the same meaning.

### Grounded answering

`answer(question)` retrieves 3 chunks and sends two messages: a system message with `POLICY ASSISTANT` that tells the model to answer only from the context and otherwise say it could not find the information, and a user message of the form `CONTEXT:\n[source.txt]\n<chunk>\n...\nQUESTION:\n<question>`. It returns the model text and the sources. Example: "Who must approve a refund above INR 1000?" → "A refund above INR 1000 must be approved by a human support lead before money is returned." (sources: refund_policy.txt ×3).

### Evaluation on `policy_eval.csv`

| metric | overlap 10 (15 chunks) | overlap 0 (13 chunks) |
|---|---|---|
| answerable: expected_doc in top 3 | 12 / 12 | 12 / 12 |
| answerable: answer contains expected_fact | **12 / 12** | **10 / 12** |
| unanswerable: fallback "I could not find…" | 2 / 2 | 2 / 2 |

Questions that break with overlap 0, both because the answer sentence is cut across a chunk boundary:

- "What delivery time makes a completed order eligible for a full refund?" The sentence is split as `...delivered in more than 45 minutes, the customer` | `receives a full refund of the order amount.` The second half becomes the best-matching "sentence", so the answer loses "more than 45 minutes".
- "How soon is money returned when a payment fails but is deducted?" The split is `...the amount is returned automatically` | `within 48 hours.` The answer stops before "48 hours".

With a 10-word overlap, each of these sentences appears whole in at least one chunk.

### Tools

All five tools are in `TOOLS_SPEC` (name, one-sentence description, JSON-style parameters) and in `TOOL_MAP` (name → Python function). The model asks for a tool by name with an arguments dict, and the loop calls `TOOL_MAP[name](**arguments)`.

| tool | behaviour |
|---|---|
| get_order(order_id) | reads `data/instamart.db` → {order_id, status, order_amount, delivery_minutes} or {"error": "order not found"} |
| search_policy(query) | {"results": retrieve(query)} reusing the Part D retrieval |
| calculate_refund(order_amount, delivery_minutes) | delivered in more than 45 min → 100% of order_amount, otherwise 0 |
| issue_refund(order_id, refund_inr) | appends {timestamp, tool, run_id, order_id, refund_inr} to `outputs/refund_ledger.jsonl` |
| escalate_to_human(order_id, refund_inr) | same ledger line with tool escalate_to_human (used for refunds above INR 1000) |

### ReAct loop: the seven scenarios (max_iterations=6, token_limit=6000, time_limit_seconds=10)

| scenario | expected | status | refund (INR) | model calls | tokens used | tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND | REFUND | 128.25 | 5 | 1,192 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | REFUND | 135.00 | 5 | 1,192 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | ESCALATE | 1,999.00 | 5 | 1,179 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | DENIED | 0.00 | 4 | 841 | get_order → search_policy → calculate_refund |
| S5 | DENIED | DENIED | 0.00 | 3 | 514 | get_order → search_policy |
| S6 | NOT_FOUND | NOT_FOUND | 0.00 | 2 | 190 | get_order |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,354 | search_policy × 6 |

Every row of `outputs/agent_runs.jsonl` holds scenario_id, status, final_answer, llm_calls, tokens_used and each step's tool name, arguments and observation. The file also has the three demonstration runs below, each tagged with `run_label`.

### Stop safely

| demonstration | status | detail |
|---|---|---|
| S7 with defaults | **FALLBACK_MAX_ITERATIONS** | 6 model calls without a final answer, 3,354 tokens (under 6,000, so the token limit did not stop it early) |
| S1 with token_limit=200 | **FALLBACK_TOKEN_LIMIT** | stopped after call 2 (220 tokens) without executing the requested search_policy tool |
| S1 with time_limit_seconds=0 | **FALLBACK_TIMEOUT** | stopped at the top of the first iteration, before any model call |

Each fallback returns the steps completed so far, plus a human hand-off summary in final_answer. For example: "HANDOFF TO HUMAN: token limit 200 exceeded. Request: '…'. Completed 1 tool step(s) (get_order) in 2 model call(s), 220 tokens. Please review and finish this case manually."

### Crash and resume (S1)

After every tool observation a checkpoint `state/<run_id>.json` is written. It holds the run id, scenario id, full message list, tokens_used, llm_calls, step number, tools executed and steps. A `SimulatedCrash` is raised right after the second tool's checkpoint is saved, and `resume_agent("S1_crash_resume_demo")` loads it and continues.

Tools executed across the crash and the resume, in order:

1. get_order (before the crash)
2. search_policy (before the crash; checkpoint saved, then the crash)
3. calculate_refund (after the resume)
4. issue_refund (after the resume)

That is four executions with none repeated. Final status **REFUND, INR 128.25** ("REFUND: INR 128.25 issued.").

`outputs/refund_ledger.jsonl` contains 4 actions: issue_refund ORD-100048 INR 128.25 (S1), issue_refund ORD-100067 INR 135.00 (S2), escalate_to_human ORD-100145 INR 1,999.00 (S3), and issue_refund ORD-100048 INR 128.25 from the crash-resume demo.

### ReAct versus plan-and-execute

Our ReAct agent calls the model once per step and lets it see every observation: S1 took 5 calls (1,192 tokens), S5 took 3 and S6 took 2. Token use grows with each call because the whole history is re-sent (S7 used 3,354 tokens over 6 calls). A plan-and-execute design would ask once for the full plan (get_order → search_policy → calculate_refund → issue_refund), run the tools without the model, and call it again only to replan on an error. On the happy path that is about 2 calls instead of 5, with far fewer tokens.

The weakness of plan-and-execute is unexpected observations. When get_order returns NOT_FOUND (S6), or the order turns out to be cancelled (S5), the fixed plan would still try to calculate and issue a refund unless every such branch is coded as a replanning trigger. ReAct simply reads the observation and stops after 2 or 3 calls with the correct DENIED or NOT_FOUND. Refunds move real money and the branches (not found, cancelled, on time, above INR 1000) are decided by data the agent only sees mid-run.

For this workflow I would choose ReAct, guarded by the step, token and time limits, the INR 1000 escalation and the checkpoints we already have. It costs a few more calls per request, but it never issues a refund the observations do not support. Plan-and-execute would suit high-volume, uniform requests where the extra calls dominate cost.

## Limitations and Next Steps

- **The Premium flag is a post-hoc whole-window label.** It comes from RFM over April–September, which includes the September window that defines the lapse label, so the Premium retention list is illustrative and not a leakage-free evaluation. A production version should compute segments as of the cutoff date.
- The data is synthetic, built from three customer archetypes with Poisson ordering, so the segments and the lapse model are much cleaner than real behaviour; results will not transfer directly.
- The lapse model uses one cutoff and a 232-row test set. Its F1 (0.655) has wide uncertainty; rolling cutoffs (time-based cross-validation) and threshold tuning against a cost matrix would give a more robust selection.
- The mock LLM is a deterministic keyword classifier, so the zero-shot vs few-shot gap shows the pipeline mechanics, not real prompt quality. A real model would need the same validation, retries and budget, plus evaluation on more varied text.
- The PII regexes cover only emails, 10-digit Indian mobiles and order ids. Names, addresses, UPI ids or card numbers would pass through; a broader detector is needed.
- TF-IDF retrieval misses synonyms and paraphrases; embedding-based retrieval or hybrid search would help, at the cost of a model dependency.
- The refund agent does not check the 7-day request window (all requests are assumed in time) and has no idempotency key. The crash-resume demo re-issues S1's refund as a separate run, so a real system should refuse a second refund for the same order_id.
- Next steps: schedule the pipeline daily, add unit tests for each cleaning rule and tool, monitor retry/flag rates and token spend per day, and send human-review tickets back as labelled data.
