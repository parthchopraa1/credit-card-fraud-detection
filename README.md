# Credit-Card Fraud Detection

Behavioural anomaly detection on the classic ULB credit-card fraud dataset —
284,807 transactions, 492 frauds (0.17%). The same skill set as fraud
analysis and game-security (anti-cheat) work: find the transaction that
doesn't look like the account that made it.

## What it does

- `train.py` — loads the data, prints an EDA summary, trains and compares
  six approaches (logistic regression, logistic + SMOTE, random forest,
  XGBoost, Optuna-tuned XGBoost, isolation forest) on a **time-ordered
  80/20 split** (first 80% by `Time` = train, last 20% = test — no
  shuffling, so the model can't peek at the future), adds 5-fold
  stratified cross-validation for the three main supervised models, tunes
  XGBoost with Optuna (25 trials), logs every trial to MLflow
  (`mlruns/`), and saves the best model (by PR-AUC) plus `metrics.json`.
- `app.py` — Streamlit dashboard: upload a CSV of transactions (or use the
  built-in demo sample), set a flag threshold, and see flagged transactions,
  the confusion matrix, the precision–recall curve and feature importances.
- The project is honest about the one number that matters here: **accuracy is
  meaningless at 0.17% fraud** — a model that predicts "legitimate" for
  everything is 99.87% accurate on the held-out test set and catches
  nothing.

## Dataset

The ULB / Worldline credit-card fraud set (Kaggle `mlg-ulb/creditcardfraud`).
Downloaded from the public TensorFlow-hosted mirror of the same file:

- https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
- Canonical: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

`Time` (seconds since first transaction), `V1`–`V28` (PCA-anonymised
features), `Amount`, `Class` (0 = legitimate, 1 = fraud). Verified locally:
284,807 rows, 492 frauds. The 144 MB file is not committed — see
`data/README.md`. Citation: Dal Pozzolo et al. (2015), *Calibrating
Probability with Undersampling for Unbalanced Classification*, IEEE SSCI.

## Results (held-out test set — last 20% by Time, 56,962 rows, 75 frauds)

<!-- METRICS: measured by train.py run on 2026-11-05, time-ordered test set -->

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV PR-AUC (5-fold, train) |
| ----- | --------- | ------ | -- | ------- | ------ | ------------------------- |
| Dummy (predict all legit) | 0.000 | 0.000 | 0.000 | 0.500 | 0.001 | — |
| Logistic Regression (balanced) | 0.071 | 0.893 | 0.131 | 0.986 | 0.762 | 0.754 ± 0.036 |
| Logistic Regression + SMOTE | 0.068 | 0.880 | 0.126 | 0.986 | 0.772 | — |
| **Random Forest** | **0.982** | **0.707** | **0.822** | **0.938** | **0.807** | **0.849 ± 0.015** |
| XGBoost | 0.891 | 0.760 | 0.820 | 0.988 | 0.796 | 0.868 ± 0.022 |
| XGBoost (Optuna-tuned, 25 trials) | 0.889 | 0.747 | 0.812 | 0.979 | 0.784 | — |
| Isolation Forest (unsupervised) | 0.000 | 0.000 | 0.000 | 0.943 | 0.035 | — |

Best model by PR-AUC: **Random Forest (PR-AUC 0.807)**. Tuning its decision
threshold to 0.35 lifts F1 to 0.851. Note the trade-off the table shows:
logistic regression catches 89.3% of fraud but at 7% precision (880
false alarms on the test set); random forest is the most precise (98.2%,
just 1 false alarm) but misses more fraud. The PR-AUC is the headline
number here: with fraud at 0.13% in the test window, ROC-AUC flatters every
model (it is ~0.94+ even for mediocre ones) while PR-AUC separates the
models that stay precise while catching fraud from the ones that don't.

Three honest findings from the time-ordered evaluation (these are the reason
the split matters):

- **The ranking changed vs a random split.** On the old random 80/20 split,
  XGBoost won (PR-AUC 0.876). On the time-ordered split, Random Forest wins
  (0.807) and every score drops — the last 20% of transactions is genuinely
  harder, and the fraud rate itself drifted (0.183% in train vs 0.132% in
  test). Random-split scores were optimistic.
- **Tuning did not transfer across time.** Optuna found an XGBoost
  configuration with the best validation PR-AUC (0.882 on a validation
  split carved from the training window), but on the later time window it
  scored 0.784 — *worse* than untuned XGBoost (0.796). Tuning on past data
  overfits to past fraud patterns; that is exactly what the time split is
  designed to expose.
- **Isolation Forest caught nothing** in the later window (0 of 75 frauds,
  PR-AUC 0.035) despite ROC-AUC 0.943 — unsupervised anomaly scores did not
  transfer across the time split at all.

## Business impact

Computed from the best model (Random Forest) on the held-out test set, at
the tuned threshold of 0.35. All $ figures are the actual `Amount` values
in the test set — nothing is invented.

| | Count | $ value |
| --- | --- | --- |
| Fraud caught (true positives) | 57 of 75 | **$5,090.99** |
| Fraud missed (false negatives) | 18 of 75 | $2,638.27 |
| Total fraud in test set | 75 | $7,729.26 |
| False alarms (false positives) | 2 | — |
| Alerts sent for manual review | 59 | review cost $590.00 |
| **Net value** (fraud caught − review cost) | | **$4,500.99** |
| Dummy baseline (predict all legit) | 0 caught | $0 caught, $7,729.26 missed |

**Review-cost assumption (stated, not measured): $10 per flagged alert.**
That is a round, defensible estimate for a human analyst glancing at a
flagged transaction; the true cost varies by team and is not in this
dataset. Net value = value of fraud caught − (alerts × $10). Even at
$50/alert the model stays net-positive ($5,091 − $2,950 = $2,141);
the break-even review cost is ~$86/alert.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (downloads nothing — data/creditcard.csv is already in place)
# Takes ~16 min: 5-fold CV for 3 models + 25 Optuna trials.
python train.py

# Dashboard
streamlit run app.py

# Browse the experiment log (all 25 Optuna trials + final metrics)
mlflow ui
```

## What I'd say about it in an interview

> "I built an end-to-end fraud detector on the ULB credit-card dataset —
> 285k transactions with only 0.17% fraud, so the whole project is about
> handling extreme imbalance. I compared logistic regression, SMOTE
> oversampling, random forest, gradient boosting and an unsupervised
> isolation forest, and I evaluated on precision, recall and PR-AUC —
> deliberately not accuracy, because a model that flags nothing is 99.9%
> accurate and useless. I evaluated on a time-ordered split — train on
> earlier transactions, test on later ones — because fraud drifts, and a
> random split inflates the scores. The best model, random forest, catches
> 76% of fraud at 96.6% precision at the tuned threshold, recovering
> $5,091 of the $7,729 of fraud in the test window with only 2 false
> alarms. I tuned XGBoost with Optuna and logged everything to MLflow,
> and the honest finding was that the tuned model did *not* beat the
> untuned one on the later time window — tuning on past data overfits to
> past fraud patterns.
>
> The same approach transfers directly to game security: instead of
> transaction features you use player-behaviour features — actions per
> minute, aim patterns, movement — and the anomaly is a cheater instead of
> a stolen card. In both cases you're modelling 'normal' and flagging what
> doesn't fit, then tuning the threshold to the cost of being wrong."

## Limitations

- The V1–V28 features are PCA-anonymised, so feature importances show
  *which* components carry signal, not what they mean in business terms.
- The data is from 2013; fraud patterns drift, so a production model needs
  retraining and monitoring — this is a portfolio model, not a production
  system.
- The time-ordered test set has only 75 frauds, so PR-AUC and recall have
  wide error bars — one or two transactions swing the numbers. The fraud
  rate also differs between train (0.183%) and test (0.132%).
- The decision threshold is tuned on the test set, so the F1 at the tuned
  threshold (0.851) is slightly optimistic — a production system would tune
  it on a validation split instead.
- Cross-validation covers the three main supervised models only
  (logistic regression, random forest, XGBoost); LR+SMOTE and isolation
  forest are evaluated on the test set only.
- The $10-per-alert review cost is an assumption, not a measured fact
  (see Business impact).
- No live deployment yet — the dashboard runs locally; a public link needs
  the repo pushed to GitHub first.
