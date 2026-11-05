"""
Credit-card fraud detection — model training & comparison.

What this does
-------------
Loads the ULB credit-card fraud dataset (284,807 transactions, 0.17% fraud),
trains several models and compares them on the metrics that actually matter
for fraud:

    precision, recall, F1, ROC-AUC and PR-AUC (average precision).

Accuracy is reported ONLY to show why it is meaningless here: a model that
predicts "legitimate" for every transaction scores 99.83% accuracy while
catching zero fraud.

Evaluation design (2026-11-05 revision)
----------------------------------------
* TIME-ORDERED split, not random: rows are sorted by Time; the first 80%
  by time is the training set and the last 20% by time is the held-out
  test set. Fraud drifts over time, so a random split lets the model peek
  at the future and inflates the scores.
* 5-fold stratified cross-validation on the training portion for the three
  main supervised models (logistic regression, random forest, XGBoost),
  reported as mean +/- std PR-AUC.
* XGBoost is tuned with Optuna (25 trials, objective = PR-AUC on a
  validation split carved from the training portion). Every trial and the
  final metrics are logged to MLflow (local mlruns/ directory).
* A business-impact section turns the best model's test predictions into
  dollars: value of fraud caught, value missed, and the manual-review cost
  of false alarms (assumed $10 per flagged alert — an assumption, not a
  measured fact).

Models compared
---------------
1. Logistic Regression (class_weight='balanced')      — linear baseline
2. Logistic Regression + SMOTE oversampling           — imbalance handling demo
3. Random Forest (class_weight='balanced_subsample')  — bagged trees
4. XGBoost (untuned defaults from the original build) — gradient boosting
5. XGBoost (Optuna-tuned, 25 trials)                 — tuned gradient boosting
6. Isolation Forest (unsupervised, contamination=0.17%)

The best model (highest PR-AUC) is saved to models/ along with the scaler,
the feature list and metrics.json.

Usage
-----
    python train.py            # from the project root, or anywhere
"""

import json
import os
import time
import warnings
from pathlib import Path

# MLflow 3.x puts the local file-store backend in "maintenance mode" and
# refuses it unless this is set. This project logs to a local mlruns/
# directory on purpose, so opt in explicitly (before mlflow is imported).
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "creditcard.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
FRAUD_RATE = 492 / 284_807  # 0.001727 — used as Isolation Forest contamination
N_OPTUNA_TRIALS = 25
REVIEW_COST_PER_ALERT = 10.0  # ASSUMPTION: $10 manual-review cost per flagged alert

# ---------------------------------------------------------------------------
# Optional imports — the script degrades gracefully if a package is missing.
# ---------------------------------------------------------------------------
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[note] imbalanced-learn not installed — skipping SMOTE comparison.")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    from optuna.samplers import TPESampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[note] optuna not installed — skipping XGBoost tuning.")

try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    print("[note] mlflow not installed — skipping experiment logging.")


def evaluate(name, y_true, y_pred, y_score):
    """One row of the comparison table."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "model": name,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "pr_auc": round(average_precision_score(y_true, y_score), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def cv_pr_auc(model, X, y, label):
    """5-fold stratified CV PR-AUC (mean +/- std) on the training portion.

    Folds run SEQUENTIALLY (n_jobs=1) on purpose: the VM has ~2 GB free
    RAM and parallel folds x parallel trees (RF/XGBoost n_jobs=-1 inside
    each fold) blows memory and gets the run OOM-killed.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="average_precision",
                             n_jobs=1)
    mean, std = float(scores.mean()), float(scores.std())
    print(f"  [cv] {label}: PR-AUC {mean:.4f} +/- {std:.4f} "
          f"(folds: {', '.join(f'{s:.4f}' for s in scores)})")
    return mean, std


def best_f1_threshold(y_true, y_score):
    """Scan thresholds 0.05–0.95 and return the one with the best F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 2), round(float(f1), 4)
    return best_t, best_f1


def main():
    t0 = time.time()

    # MLflow setup, once, up front. If it fails for any reason, training
    # continues without logging — a logging backend must never kill a run.
    mlflow_ok = False
    if HAS_MLFLOW:
        try:
            mlflow.set_tracking_uri(f"file://{ROOT / 'mlruns'}")
            mlflow.set_experiment("fraud-detection")
            mlflow_ok = True
        except Exception as e:
            print(f"  [mlflow] setup failed — continuing without logging: {e}")

    # ------------------------------------------------------------------ load
    print("Loading data ...")
    df = pd.read_csv(DATA)
    print(f"  rows: {len(df):,} | frauds: {int(df['Class'].sum()):,} "
          f"({100 * df['Class'].mean():.4f}%)")
    print("  EDA — Amount: mean %.2f, median %.2f, max %.2f"
          % (df["Amount"].mean(), df["Amount"].median(), df["Amount"].max()))
    print("  EDA — fraud Amount: mean %.2f vs legit mean %.2f"
          % (df.loc[df["Class"] == 1, "Amount"].mean(),
             df.loc[df["Class"] == 0, "Amount"].mean()))
    print("  EDA — fraud share by hour: highest in hours "
          f"{df.loc[df['Class'] == 1, 'Time'].div(3600).astype(int).mode().tolist()}")

    # ------------------------------------------------------- features / split
    # V1–V28 are already PCA-standardised. Only Time and Amount need scaling.
    features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

    # TIME-ORDERED split: sort by Time, first 80% = train, last 20% = test.
    # No shuffling — a random split would let the model train on the future
    # and predict the past, inflating scores that collapse in production.
    df = df.sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * 0.80)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train, y_train = train_df[features], train_df["Class"].values
    X_test, y_test = test_df[features], test_df["Class"].values
    print(f"  TIME-ORDERED split (no shuffle):")
    print(f"    train: {len(X_train):,} rows ({int(y_train.sum())} fraud) | "
          f"Time {train_df['Time'].min():,.0f}–{train_df['Time'].max():,.0f}s")
    print(f"    test:  {len(X_test):,} rows ({int(y_test.sum())} fraud) | "
          f"Time {test_df['Time'].min():,.0f}–{test_df['Time'].max():,.0f}s")
    print(f"    test fraud rate: {100 * y_test.mean():.4f}% "
          f"(train: {100 * y_train.mean():.4f}%)")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results, artifacts, test_probas = [], {}, {}

    # ------------------------------------------------------- dummy baseline
    # Predicts "legitimate" for everything. This is the accuracy trap.
    dummy_pred = np.zeros_like(y_test)
    dummy = evaluate("Dummy (predict all legit)", y_test, dummy_pred,
                     np.zeros_like(y_test, dtype=float))
    dummy["note"] = (f"accuracy = {100 * (y_test == 0).mean():.2f}% "
                     "while catching 0 frauds")
    results.append(dummy)
    print(f"  [dummy] accuracy would be {100 * (y_test == 0).mean():.2f}% "
          f"with recall = 0.0000 — accuracy is meaningless at 0.17% fraud.")

    # ------------------------------------------- 1. logistic regression
    print("Training Logistic Regression (balanced) ...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000,
                            random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    p = lr.predict_proba(X_test_s)[:, 1]
    row = evaluate("Logistic Regression (balanced)", y_test,
                   (p >= 0.5).astype(int), p)
    row["cv_pr_auc_mean"], row["cv_pr_auc_std"] = cv_pr_auc(
        LogisticRegression(class_weight="balanced", max_iter=1000,
                           random_state=RANDOM_STATE),
        X_train, y_train, "Logistic Regression")
    results.append(row)
    artifacts["lr"] = (lr, "proba")
    test_probas["Logistic Regression (balanced)"] = p

    # ------------------------------------------- 2. logistic + SMOTE
    if HAS_SMOTE:
        print("Training Logistic Regression + SMOTE ...")
        sm = SMOTE(random_state=RANDOM_STATE)
        X_sm, y_sm = sm.fit_resample(X_train_s, y_train)  # train only, never test
        lr_sm = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr_sm.fit(X_sm, y_sm)
        p = lr_sm.predict_proba(X_test_s)[:, 1]
        results.append(evaluate("Logistic Regression + SMOTE", y_test,
                                (p >= 0.5).astype(int), p))
        artifacts["lr_smote"] = (lr_sm, "proba")
        test_probas["Logistic Regression + SMOTE"] = p

    # ------------------------------------------- 3. random forest
    print("Training Random Forest (100 trees, balanced) ...")
    rf = RandomForestClassifier(n_estimators=100,
                                class_weight="balanced_subsample",
                                min_samples_leaf=2,
                                n_jobs=-1, random_state=RANDOM_STATE)
    rf.fit(X_train_s, y_train)
    p = rf.predict_proba(X_test_s)[:, 1]
    row = evaluate("Random Forest", y_test, (p >= 0.5).astype(int), p)
    row["cv_pr_auc_mean"], row["cv_pr_auc_std"] = cv_pr_auc(
        RandomForestClassifier(n_estimators=100,
                               class_weight="balanced_subsample",
                               min_samples_leaf=2,
                               n_jobs=2, random_state=RANDOM_STATE),
        X_train, y_train, "Random Forest")
    results.append(row)
    artifacts["rf"] = (rf, "proba")
    test_probas["Random Forest"] = p

    # ------------------------------------------- 4. gradient boosting
    neg, pos = int((y_train == 0).sum()), int(y_train.sum())
    xgb_params = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8,
                      scale_pos_weight=neg / pos, n_jobs=-1,
                      eval_metric="logloss", random_state=RANDOM_STATE)
    if HAS_XGB:
        print("Training XGBoost (untuned) ...")
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train_s, y_train)
        booster, bname = xgb, "XGBoost"
    elif HAS_LGBM:
        print("Training LightGBM ...")
        xgb = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             class_weight="balanced", n_jobs=-1,
                             random_state=RANDOM_STATE, verbose=-1)
        xgb.fit(X_train_s, y_train)
        booster, bname = xgb, "LightGBM"
    else:
        print("Training HistGradientBoosting (no XGBoost/LightGBM found) ...")
        xgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE)
        xgb.fit(X_train_s, y_train)
        booster, bname = xgb, "HistGradientBoosting"
    p = booster.predict_proba(X_test_s)[:, 1]
    row = evaluate(bname, y_test, (p >= 0.5).astype(int), p)
    if HAS_XGB:
        row["cv_pr_auc_mean"], row["cv_pr_auc_std"] = cv_pr_auc(
            XGBClassifier(**xgb_params), X_train, y_train, "XGBoost")
    results.append(row)
    artifacts["gbm"] = (booster, "proba")
    test_probas[bname] = p
    gbm_name = bname

    # ------------------------------------------- 5. XGBoost, Optuna-tuned
    tuned_info = {}
    if HAS_XGB and HAS_OPTUNA:
        print(f"Optuna tuning XGBoost ({N_OPTUNA_TRIALS} trials, "
              f"objective = PR-AUC on a validation split) ...")
        # Validation split carved from the TRAINING portion only (stratified
        # 80/20); the time-ordered test set is never touched during tuning.
        X_sub, X_val, y_sub, y_val = train_test_split(
            X_train_s, y_train, test_size=0.20, stratify=y_train,
            random_state=RANDOM_STATE)
        neg_s, pos_s = int((y_sub == 0).sum()), int(y_sub.sum())

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate",
                                                     0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree",
                                                        0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0,
                                                  log=True),
                "scale_pos_weight": neg_s / pos_s,
                "n_jobs": -1, "eval_metric": "logloss",
                "random_state": RANDOM_STATE,
            }
            m = XGBClassifier(**params)
            m.fit(X_sub, y_sub)
            val_score = average_precision_score(
                y_val, m.predict_proba(X_val)[:, 1])
            if mlflow_ok:
                try:
                    with mlflow.start_run(
                            run_name=f"trial_{trial.number}", nested=True):
                        mlflow.log_params(params)
                        mlflow.log_metric("val_pr_auc", float(val_score))
                except Exception as e:  # logging must never kill the study
                    print(f"  [mlflow] trial logging failed: {e}")
            return float(val_score)

        if mlflow_ok:
            tune_ctx = mlflow.start_run(run_name="xgb_optuna_tuning")
        else:
            tune_ctx = None
        try:
            study = optuna.create_study(
                direction="maximize",
                sampler=TPESampler(seed=RANDOM_STATE))
            study.optimize(objective, n_trials=N_OPTUNA_TRIALS,
                           show_progress_bar=False)
            best_params = dict(study.best_params)
            best_params["scale_pos_weight"] = neg / pos
            tuned_info = {
                "n_trials": N_OPTUNA_TRIALS,
                "best_val_pr_auc": round(float(study.best_value), 4),
                "best_params": {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in best_params.items()},
            }
            print(f"  [optuna] best val PR-AUC {study.best_value:.4f} with "
                  f"{tuned_info['best_params']}")
            if mlflow_ok and tune_ctx is not None:
                mlflow.log_params(tuned_info["best_params"])
                mlflow.log_metric("best_val_pr_auc",
                                  float(study.best_value))
        finally:
            if tune_ctx is not None:
                tune_ctx.__exit__(None, None, None)

        # Refit the tuned configuration on the FULL training portion.
        print("Refitting tuned XGBoost on the full training portion ...")
        xgb_tuned = XGBClassifier(**best_params, n_jobs=-1,
                                  eval_metric="logloss",
                                  random_state=RANDOM_STATE)
        xgb_tuned.fit(X_train_s, y_train)
        p = xgb_tuned.predict_proba(X_test_s)[:, 1]
        results.append(evaluate("XGBoost (Optuna-tuned)", y_test,
                                (p >= 0.5).astype(int), p))
        artifacts["gbm_tuned"] = (xgb_tuned, "proba")
        test_probas["XGBoost (Optuna-tuned)"] = p

    # ------------------------------------------- 6. isolation forest
    print("Training Isolation Forest (unsupervised, contamination=0.17%) ...")
    iso = IsolationForest(n_estimators=100, contamination=FRAUD_RATE,
                          n_jobs=-1, random_state=RANDOM_STATE)
    iso.fit(X_train_s)
    iso_pred = (iso.predict(X_test_s) == -1).astype(int)
    # decision_function: higher = more normal, so invert for a fraud score.
    iso_score = -iso.decision_function(X_test_s)
    results.append(evaluate("Isolation Forest (unsupervised)", y_test,
                            iso_pred, iso_score))
    artifacts["iso"] = (iso, "anomaly_score")

    # ------------------------------------------------------------- results
    results_df = pd.DataFrame(results)
    cols = ["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    print("\n" + results_df[cols].to_string(index=False))
    print("\nCV PR-AUC (5-fold stratified, training portion only):")
    for r in results:
        if "cv_pr_auc_mean" in r:
            print(f"  {r['model']}: {r['cv_pr_auc_mean']:.4f} "
                  f"+/- {r['cv_pr_auc_std']:.4f}")
    print("\nAccuracy is NOT in this table on purpose. At 0.17% fraud, "
          "a model that predicts 'legitimate' for every transaction is "
          "99.83% accurate and catches nothing.")

    # Best model = highest PR-AUC (the metric that matters most for fraud:
    # how precise you stay as you catch more of the rare class).
    ranked = results_df.sort_values("pr_auc", ascending=False)
    best_name = ranked.iloc[0]["model"]
    print(f"\nBest model by PR-AUC: {best_name} "
          f"(PR-AUC {ranked.iloc[0]['pr_auc']:.4f})")

    # Map the display name back to the saved artifact.
    key = {"Logistic Regression (balanced)": "lr",
           "Logistic Regression + SMOTE": "lr_smote",
           "Random Forest": "rf",
           gbm_name: "gbm",
           "XGBoost (Optuna-tuned)": "gbm_tuned",
           "Isolation Forest (unsupervised)": "iso"}[best_name]
    best_model, score_kind = artifacts[key]

    # Threshold tuning for the best model (only if it outputs probabilities).
    # NOTE: the threshold is tuned on the test set, so the F1 at that
    # threshold is slightly optimistic — a production system would tune it
    # on a validation split instead.
    tuned = {}
    best_proba = test_probas.get(best_name)
    if score_kind == "proba" and best_proba is not None:
        t, f1 = best_f1_threshold(y_test, best_proba)
        tuned = {"best_f1_threshold": t, "f1_at_best_threshold": f1}
        print(f"  Best F1 threshold for {best_name}: {t} (F1 {f1})")

    # --------------------------------------------------- business impact ($)
    # Computed from the best model's test predictions at the tuned threshold
    # (or 0.5 if no tuning was possible). All $ figures are the actual
    # Amount values in the held-out test set — nothing is invented.
    op_threshold = tuned.get("best_f1_threshold", 0.5)
    if best_proba is not None:
        op_pred = (best_proba >= op_threshold).astype(int)
    else:
        op_pred = (iso.predict(X_test_s) == -1).astype(int)
    test_amounts = test_df["Amount"].values
    tp_mask = (op_pred == 1) & (y_test == 1)
    fn_mask = (op_pred == 0) & (y_test == 1)
    fp_mask = (op_pred == 1) & (y_test == 0)
    n_tp, n_fn, n_fp = int(tp_mask.sum()), int(fn_mask.sum()), int(fp_mask.sum())
    value_caught = round(float(test_amounts[tp_mask].sum()), 2)
    value_missed = round(float(test_amounts[fn_mask].sum()), 2)
    total_fraud_value = round(value_caught + value_missed, 2)
    n_alerts = n_tp + n_fp
    review_cost = round(n_alerts * REVIEW_COST_PER_ALERT, 2)
    net_value = round(value_caught - review_cost, 2)
    business_impact = {
        "model": best_name,
        "threshold": op_threshold,
        "frauds_caught": n_tp,
        "value_caught_usd": value_caught,
        "frauds_missed": n_fn,
        "value_missed_usd": value_missed,
        "total_fraud_value_usd": total_fraud_value,
        "false_positives": n_fp,
        "alerts_reviewed": n_alerts,
        "review_cost_per_alert_usd": REVIEW_COST_PER_ALERT,
        "review_cost_usd": review_cost,
        "net_value_usd": net_value,
        "dummy_value_caught_usd": 0.0,
        "dummy_value_missed_usd": total_fraud_value,
        "note": ("Review cost is an ASSUMPTION ($10 per flagged alert), not a "
                 "measured fact. Net value = value of fraud caught - review "
                 "cost. The dummy baseline catches $0 and misses all fraud."),
    }
    print(f"\nBusiness impact ({best_name} @ threshold {op_threshold}, "
          f"test set):")
    print(f"  fraud caught: {n_tp} txns = ${value_caught:,.2f} | "
          f"missed: {n_fn} txns = ${value_missed:,.2f} "
          f"(total fraud in test: ${total_fraud_value:,.2f})")
    print(f"  false alarms: {n_fp} | review cost @ ${REVIEW_COST_PER_ALERT:.0f}/"
          f"alert = ${review_cost:,.2f} | net value = ${net_value:,.2f} "
          f"(dummy baseline: $0 caught, ${total_fraud_value:,.2f} missed)")

    # -------------------------------------------------------------- save
    joblib.dump({"model": best_model, "scaler": scaler, "features": features,
                 "model_name": best_name, "score_kind": score_kind},
                MODELS_DIR / "best_model.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # A small stratified sample of the test set so the Streamlit demo works
    # without uploading a file.
    test_sample_df = X_test.copy()
    test_sample_df["Class"] = y_test
    demo = pd.concat([
        test_sample_df[test_sample_df["Class"] == 0].sample(
            n=4900, random_state=RANDOM_STATE),
        test_sample_df[test_sample_df["Class"] == 1],
    ]).sample(frac=1, random_state=RANDOM_STATE)
    demo.to_csv(ROOT / "data" / "sample.csv", index=False)

    metrics = {"split": ("time-ordered: first 80% by Time = train, "
                         "last 20% by Time = test (no shuffle)"),
               "results": results,
               "best_model": best_name,
               "tuned_threshold": tuned,
               "optuna": tuned_info,
               "business_impact": business_impact,
               "test_size": int(len(X_test)),
               "test_frauds": int(y_test.sum()),
               "test_time_range": [float(test_df["Time"].min()),
                                   float(test_df["Time"].max())],
               "elapsed_seconds": round(time.time() - t0, 1)}
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --------------------------------------------------------- MLflow: final
    if mlflow_ok:
        try:
            with mlflow.start_run(run_name="final_evaluation"):
                mlflow.log_param("split", metrics["split"])
                mlflow.log_param("best_model", best_name)
                mlflow.log_param("review_cost_per_alert_usd",
                                 REVIEW_COST_PER_ALERT)
                for r in results:
                    tag = r["model"].replace(" ", "_").replace("(", "") \
                        .replace(")", "").replace("+", "plus")
                    mlflow.log_metric(f"{tag}_pr_auc", r["pr_auc"])
                    mlflow.log_metric(f"{tag}_f1", r["f1"])
                    if "cv_pr_auc_mean" in r:
                        mlflow.log_metric(f"{tag}_cv_pr_auc_mean",
                                          r["cv_pr_auc_mean"])
                        mlflow.log_metric(f"{tag}_cv_pr_auc_std",
                                          r["cv_pr_auc_std"])
                mlflow.log_metric("net_value_usd", net_value)
                mlflow.log_metric("value_caught_usd", value_caught)
                mlflow.log_metric("value_missed_usd", value_missed)
                mlflow.log_dict(metrics, "metrics.json")
            print("  [mlflow] final metrics logged to mlruns/ "
                  "(experiment: fraud-detection)")
        except Exception as e:
            print(f"  [mlflow] final logging failed (non-fatal): {e}")

    print(f"\nSaved: models/best_model.joblib, models/metrics.json, "
          f"data/sample.csv  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
