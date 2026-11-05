"""
Fraud Detection Dashboard — Streamlit app.

What it does
------------
Loads the best model saved by train.py and lets you score a batch of
transactions: upload a CSV (same columns as the training data) or use the
built-in 5,000-row demo sample. It shows the flagged transactions, the
confusion matrix, the precision–recall curve and feature importances at the
threshold you choose.

Run with:
    streamlit run app.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score)

ROOT = Path(__file__).resolve().parent
ARTIFACT = ROOT / "models" / "best_model.joblib"
METRICS = ROOT / "models" / "metrics.json"
SAMPLE = ROOT / "data" / "sample.csv"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit-Card Fraud Detection")
st.caption("Behavioural anomaly detection — the same skill set as fraud analysis "
           "and game-security (anti-cheat) work: find the transaction that "
           "doesn't look like the account that made it.")


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(ARTIFACT)
    import json
    with open(METRICS) as f:
        metrics = json.load(f)
    return bundle, metrics


bundle, metrics = load_artifacts()
model = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
model_name = bundle["model_name"]
score_kind = bundle["score_kind"]

# ------------------------------------------------------------------ sidebar
st.sidebar.header("Scoring")
threshold = st.sidebar.slider("Flag threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Transactions scored above this are flagged "
                                   "as fraud. Lower = catch more fraud, "
                                   "but more false alarms.")
uploaded = st.sidebar.file_uploader("Upload a CSV of transactions",
                                    type=["csv"])
use_demo = st.sidebar.button("Use demo sample (5,000 rows)")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    source = "uploaded file"
elif use_demo or "demo" not in st.session_state:
    df = pd.read_csv(SAMPLE)
    source = "demo sample (held-out test rows)"
    st.session_state["demo"] = True
else:
    df = pd.read_csv(SAMPLE)
    source = "demo sample (held-out test rows)"

missing = [c for c in features if c not in df.columns]
if missing:
    st.error(f"Missing columns in the uploaded file: {missing}. "
             f"Expected: {features}")
    st.stop()

# ------------------------------------------------------------------ scoring
X = scaler.transform(df[features])
if score_kind == "proba":
    scores = model.predict_proba(X)[:, 1]
else:  # Isolation Forest: convert anomaly scores to a 0–1 fraud score
    raw = -model.decision_function(X)
    scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

df = df.copy()
df["fraud_score"] = scores
df["flagged"] = (scores >= threshold).astype(int)
has_labels = "Class" in df.columns

# ------------------------------------------------------------ headline stats
n_flagged = int(df["flagged"].sum())
st.subheader(f"Scored {len(df):,} transactions from {source} "
             f"with **{model_name}**")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flagged as fraud", f"{n_flagged:,}")
c2.metric("Flag rate", f"{100 * n_flagged / len(df):.2f}%")
if has_labels:
    y = df["Class"].values
    pred = df["flagged"].values
    c3.metric("Precision @ threshold",
              f"{precision_score(y, pred, zero_division=0):.3f}")
    c4.metric("Recall @ threshold",
              f"{recall_score(y, pred, zero_division=0):.3f}")
else:
    c3.metric("Precision @ threshold", "— (no labels)")
    c4.metric("Recall @ threshold", "— (no labels)")

# ------------------------------------------------------- flagged transactions
st.subheader("🚩 Flagged transactions")
flagged = df[df["flagged"] == 1].sort_values("fraud_score", ascending=False)
if flagged.empty:
    st.info("No transactions flagged at this threshold — try lowering it.")
else:
    cols = ["Time", "Amount", "fraud_score"] + (
        ["Class"] if has_labels else [])
    st.dataframe(flagged[cols].head(50), use_container_width=True)
    st.caption(f"Showing top 50 of {len(flagged):,} flagged.")

# ------------------------------------------------------------- model quality
if has_labels:
    y = df["Class"].values
    pred = df["flagged"].values
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y, pred)
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        fontsize=14)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted legit", "Predicted fraud"])
        ax.set_yticklabels(["Actual legit", "Actual fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
        st.caption(f"F1 @ threshold: "
                   f"{f1_score(y, pred, zero_division=0):.3f} · "
                   f"PR-AUC: {average_precision_score(y, scores):.3f}")

    with col_b:
        st.subheader("Precision–Recall curve")
        prec, rec, _ = precision_recall_curve(y, scores)
        fig, ax = plt.subplots()
        ax.plot(rec, prec)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"PR-AUC = {average_precision_score(y, scores):.3f} "
                     f"(random = {y.mean():.4f})")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption("Fraud is 0.17% of transactions, so the PR curve — not "
                   "the ROC curve — is the honest picture.")

# ------------------------------------------------------- feature importance
if hasattr(model, "feature_importances_"):
    st.subheader("Feature importance")
    imp = pd.Series(model.feature_importances_,
                    index=features).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    imp[::-1].plot.barh(ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    st.caption("V1–V28 are PCA-anonymised, so importances show *which* "
               "anonymised components carry the fraud signal, not what "
               "they mean in business terms.")

# ------------------------------------------------------------- held-out table
st.subheader("Held-out test-set results (from train.py)")
rows = metrics["results"]
st.dataframe(
    pd.DataFrame(rows)[["model", "precision", "recall", "f1",
                       "roc_auc", "pr_auc"]],
    use_container_width=True)
st.caption(f"Best model by PR-AUC: **{metrics['best_model']}** · "
           f"test set: {metrics['test_size']:,} rows, "
           f"{metrics['test_frauds']} frauds · "
           f"a 'predict everything legitimate' model would be 99.83% "
           f"accurate and catch 0 frauds.")
