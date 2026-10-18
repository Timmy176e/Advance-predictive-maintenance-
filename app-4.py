"""
Predictive Maintenance Intelligence Platform - Streamlit app.

Upload a CSV of current asset readings and get, per asset: a recommended
maintenance type (with confidence), a failure-risk score and tier, a
recommended action, remaining useful life, and the predicted next
maintenance due date. Beyond the original classifier, this build adds:

  - input validation: flags unseen categories, out-of-range readings,
    missing values and bad types before they quietly skew a prediction
  - per-asset explanations: why a given risk score came out the way it did
    (real SHAP values when the `shap` package is available, a transparent
    fallback heuristic otherwise)
  - an "Investigate Asset" view built from the running history of every
    batch predicted through this app
  - a "Model Performance" tab: training-time metrics, a population-drift
    check of the current upload against the training data, and an optional
    feedback loop that scores the model against real outcomes once they're
    known

The 4 model files are too large for the Git repo, so they're hosted on
Google Drive and downloaded into a local `models/` folder the first time
the app starts (then cached for the life of the running container).
`reference_stats.json` and `metrics_summary.json` are tiny and ship
straight in the repo next to this file.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import gdown
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import pm_logic as L

APP_DIR = Path(__file__).parent
MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOG_PATH = APP_DIR / "logs" / "predictions_log.csv"

# Google Drive file IDs for each model (the id between /d/ and /view in the share link).
# Replace these if you re-upload a retrained model to Drive later.
MODEL_FILES = {
    "type": ("1yF-zgkKA_fX9qPkVeMRuAL6N9EafVlwL", "maintenance_type_classifier.joblib"),
    "risk": ("1oNpbkDPf8W5LlWumroQlBWmHgiy6K0mo", "failure_risk_regressor.joblib"),
    "rul":  ("1BjAYSTWLqKSktTWri1mNzxSIF0mMHbcW", "rul_regressor.joblib"),
    "due":  ("1wqk3O2bw_cTIZQw9leNhoTQ3NSv4ZxV9", "days_until_due_regressor.joblib"),
}

TIER_COLORS = {"Critical": "#f8d7da", "High": "#ffe5cc", "Medium": "#fff3cd", "Low": "#d4edda"}
TYPE_COLORS = {"Corrective": "#f8d7da", "Predictive": "#fff3cd", "Preventive": "#d4edda"}

st.set_page_config(page_title="Predictive Maintenance Intelligence Platform", layout="wide", page_icon="🔧")


@st.cache_resource
def load_models():
    models = {}
    for key, (file_id, filename) in MODEL_FILES.items():
        local_path = MODEL_DIR / filename
        if not local_path.exists():
            gdown.download(id=file_id, output=str(local_path), quiet=False)
        models[key] = joblib.load(local_path)
    return models


@st.cache_data
def load_reference_and_metrics():
    ref = L.load_json(APP_DIR / "reference_stats.json")
    metrics = L.load_json(APP_DIR / "metrics_summary.json")
    return ref, metrics


def highlight_tier(val):
    return f"background-color: {TIER_COLORS.get(val, '')}"


def highlight_type(val):
    return f"background-color: {TYPE_COLORS.get(val, '')}"


def asset_label(row):
    parts = [str(row[c]) for c in ("component_id", "area_code") if c in row and pd.notna(row[c])]
    return " / ".join(parts) if parts else f"row {row.name}"


# ---------------------------------------------------------------------------
# Load models + reference data
# ---------------------------------------------------------------------------
st.title("🔧 Predictive Maintenance Intelligence Platform")
st.caption(
    "Upload a CSV of current asset readings for a failure-risk assessment, a recommended "
    "maintenance type and due date, a plain-language explanation of each alert, and a running "
    "view of how the models are performing."
)

try:
    with st.spinner("Loading models (first run downloads them from Google Drive)..."):
        models = load_models()
    ref, metrics = load_reference_and_metrics()
    global_importance = L.global_importance_map(metrics, "failure_risk_regressor")
except Exception as e:
    st.error(
        f"Couldn't load the models or reference data: {e}\n\n"
        "Most likely cause: one of the Google Drive model files isn't shared as "
        "\"Anyone with the link\" (Viewer), or reference_stats.json / metrics_summary.json "
        "is missing from the repo next to app.py."
    )
    st.stop()

if "last_results" not in st.session_state:
    st.session_state["last_results"] = None
if "last_quality_report" not in st.session_state:
    st.session_state["last_quality_report"] = None

tab_predict, tab_explain, tab_investigate, tab_performance = st.tabs(
    ["🔍 Predict & Alert", "🧠 Explain a Prediction", "📊 Investigate Asset", "📈 Model Performance"]
)

# =============================================================================
# TAB 1 - Predict & Alert
# =============================================================================
with tab_predict:
    with st.expander("Expected CSV columns", expanded=False):
        st.markdown(f"**Required:** `{'`, `'.join(L.FEATURE_COLS)}`")
        st.markdown(f"**Optional** (carried through for labeling): `{'`, `'.join(L.OPTIONAL_ID_COLS + [L.DATE_COL])}`")
        st.markdown(
            "- `comp_type`: " + ", ".join(ref["categorical_values"]["comp_type"]) + "\n"
            "- `last_maintenance_type`: " + ", ".join(ref["categorical_values"]["last_maintenance_type"]) + "\n"
            f"- `{L.DATE_COL}`: when the readings were taken (defaults to today if omitted)\n\n"
            "Rows don't need to be perfect - unknown categories, missing values and "
            "out-of-range readings are flagged and handled rather than rejected (see the "
            "data-quality panel after you upload)."
        )
        sample_path = APP_DIR / "sample_input.csv"
        if sample_path.exists():
            with open(sample_path, "rb") as f:
                st.download_button("Download a sample CSV", f, file_name="sample_input.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload asset data (CSV)", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to get started.")
    else:
        try:
            raw_df = pd.read_csv(uploaded)
            clean_df, quality_report, issues_summary = L.validate_and_clean(raw_df, ref)
        except ValueError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"Couldn't process that file: {e}")
            st.stop()

        # ---- data-quality panel ----
        if issues_summary["rows_flagged"] > 0:
            st.warning(
                f"⚠️ {issues_summary['rows_flagged']} of {issues_summary['total_rows']} rows had a data-quality "
                f"issue: {issues_summary['unseen_category_rows']} unseen categor{'y' if issues_summary['unseen_category_rows']==1 else 'ies'}, "
                f"{issues_summary['out_of_range_rows']} out-of-range reading(s), "
                f"{issues_summary['missing_value_rows']} missing value(s), "
                f"{issues_summary['non_numeric_rows']} bad type(s). Predictions were still made for every "
                "row (missing values imputed with the training median, unseen categories handled "
                "by the model as 'no signal') - check the table below before trusting a flagged row."
            )
            with st.expander(f"Data-quality details ({len(quality_report)} flagged rows)", expanded=False):
                st.dataframe(quality_report, use_container_width=True)
        else:
            st.success("✅ All rows passed validation - every value was within the training data's known range.")

        results = L.run_predictions(clean_df, models)
        st.session_state["last_results"] = results
        st.session_state["last_quality_report"] = quality_report

        logged_n = L.append_to_log(LOG_PATH, results)

        # ---- summary cards ----
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Assets analyzed", len(results))
        c2.metric("Critical risk", int((results["risk_tier"] == "Critical").sum()))
        c3.metric("Needing Corrective action", int((results["predicted_maintenance_type"] == "Corrective").sum()))
        c4.metric("Average failure risk", f"{results['predicted_failure_risk_score'].mean():.2f}")
        c5.metric("Soonest due date", results["predicted_next_maintenance_due_date"].min().date().isoformat())
        st.caption(f"Logged {logged_n} prediction(s) to this session's history for the Investigate / Performance tabs.")

        # ---- results table ----
        st.subheader("Predictions")
        display_cols = (
            [c for c in L.OPTIONAL_ID_COLS if c in results.columns]
            + [L.DATE_COL, "predicted_maintenance_type", "predicted_maintenance_type_confidence",
               "predicted_failure_risk_score", "risk_tier", "predicted_remaining_useful_life_days",
               "predicted_next_maintenance_due_date", "recommended_action"]
        )
        table = results[display_cols].sort_values("predicted_failure_risk_score", ascending=False).reset_index(drop=True)
        st.dataframe(
            table.style.map(highlight_tier, subset=["risk_tier"]).map(highlight_type, subset=["predicted_maintenance_type"]),
            use_container_width=True,
            height=min(80 + 35 * len(table), 500),
        )
        st.download_button(
            "Download predictions as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_predictions.csv",
            mime="text/csv",
        )

        # ---- breakdown charts ----
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Maintenance type breakdown")
            st.bar_chart(results["predicted_maintenance_type"].value_counts())
        with col_b:
            st.subheader("Risk tier breakdown")
            tier_order = ["Low", "Medium", "High", "Critical"]
            counts = results["risk_tier"].value_counts().reindex(tier_order).fillna(0)
            st.bar_chart(counts)

        st.caption(
            "Models are Random Forests trained on a synthetic dataset built to mirror this plant's real "
            "shift maintenance reports - treat outputs as a proof-of-concept until retrained on real "
            "historical maintenance records."
        )

# =============================================================================
# TAB 2 - Explain a Prediction
# =============================================================================
with tab_explain:
    results = st.session_state["last_results"]
    if results is None or len(results) == 0:
        st.info("Run a prediction in the **Predict & Alert** tab first, then come back here to explain any asset's score.")
    else:
        st.caption(
            "Pick an asset from the last batch to see what's driving its failure-risk score - "
            "which readings pushed it up, which pulled it down, versus what the model learned from training."
        )
        options = list(results.index)
        labels = {i: f"{asset_label(results.loc[i])} - risk {results.loc[i, 'predicted_failure_risk_score']:.2f} ({results.loc[i, 'risk_tier']})"
                  for i in options}
        picked = st.selectbox("Asset", options, format_func=lambda i: labels[i])
        row = results.loc[[picked]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted failure risk", f"{row['predicted_failure_risk_score'].iloc[0]:.2f}",
                   help="0 = no concern, 1 = failure imminent")
        c2.metric("Risk tier", f"{row['risk_tier_emoji'].iloc[0]} {row['risk_tier'].iloc[0]}")
        c3.metric("Remaining useful life", f"{row['predicted_remaining_useful_life_days'].iloc[0]} days")
        st.info(f"**Recommended action:** {row['recommended_action'].iloc[0]}")

        issues_for_row = None
        qr = st.session_state["last_quality_report"]
        if qr is not None and len(qr) and picked in qr["row"].values:
            issues_for_row = qr.loc[qr["row"] == picked, "issues"].iloc[0]
        if issues_for_row:
            st.warning(f"⚠️ This row had a data-quality flag: {issues_for_row}. Treat this explanation with that in mind.")

        explanation, method = L.explain_row(models["risk"], row, ref, global_importance, top_n=8)
        contrib = pd.Series({e["feature"]: e["contribution"] for e in explanation}).sort_values()
        st.subheader("What's driving this risk score")
        st.bar_chart(contrib)
        st.dataframe(pd.DataFrame(explanation)[["feature", "contribution", "detail"]], use_container_width=True, hide_index=True)
        method_note = (
            "Exact SHAP values from the trained risk model." if method == "shap" else
            "Heuristic estimate (how far each reading sits from the training average, weighted by that "
            "feature's overall importance) - the `shap` package wasn't available at runtime, so this is a "
            "transparent stand-in rather than exact Shapley values."
        )
        st.caption(f"Explanation method: **{method}**. {method_note} Positive bars push risk up, negative bars pull it down.")

# =============================================================================
# TAB 3 - Investigate Asset
# =============================================================================
with tab_investigate:
    log_df = L.read_log(LOG_PATH)
    if log_df.empty or "component_id" not in log_df.columns:
        st.info(
            "No prediction history yet (or uploaded files don't include `component_id`). Run a few "
            "batches through **Predict & Alert** - ideally the same assets across different dates - "
            "and their trend will build up here."
        )
    else:
        assets = sorted(log_df["component_id"].dropna().unique().tolist())
        picked_asset = st.selectbox("Asset (component_id)", assets)
        hist = log_df[log_df["component_id"] == picked_asset].sort_values(L.DATE_COL)

        if len(hist) == 1:
            st.caption("Only one logged snapshot for this asset so far - trend charts need at least two.")
        latest = hist.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest risk score", f"{latest['predicted_failure_risk_score']:.2f}")
        c2.metric("Latest tier", f"{latest.get('risk_tier_emoji','')} {latest.get('risk_tier','')}")
        c3.metric("Latest recommended type", latest["predicted_maintenance_type"])
        c4.metric("Snapshots logged", len(hist))

        if len(hist) > 1:
            st.subheader("Failure risk score over time")
            st.line_chart(hist.set_index(L.DATE_COL)["predicted_failure_risk_score"])

            trend_cols = [c for c in ["fault_count_last_90d", "downtime_minutes_last_90d", "degradation_index"] if c in hist.columns]
            if trend_cols:
                st.subheader("Underlying readings over time")
                st.line_chart(hist.set_index(L.DATE_COL)[trend_cols])

        st.subheader("Full logged history for this asset")
        hist_display_cols = [c for c in (
            [L.DATE_COL] + L.OPTIONAL_ID_COLS + ["predicted_maintenance_type", "predicted_failure_risk_score",
             "risk_tier", "predicted_remaining_useful_life_days", "predicted_next_maintenance_due_date", "logged_at"]
        ) if c in hist.columns]
        st.dataframe(hist[hist_display_cols], use_container_width=True, hide_index=True)

    with st.expander("Carry prediction history between sessions"):
        st.caption(
            "The log above lives on this running container and resets on redeploy or when a free-tier "
            "app goes to sleep. To keep history across sessions, download it here and re-upload it next "
            "time (or wire this up to a real database before using it on production data)."
        )
        if not log_df.empty:
            st.download_button("Download full log (CSV)", log_df.to_csv(index=False).encode("utf-8"),
                                file_name="predictions_log.csv", mime="text/csv")
        carried = st.file_uploader("Re-upload a previously downloaded log to merge it in", type=["csv"], key="log_merge")
        if carried is not None:
            try:
                prior = pd.read_csv(carried)
                prior.to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
                st.success(f"Merged {len(prior)} row(s) into the running log. Refresh the asset list above to see them.")
            except Exception as e:
                st.error(f"Couldn't merge that file: {e}")

# =============================================================================
# TAB 4 - Model Performance
# =============================================================================
with tab_performance:
    st.subheader("Training-time metrics")
    st.caption("Measured on a held-out, time-based test split (most recent months) when the models were last trained.")

    cls_m = metrics["maintenance_type_classifier"]
    c1, c2 = st.columns(2)
    c1.metric("Maintenance type - accuracy", f"{cls_m['accuracy']:.1%}")
    c1.metric("Maintenance type - macro F1", f"{cls_m['macro_f1']:.2f}")
    with c2:
        st.caption("Confusion matrix (rows = actual, columns = predicted)")
        cm = pd.DataFrame(cls_m["confusion_matrix"], index=cls_m["labels"], columns=cls_m["labels"])
        st.dataframe(cm.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)

    reg_cols = st.columns(3)
    for col, key, title in zip(reg_cols,
                                ["failure_risk_regressor", "rul_regressor", "days_until_due_regressor"],
                                ["Failure risk score", "Remaining useful life", "Days until due"]):
        m = metrics[key]
        with col:
            st.metric(f"{title} - MAE", m["mae"])
            st.metric(f"{title} - R²", m["r2"])

    st.subheader("What drives each model (global feature importance)")
    fi_col1, fi_col2 = st.columns(2)
    with fi_col1:
        st.caption("Maintenance type classifier")
        fi = pd.Series(dict(metrics["feature_importance"]["maintenance_type_classifier"])).sort_values()
        st.bar_chart(fi)
    with fi_col2:
        st.caption("Failure risk regressor")
        fi2 = pd.Series(dict(metrics["feature_importance"]["failure_risk_regressor"])).sort_values()
        st.bar_chart(fi2)

    st.divider()
    st.subheader("This session's usage")
    log_df = L.read_log(LOG_PATH)
    if log_df.empty:
        st.info("No predictions logged yet this session.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions logged", len(log_df))
        c2.metric("Unique assets seen", log_df["component_id"].nunique() if "component_id" in log_df.columns else "n/a")
        if "risk_tier" in log_df.columns:
            c3.metric("Currently Critical/High", int(log_df["risk_tier"].isin(["Critical", "High"]).sum()))
        if "risk_tier" in log_df.columns:
            tier_order = ["Low", "Medium", "High", "Critical"]
            st.bar_chart(log_df["risk_tier"].value_counts().reindex(tier_order).fillna(0))

    st.divider()
    st.subheader("Data drift - current upload vs. training data")
    last_results = st.session_state["last_results"]
    if last_results is None:
        st.info("Upload a batch in **Predict & Alert** to see how its feature distribution compares to training.")
    else:
        drift = L.numeric_drift_table(last_results, ref)
        n_drift = (drift["flag"] == "drift").sum()
        if n_drift:
            st.warning(f"⚠️ {n_drift} feature(s) in the last upload differ from the training distribution by more than 1 standard deviation - the model may be extrapolating for this batch.")
        else:
            st.success("✅ The last upload's feature distribution looks consistent with training data.")
        st.dataframe(drift, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Score against real outcomes (feedback loop)")
    st.caption(
        "Once real outcomes are known for logged predictions, upload them here to see realized "
        "performance - not just the training-time test score above. Needs a `component_id` column "
        "(and ideally `snapshot_date`) plus any of: `actual_maintenance_type`, "
        "`actual_failure_risk_score` (0-1), `actual_remaining_useful_life_days`."
    )
    actuals_file = st.file_uploader("Upload actual outcomes (CSV)", type=["csv"], key="actuals")
    if actuals_file is not None:
        try:
            actuals_df = pd.read_csv(actuals_file)
            if L.DATE_COL in actuals_df.columns:
                actuals_df[L.DATE_COL] = pd.to_datetime(actuals_df[L.DATE_COL], errors="coerce")
            log_for_scoring = L.read_log(LOG_PATH)
            scored = L.score_against_actuals(log_for_scoring, actuals_df)
            if scored["n_matched"] == 0:
                st.warning("No matching component_id / snapshot_date found between the log and this file.")
            else:
                st.success(f"Matched {scored['n_matched']} logged prediction(s) to real outcomes.")
                cols = st.columns(max(len(scored) - 1, 1))
                i = 0
                for k, v in scored.items():
                    if k == "n_matched":
                        continue
                    cols[i].metric(k.replace("_", " ").title(), v)
                    i += 1
        except Exception as e:
            st.error(f"Couldn't score that file: {e}")
