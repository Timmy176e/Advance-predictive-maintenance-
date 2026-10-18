"""
pm_logic.py - core decision-support logic for the Predictive Maintenance app.

Deliberately free of any `streamlit` or `shap`-at-import-time dependency, so
it can be exercised directly with plain `python3` (see the bottom of this
file / the project's test script) before ever touching the UI layer. `app.py`
imports this module and only adds the Streamlit widgets around it.

Covers:
  - schema/type validation and a per-row data-quality report (unseen
    categories, out-of-range readings, missing values, bad types)
  - running the 4 trained pipelines + turning the raw risk score into a
    risk tier with a recommended action
  - per-row explanations (real SHAP values when the `shap` package is
    available, a transparent z-score x global-importance heuristic when it
    isn't - same output shape either way)
  - a simple population-drift check of an uploaded batch against the
    training distribution
  - append/read helpers for the on-disk prediction log that powers the
    "Investigate Asset" and "Model Performance" tabs
"""
from __future__ import annotations

import json
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema (must match train_models.py)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "age_days", "fault_count_last_30d", "fault_count_last_90d",
    "downtime_minutes_last_90d", "days_since_last_maintenance", "degradation_index",
]
CATEGORICAL_FEATURES = ["comp_type", "section", "last_maintenance_type", "line"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
OPTIONAL_ID_COLS = ["component_id", "area_code", "component_name"]
DATE_COL = "snapshot_date"

PREDICTION_COLS = [
    "predicted_maintenance_type", "predicted_maintenance_type_confidence",
    "predicted_failure_risk_score", "risk_tier", "risk_tier_emoji", "recommended_action",
    "predicted_remaining_useful_life_days", "predicted_days_until_due",
    "predicted_next_maintenance_due_date",
]
LOG_COLUMNS = OPTIONAL_ID_COLS + [DATE_COL] + FEATURE_COLS + PREDICTION_COLS + ["logged_at"]

# (min_score, tier label, emoji, recommended action) - evaluated top-down
RISK_TIERS = [
    (0.80, "Critical", "🔴", "Schedule corrective maintenance immediately - treat as a near-term failure risk."),
    (0.55, "High", "🟠", "Plan predictive maintenance within the next cycle and monitor closely until then."),
    (0.30, "Medium", "🟡", "Keep on the normal preventive schedule; re-check at the next snapshot."),
    (0.00, "Low", "🟢", "No action needed beyond routine preventive maintenance."),
]


def risk_tier(score: float):
    """Map a 0-1 failure risk score to (label, emoji, recommended_action)."""
    score = 0.0 if score is None or np.isnan(score) else float(score)
    for threshold, label, emoji, action in RISK_TIERS:
        if score >= threshold:
            return label, emoji, action
    return RISK_TIERS[-1][1], RISK_TIERS[-1][2], RISK_TIERS[-1][3]


# ---------------------------------------------------------------------------
# Reference stats / metrics loaders
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def global_importance_map(metrics: dict, model_key: str = "failure_risk_regressor") -> dict:
    """Flatten metrics_summary.json's feature_importance list into {name: importance}."""
    pairs = metrics.get("feature_importance", {}).get(model_key, [])
    return {name: val for name, val in pairs}


# ---------------------------------------------------------------------------
# Validation / data-quality
# ---------------------------------------------------------------------------
def validate_and_clean(df: pd.DataFrame, ref: dict):
    """
    Validate + clean an uploaded dataframe against the training reference
    stats. Returns (clean_df, quality_report_df, issues_summary).

    clean_df   - required columns coerced to the right dtype, with missing
                 numeric values imputed using the TRAIN median (unseen /
                 missing categorical values are left as-is: the pipeline's
                 OneHotEncoder(handle_unknown="ignore") already degrades
                 gracefully for those, it just can't use that one signal).
    quality_report_df - one row per flagged input row, listing what was
                 wrong with it.
    issues_summary     - counts for a top-of-page banner.
    """
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required column(s): {missing_cols}")

    df = df.reset_index(drop=True).copy()
    flags: dict[int, list[str]] = {i: [] for i in df.index}

    # ---- categorical: stringify, flag categories never seen in training ----
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str).str.strip()
        allowed = set(ref["categorical_values"].get(col, []))
        unseen_mask = ~df[col].isin(allowed)
        for i in df.index[unseen_mask]:
            flags[i].append(f"unseen {col}='{df.at[i, col]}'")

    # ---- numeric: coerce type, impute missing, flag out-of-range ----
    for col in NUMERIC_FEATURES:
        original = df[col]
        coerced = pd.to_numeric(original, errors="coerce")

        was_text = original.notna() & (original.astype(str).str.strip() != "")
        bad_type_mask = coerced.isna() & was_text
        for i in df.index[bad_type_mask]:
            flags[i].append(f"non-numeric {col}='{original.at[i]}'")

        stats = ref["numeric_stats"][col]
        missing_mask = coerced.isna()
        for i in df.index[missing_mask]:
            flags[i].append(f"missing {col} (imputed with training median {stats['p50']:g})")
        coerced = coerced.fillna(stats["p50"])

        lo, hi = stats["min"], stats["max"]
        span = max(hi - lo, 1e-6)
        soft_lo, soft_hi = lo - 0.10 * span, hi + 0.10 * span
        oor_mask = (coerced < soft_lo) | (coerced > soft_hi)
        for i in df.index[oor_mask]:
            flags[i].append(f"{col}={coerced.at[i]:.3g} is outside the training range [{lo:.3g}, {hi:.3g}]")

        df[col] = coerced

    quality_rows = []
    for i, issues in flags.items():
        if issues:
            quality_rows.append({
                "row": i,
                "component_id": df.at[i, "component_id"] if "component_id" in df.columns else None,
                "n_issues": len(issues),
                "issues": "; ".join(issues),
            })
    quality_report_df = pd.DataFrame(
        quality_rows, columns=["row", "component_id", "n_issues", "issues"]
    )

    def _count(substr):
        return sum(1 for v in flags.values() if any(substr in s for s in v))

    issues_summary = {
        "total_rows": int(len(df)),
        "rows_flagged": int(len(quality_rows)),
        "unseen_category_rows": _count("unseen "),
        "out_of_range_rows": _count("outside the training range"),
        "missing_value_rows": _count("missing "),
        "non_numeric_rows": _count("non-numeric "),
    }
    return df, quality_report_df, issues_summary


def numeric_drift_table(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Compare an uploaded batch's numeric feature distribution to the training one."""
    rows = []
    for col in NUMERIC_FEATURES:
        stats = ref["numeric_stats"][col]
        train_std = stats["std"] or 1e-6
        batch_mean = float(df[col].mean())
        batch_std = float(df[col].std()) if len(df) > 1 else 0.0
        shift = (batch_mean - stats["mean"]) / train_std
        rows.append({
            "feature": col,
            "train_mean": stats["mean"],
            "batch_mean": round(batch_mean, 4),
            "shift_in_std_devs": round(float(shift), 2),
            "flag": "drift" if abs(shift) > 1.0 else "ok",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def run_predictions(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """Run all 4 pipelines on an already-cleaned dataframe; adds prediction columns."""
    df = df.copy()
    X = df[FEATURE_COLS]

    clf = models["type"]
    proba = clf.predict_proba(X)
    classes = np.asarray(clf.classes_)
    pred_idx = proba.argmax(axis=1)
    df["predicted_maintenance_type"] = classes[pred_idx]
    df["predicted_maintenance_type_confidence"] = np.round(proba.max(axis=1), 3)

    risk_scores = np.clip(models["risk"].predict(X).astype(float), 0.0, 1.0)
    df["predicted_failure_risk_score"] = np.round(risk_scores, 3)
    tiers = [risk_tier(s) for s in risk_scores]
    df["risk_tier"] = [t[0] for t in tiers]
    df["risk_tier_emoji"] = [t[1] for t in tiers]
    df["recommended_action"] = [t[2] for t in tiers]

    rul = np.round(models["rul"].predict(X)).astype(int)
    df["predicted_remaining_useful_life_days"] = np.maximum(rul, 0)

    days_until_due = np.round(models["due"].predict(X)).astype(int)
    df["predicted_days_until_due"] = days_until_due

    if DATE_COL not in df.columns or df[DATE_COL].isna().all():
        df[DATE_COL] = pd.Timestamp(date.today())
    else:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["predicted_next_maintenance_due_date"] = df[DATE_COL] + pd.to_timedelta(days_until_due, unit="D")
    return df


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
def explain_row(pipeline, row_df: pd.DataFrame, ref: dict, global_importance: dict, top_n: int = 6):
    """
    Per-row explanation of a *regressor* pipeline's prediction for one row
    (used for the failure-risk and RUL regressors - both single-output
    RandomForestRegressors, so shap.TreeExplainer returns one contribution
    per feature with no class axis to worry about).

    Returns (rows, method) where method is "shap" or "heuristic" and rows is
    a list of {feature, contribution, detail} sorted by |contribution| desc.
    A positive contribution pushed the prediction up; negative pulled it down.
    """
    try:
        import shap  # imported lazily: optional dependency, see requirements.txt

        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        X_trans = prep.transform(row_df[FEATURE_COLS])
        feat_names = [n.replace("cat__", "").replace("num__", "") for n in prep.get_feature_names_out()]

        explainer = shap.TreeExplainer(model)
        sv = np.asarray(explainer.shap_values(X_trans)).reshape(-1)
        if sv.shape[0] != len(feat_names):
            raise ValueError(f"unexpected shap output shape {sv.shape} for {len(feat_names)} features")

        rows = sorted(
            ({"feature": f, "contribution": round(float(c), 4)} for f, c in zip(feat_names, sv)),
            key=lambda d: abs(d["contribution"]), reverse=True,
        )[:top_n]
        for r in rows:
            r["detail"] = "SHAP contribution to the predicted value"
        return rows, "shap"
    except Exception:
        return _heuristic_explain(row_df, ref, global_importance, top_n), "heuristic"


def _heuristic_explain(row_df: pd.DataFrame, ref: dict, global_importance: dict, top_n: int):
    """
    Fallback used when `shap` isn't installed / fails: for each numeric
    feature, how far this row sits from the training-population mean (in
    standard deviations), scaled by that feature's global importance in the
    trained model. For categoricals, the one-hot feature's global importance
    is used directly. Not true Shapley values, but a transparent, cheap
    stand-in that needs nothing beyond what's already trained.
    """
    row = row_df.iloc[0]
    scored = []
    for col in NUMERIC_FEATURES:
        stats = ref["numeric_stats"][col]
        std = stats["std"] or 1e-6
        z = (float(row[col]) - stats["mean"]) / std
        imp = global_importance.get(col, 0.0)
        scored.append({
            "feature": col,
            "contribution": round(float(z * imp), 4),
            "detail": f"value {row[col]:.3g} vs training avg {stats['mean']:.3g}",
        })
    for col in CATEGORICAL_FEATURES:
        key = f"{col}_{row[col]}"
        imp = global_importance.get(key, 0.0)
        scored.append({
            "feature": key,
            "contribution": round(float(imp), 4),
            "detail": f"{col} = {row[col]}",
        })
    scored.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Prediction log (powers Investigate Asset + Model Performance tabs)
# ---------------------------------------------------------------------------
def append_to_log(log_path: Path, results_df: pd.DataFrame) -> int:
    """
    Append this batch's rows to the on-disk prediction log. Returns rows written.

    Always writes the FULL LOG_COLUMNS schema (missing optional columns such
    as component_id become blank), never just whichever columns this
    particular batch happened to have. A CSV append is purely positional -
    if one batch lacked component_id/area_code and we wrote a narrower set
    of columns, those values would land under the wrong header for every
    row written after it, silently corrupting the whole log.
    """
    out = results_df.copy()
    out["logged_at"] = datetime.utcnow().isoformat(timespec="seconds")
    out = out.reindex(columns=LOG_COLUMNS)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    return len(out)


def read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(log_path)
    for c in (DATE_COL, "logged_at", "predicted_next_maintenance_due_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Realized-performance feedback loop (Model Performance tab)
# ---------------------------------------------------------------------------
def score_against_actuals(log_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict:
    """
    Join the logged predictions to a user-supplied actuals file (on
    component_id, and snapshot_date when both sides have it) and compute
    realized performance - i.e. how the model is doing on real outcomes,
    not just the held-out test set from training time.

    actuals_df may supply any subset of: actual_maintenance_type,
    actual_failure_risk_score (0-1), actual_remaining_useful_life_days.
    Returns a dict of whichever metrics could be computed from the columns
    actually supplied; always includes "n_matched".
    """
    if "component_id" not in log_df.columns or "component_id" not in actuals_df.columns:
        raise ValueError("Both the prediction log and the actuals file need a component_id column to join on.")

    join_cols = ["component_id"]
    if DATE_COL in log_df.columns and DATE_COL in actuals_df.columns:
        join_cols.append(DATE_COL)

    merged = log_df.merge(actuals_df, on=join_cols, how="inner", suffixes=("", "_actual"))
    out = {"n_matched": int(len(merged))}
    if merged.empty:
        return out

    if "actual_maintenance_type" in merged.columns:
        correct = merged["predicted_maintenance_type"] == merged["actual_maintenance_type"]
        out["maintenance_type_accuracy"] = round(float(correct.mean()), 4)
    if "actual_failure_risk_score" in merged.columns:
        err = (merged["predicted_failure_risk_score"] - merged["actual_failure_risk_score"]).abs()
        out["failure_risk_mae"] = round(float(err.mean()), 4)
    if "actual_remaining_useful_life_days" in merged.columns:
        err = (merged["predicted_remaining_useful_life_days"] - merged["actual_remaining_useful_life_days"]).abs()
        out["rul_mae"] = round(float(err.mean()), 4)
    return out
