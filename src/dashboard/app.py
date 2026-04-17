from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.monitoring.drift import (
    LOG_DIR,
    REPORT_DIR,
    analyze_prediction_drift,
    load_feedback,
    load_predictions,
    load_saved_summary,
)

st.set_page_config(
    page_title="RecEngine Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_data(ttl=15, show_spinner=False)
def _load_predictions(log_dir: str) -> pd.DataFrame:
    return load_predictions(Path(log_dir))


@st.cache_data(ttl=15, show_spinner=False)
def _load_feedback(log_dir: str) -> pd.DataFrame:
    return load_feedback(Path(log_dir))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _render_kpis(predictions: pd.DataFrame, feedback: pd.DataFrame) -> None:
    total_predictions = len(predictions)
    avg_latency = float(predictions["latency_ms"].dropna().mean()) if not predictions.empty else 0.0
    feedback_events = len(feedback)
    click_events = int((feedback.get("action") == "click").sum()) if not feedback.empty else 0
    purchase_events = int((feedback.get("action") == "purchase").sum()) if not feedback.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommendations", f"{total_predictions:,}")
    col2.metric("Avg latency", f"{avg_latency:.1f} ms")
    col3.metric("Feedback events", f"{feedback_events:,}")
    col4.metric("Click rate proxy", f"{_safe_rate(click_events, total_predictions):.1%}")

    col5, col6 = st.columns(2)
    col5.metric("Clicks", f"{click_events:,}")
    col6.metric("Purchases", f"{purchase_events:,}")


def _render_prediction_activity(predictions: pd.DataFrame) -> None:
    st.subheader("Prediction activity")
    if predictions.empty:
        st.info("No prediction logs found yet in `data/logs/predictions.jsonl`.")
        return

    activity = (
        predictions.dropna(subset=["timestamp"])
        .assign(minute=lambda df: df["timestamp"].dt.floor("min"))
        .groupby("minute")
        .size()
        .rename("requests")
        .reset_index()
        .set_index("minute")
    )
    if not activity.empty:
        st.line_chart(activity)

    variant_counts = predictions["variant"].fillna("unknown").value_counts().rename_axis("variant")
    st.bar_chart(variant_counts)


def _render_feedback_activity(feedback: pd.DataFrame) -> None:
    st.subheader("Feedback activity")
    if feedback.empty:
        st.info("No feedback logs found yet in `data/logs/feedback.jsonl`.")
        return

    action_counts = feedback["action"].fillna("unknown").value_counts().rename_axis("action")
    st.bar_chart(action_counts)

    hourly = (
        feedback.dropna(subset=["logged_at"])
        .assign(hour=lambda df: df["logged_at"].dt.floor("h"))
        .groupby(["hour", "action"])
        .size()
        .rename("events")
        .reset_index()
    )
    if not hourly.empty:
        pivot = hourly.pivot(index="hour", columns="action", values="events").fillna(0)
        st.line_chart(pivot)


def _render_recent_tables(predictions: pd.DataFrame, feedback: pd.DataFrame) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Recent predictions")
        if predictions.empty:
            st.write("No prediction rows available.")
        else:
            display = predictions.copy()
            display["items"] = display["items"].apply(lambda items: ", ".join(items[:5]))
            st.dataframe(
                display[["timestamp", "user_id", "variant", "latency_ms", "items"]]
                .tail(10)
                .sort_values("timestamp", ascending=False),
                use_container_width=True,
            )

    with right:
        st.subheader("Recent feedback")
        if feedback.empty:
            st.write("No feedback rows available.")
        else:
            st.dataframe(
                feedback[["logged_at", "timestamp", "user_id", "item_id", "action"]]
                .tail(10)
                .sort_values("logged_at", ascending=False),
                use_container_width=True,
            )


def _render_drift_section(log_dir: Path, report_dir: Path) -> None:
    st.subheader("Prediction drift")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Generate drift report", use_container_width=True):
            with st.spinner("Generating drift report..."):
                analyze_prediction_drift(log_dir=log_dir, report_dir=report_dir)
            st.cache_data.clear()

    summary = load_saved_summary(report_dir=report_dir)
    if not summary:
        st.info("No drift summary found yet. Generate a report once enough logs are available.")
        return

    status = summary.get("status", "unknown")
    reason = summary.get("reason")
    drift_detected = summary.get("drift_detected")
    drift_share = summary.get("drift_share")

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", status)
    c2.metric("Dataset drift", "yes" if drift_detected else "no")
    c3.metric("Drift share", f"{(drift_share or 0):.1%}")

    if reason:
        if status == "ok":
            st.caption(reason)
        elif status == "not_enough_data":
            st.warning(reason)
        else:
            st.info(reason)

    column_results = pd.DataFrame(summary.get("column_results", []))
    if not column_results.empty:
        st.dataframe(column_results, use_container_width=True)

    report_path = summary.get("report_path")
    if report_path and Path(report_path).exists() and status == "ok":
        with st.expander("Open Evidently HTML report", expanded=False):
            html = Path(report_path).read_text(encoding="utf-8")
            components.html(html, height=800, scrolling=True)


def _two_proportion_z_test(clicks_a: int, n_a: int, clicks_b: int, n_b: int) -> tuple[float, float]:
    """Returns (z_score, p_value) for two-proportion z-test (two-tailed)."""
    if n_a == 0 or n_b == 0:
        return 0.0, 1.0
    p_a = clicks_a / n_a
    p_b = clicks_b / n_b
    p_pool = (clicks_a + clicks_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return 0.0, 1.0
    z = (p_b - p_a) / se
    # Approximation: p-value from standard normal CDF
    p_value = 2 * (1 - _norm_cdf(abs(z)))
    return round(z, 4), round(p_value, 4)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _sample_size_per_variant(
    baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Minimum observations per variant for given MDE (minimum detectable effect)."""
    if baseline_rate <= 0 or mde <= 0 or baseline_rate >= 1:
        return 0
    z_alpha = _norm_ppf(1 - alpha / 2)
    z_beta = _norm_ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p2 = min(p2, 0.9999)
    n = (
        (z_alpha * math.sqrt(2 * p1 * (1 - p1)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
        ** 2
    ) / (mde**2)
    return math.ceil(n)


def _norm_ppf(p: float) -> float:
    """Rational approximation of the normal quantile function."""
    if p <= 0 or p >= 1:
        return float("inf")
    # Abramowitz & Stegun approximation
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    if p < 0.5:
        t = math.sqrt(-2 * math.log(p))
    else:
        t = math.sqrt(-2 * math.log(1 - p))
    num = c[0] + c[1] * t + c[2] * t**2
    den = 1 + d[0] * t + d[1] * t**2 + d[2] * t**3
    z = t - num / den
    return z if p >= 0.5 else -z


def _render_ab_test_section(predictions: pd.DataFrame, feedback: pd.DataFrame) -> None:
    st.subheader("A/B test analysis")

    # -- Conversion rate by variant over time --
    if not predictions.empty and not feedback.empty:
        merged = feedback.merge(
            predictions[["user_id", "variant", "timestamp"]].rename(
                columns={"timestamp": "pred_ts"}
            ),
            on="user_id",
            how="left",
        )
        clicks = merged[merged["action"] == "click"]
        if not clicks.empty:
            clicks_per_variant = clicks.groupby("variant").size().rename("clicks")
            preds_per_variant = predictions.groupby("variant").size().rename("n")
            ab = pd.concat([preds_per_variant, clicks_per_variant], axis=1).fillna(0)
            ab["conversion_rate"] = ab["clicks"] / ab["n"].replace(0, float("nan"))
            st.dataframe(ab.style.format({"conversion_rate": "{:.2%}"}), use_container_width=True)
        else:
            st.info("No click events yet to compute conversion rates.")
    else:
        st.info("Awaiting prediction and feedback data.")

    st.divider()

    # -- Z-test calculator --
    st.markdown("**Statistical significance calculator** (two-proportion z-test)")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Variant A")
        n_a = st.number_input("Impressions A", min_value=0, value=1000, step=100, key="na")
        clicks_a = st.number_input("Clicks A", min_value=0, value=50, step=5, key="ca")
    with c2:
        st.caption("Variant B")
        n_b = st.number_input("Impressions B", min_value=0, value=1000, step=100, key="nb")
        clicks_b = st.number_input("Clicks B", min_value=0, value=65, step=5, key="cb")

    z, p = _two_proportion_z_test(int(clicks_a), int(n_a), int(clicks_b), int(n_b))
    significant = p < 0.05
    r1, r2, r3 = st.columns(3)
    r1.metric("Z-score", f"{z:.3f}")
    r2.metric("p-value", f"{p:.4f}")
    r3.metric("Significant (α=0.05)", "Yes" if significant else "No")
    if significant:
        rate_a = clicks_a / max(n_a, 1)
        rate_b = clicks_b / max(n_b, 1)
        lift = (rate_b - rate_a) / max(rate_a, 1e-9)
        st.success(f"B outperforms A by {lift:+.1%} (statistically significant).")
    else:
        st.warning("Difference is not statistically significant yet.")

    st.divider()

    # -- Sample size calculator --
    st.markdown("**Sample size calculator**")
    s1, s2, s3 = st.columns(3)
    with s1:
        baseline = st.number_input(
            "Baseline CTR", min_value=0.001, max_value=0.999, value=0.05, step=0.005, format="%.3f"
        )
    with s2:
        mde = st.number_input(
            "Min. detectable effect (absolute)",
            min_value=0.001,
            max_value=0.5,
            value=0.01,
            step=0.005,
            format="%.3f",
        )
    with s3:
        power = st.selectbox("Power", options=[0.80, 0.90, 0.95], index=0)

    n_needed = _sample_size_per_variant(baseline, mde, power=float(power))
    st.metric("Required observations per variant", f"{n_needed:,}")


def main() -> None:
    st.title("RecEngine monitoring dashboard")
    st.caption("Operational view over recommendation traffic, feedback, and prediction drift.")

    default_log_dir = str(LOG_DIR)
    default_report_dir = str(REPORT_DIR)

    with st.sidebar:
        st.header("Settings")
        log_dir = Path(st.text_input("Log directory", value=default_log_dir))
        report_dir = Path(st.text_input("Report directory", value=default_report_dir))
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()

    predictions = _load_predictions(str(log_dir))
    feedback = _load_feedback(str(log_dir))

    _render_kpis(predictions, feedback)
    st.divider()

    left, right = st.columns(2)
    with left:
        _render_prediction_activity(predictions)
    with right:
        _render_feedback_activity(feedback)

    st.divider()
    _render_recent_tables(predictions, feedback)
    st.divider()
    _render_ab_test_section(predictions, feedback)
    st.divider()
    _render_drift_section(log_dir=log_dir, report_dir=report_dir)


if __name__ == "__main__":
    main()
