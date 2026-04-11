from __future__ import annotations

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
    _render_drift_section(log_dir=log_dir, report_dir=report_dir)


if __name__ == "__main__":
    main()
