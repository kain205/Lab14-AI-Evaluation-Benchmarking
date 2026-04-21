import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Benchmark Analytics", layout="wide")

st.title("🚀 AI Agent Benchmark Analytics")

# ─── Load data ──────────────────────────────────────────────────────────────
reports_dir = Path("reports")

# Load summary
summary_file = reports_dir / "summary.json"
results_file = reports_dir / "benchmark_results.json"

if not summary_file.exists() or not results_file.exists():
    st.error("❌ Missing reports. Run `python main.py` first to generate benchmark data.")
    st.stop()

with open(summary_file, encoding="utf-8") as f:
    summary = json.load(f)

with open(results_file, encoding="utf-8") as f:
    results = json.load(f)

# ─── Parse data ─────────────────────────────────────────────────────────────
v1_summary = summary["summaries"]["V1-Rewrite"]
v3_summary = summary["summaries"]["V3-Clarify"]

v1_results = results["V1-Rewrite"]
v3_results = results["V3-Clarify"]

# Convert to DataFrames
def results_to_df(results_list, agent_name):
    df = pd.DataFrame([{
        "Agent": agent_name,
        "Question": r.get("test_case", ""),
        "Response": r.get("agent_response", ""),
        "Judge Score": r["judge"]["final_score"],
        "Agreement Rate": r["judge"]["agreement_rate"],
        "Hit Rate": r["ragas"].get("hit_rate", 0) or r["ragas"].get("retrieval", {}).get("hit_rate", 0),
        "MRR": r["ragas"].get("mrr", 0) or r["ragas"].get("retrieval", {}).get("mrr", 0),
        "Latency (s)": r["latency"],
        "Status": r.get("status", ""),
    } for r in results_list])
    return df

v1_df = results_to_df(v1_results, "V1-Rewrite")
v3_df = results_to_df(v3_results, "V3-Clarify")
combined_df = pd.concat([v1_df, v3_df], ignore_index=True)

# ─── Dashboard sections ──────────────────────────────────────────────────────

# 1. Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("V1 Avg Score", f"{v1_summary['metrics']['avg_score']:.2f}")
with col2:
    st.metric("V3 Avg Score", f"{v3_summary['metrics']['avg_score']:.2f}")
with col3:
    delta_score = v3_summary['metrics']['avg_score'] - v1_summary['metrics']['avg_score']
    st.metric("Delta (V3-V1)", f"{delta_score:+.2f}", delta_color=("off" if -0.10 <= delta_score <= 0.30 else "inverse"))
with col4:
    st.metric("Test Cases", len(v1_df))

# Hit rates
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("V1 Hit Rate", f"{v1_summary['metrics']['hit_rate']:.2f}")
with col2:
    st.metric("V3 Hit Rate", f"{v3_summary['metrics']['hit_rate']:.2f}")
with col3:
    hit_delta = v3_summary['metrics']['hit_rate'] - v1_summary['metrics']['hit_rate']
    st.metric("Hit Rate Delta", f"{hit_delta:+.2f}")

st.divider()

# 2. Comparison charts
col1, col2 = st.columns(2)

with col1:
    # Score distribution
    fig = go.Figure()
    fig.add_trace(go.Box(y=v1_df["Judge Score"], name="V1-Rewrite", marker_color="blue"))
    fig.add_trace(go.Box(y=v3_df["Judge Score"], name="V3-Clarify", marker_color="orange"))
    fig.update_layout(title="Judge Score Distribution", height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Hit rate by agent
    fig = go.Figure()
    fig.add_trace(go.Box(y=v1_df["Hit Rate"], name="V1-Rewrite", marker_color="blue"))
    fig.add_trace(go.Box(y=v3_df["Hit Rate"], name="V3-Clarify", marker_color="orange"))
    fig.update_layout(title="Hit Rate Distribution", height=400, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# 3. Detailed results table
st.subheader("📊 Detailed Results")

tab1, tab2, tab3 = st.tabs(["V1-Rewrite", "V3-Clarify", "Combined"])

with tab1:
    st.dataframe(
        v1_df[["Question", "Judge Score", "Hit Rate", "Agreement Rate", "Latency (s)"]].sort_values("Judge Score", ascending=False),
        use_container_width=True,
        height=400
    )

with tab2:
    st.dataframe(
        v3_df[["Question", "Judge Score", "Hit Rate", "Agreement Rate", "Latency (s)"]].sort_values("Judge Score", ascending=False),
        use_container_width=True,
        height=400
    )

with tab3:
    st.dataframe(
        combined_df[["Agent", "Judge Score", "Hit Rate", "Agreement Rate", "Latency (s)"]].sort_values("Judge Score", ascending=False),
        use_container_width=True,
        height=400
    )

# 4. Analysis & filters
st.subheader("🔍 Analysis")

col1, col2 = st.columns(2)

with col1:
    # Filter by score
    score_filter = st.slider("Filter by Judge Score", 0.0, 10.0, (0.0, 10.0))
    filtered_df = combined_df[
        (combined_df["Judge Score"] >= score_filter[0]) &
        (combined_df["Judge Score"] <= score_filter[1])
    ]
    st.metric("Filtered Cases", len(filtered_df))

with col2:
    agent_filter = st.multiselect("Filter by Agent", combined_df["Agent"].unique(), default=combined_df["Agent"].unique())
    filtered_df = filtered_df[filtered_df["Agent"].isin(agent_filter)]

# Show filtered data
st.write(f"**Showing {len(filtered_df)} cases:**")
st.dataframe(
    filtered_df[["Agent", "Judge Score", "Hit Rate", "Agreement Rate"]],
    use_container_width=True,
    height=300
)

# 5. Regression report
st.subheader("📈 Regression Report")

regression_file = reports_dir / "regression_cases.txt"
if regression_file.exists():
    with open(regression_file, encoding="utf-8") as f:
        regression_content = f.read()
    st.text_area("Regression Cases", regression_content, height=400, disabled=True)
else:
    st.info("No regression report generated yet.")

# 6. Release gate
st.subheader("🎯 Release Gate Decision")

gates = summary["gates"]
release_delta = summary["delta_v3_vs_v1"]
release_hit = summary["delta_hit_rate_v3_vs_v1"]

col1, col2, col3 = st.columns(3)

with col1:
    passed = release_delta >= gates["release_delta_tolerance"]
    st.metric("Score Delta Gate",
              f"{release_delta:+.3f} (thresh: {gates['release_delta_tolerance']})",
              delta="✅ PASS" if passed else "❌ FAIL")

with col2:
    passed = release_hit >= gates["release_hit_rate_tolerance"]
    st.metric("Hit Rate Gate",
              f"{release_hit:+.3f} (thresh: {gates['release_hit_rate_tolerance']})",
              delta="✅ PASS" if passed else "❌ FAIL")

with col3:
    all_pass = (release_delta >= gates["release_delta_tolerance"] and
                release_hit >= gates["release_hit_rate_tolerance"])
    status = "✅ APPROVE" if all_pass else "❌ BLOCK"
    st.metric("Release Decision", status)

st.divider()
st.caption(f"Generated: {v1_summary['metadata']['timestamp']}")
