"""
Airline Loyalty & Retention Intelligence Platform
Streamlit Dashboard — Main Application

Pages:
  1. 🏠 Executive Overview
  2. ⚠️  Churn Risk Dashboard
  3. 👥  Customer Segments
  4. 🎯  Retention Actions
  5. 🔍  Individual Customer Lookup
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="AirLoyalty Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "analytics", "scored_customers.parquet")
FLIGHT_FILE = os.path.join(BASE_DIR, "Customer Flight Activity.csv")

SEGMENT_COLORS = {
    "Champions":         "#00C9A7",
    "Loyalists":         "#4F8EF7",
    "At-Risk Actives":   "#FF6B6B",
    "Occasional Flyers": "#FFD166",
    "Dormant":           "#A8A8B3",
}
RISK_COLORS = {
    "Critical": "#FF3B30",
    "High":     "#FF9500",
    "Medium":   "#FFCC00",
    "Low":      "#34C759",
}
SEGMENT_ICONS = {
    "Champions":         "🏆",
    "Loyalists":         "💎",
    "At-Risk Actives":   "⚠️",
    "Occasional Flyers": "✈️",
    "Dormant":           "💤",
}

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark background */
.stApp {
    background: linear-gradient(135deg, #0d0f1a 0%, #141624 50%, #0d0f1a 100%);
    color: #e8eaf6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0c16 0%, #131525 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4F8EF7, #00C9A7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.8rem;
    color: rgba(232,234,246,0.6);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
    font-weight: 500;
}
.metric-delta-up   { color: #00C9A7; font-size: 0.85rem; }
.metric-delta-down { color: #FF6B6B; font-size: 0.85rem; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: rgba(232,234,246,0.9);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 28px 0 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding-bottom: 8px;
}

/* Segment badge */
.seg-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* Risk badge */
.risk-critical { background: rgba(255,59,48,0.2);  color: #FF3B30; border: 1px solid #FF3B30; border-radius: 8px; padding: 3px 10px; font-weight: 700; font-size: 0.78rem; }
.risk-high     { background: rgba(255,149,0,0.2);  color: #FF9500; border: 1px solid #FF9500; border-radius: 8px; padding: 3px 10px; font-weight: 700; font-size: 0.78rem; }
.risk-medium   { background: rgba(255,204,0,0.2);  color: #FFCC00; border: 1px solid #FFCC00; border-radius: 8px; padding: 3px 10px; font-weight: 700; font-size: 0.78rem; }
.risk-low      { background: rgba(52,199,89,0.2);  color: #34C759; border: 1px solid #34C759; border-radius: 8px; padding: 3px 10px; font-weight: 700; font-size: 0.78rem; }

/* Action card */
.action-card {
    background: linear-gradient(135deg, rgba(79,142,247,0.08) 0%, rgba(0,201,167,0.05) 100%);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.action-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #4F8EF7;
    margin-bottom: 8px;
}
.action-detail {
    font-size: 0.88rem;
    color: rgba(232,234,246,0.8);
    line-height: 1.6;
}
.action-meta {
    display: flex;
    gap: 16px;
    margin-top: 12px;
    flex-wrap: wrap;
}
.action-meta-item {
    font-size: 0.78rem;
    color: rgba(232,234,246,0.5);
}
.action-meta-item span {
    color: rgba(232,234,246,0.9);
    font-weight: 600;
}

/* Table styling */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* Profile card */
.profile-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
}
.profile-name {
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: 4px;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a1f3e 0%, #0d1b2a 50%, #1a2a1a 100%);
    border: 1px solid rgba(79,142,247,0.15);
    border-radius: 20px;
    padding: 32px 36px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4F8EF7 0%, #00C9A7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 8px;
}
.hero-subtitle {
    font-size: 0.95rem;
    color: rgba(232,234,246,0.6);
    font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_scored_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_parquet(DATA_PATH)
    # Ensure string types for categorical columns
    for col in ["churn_risk_tier", "segment", "churn_reason"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_flight_activity() -> pd.DataFrame:
    df = pd.read_csv(FLIGHT_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["period_date"] = pd.to_datetime(
        df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2) + "-01"
    )
    return df


def run_pipeline_with_progress():
    """Run the full pipeline with Streamlit progress indicators."""
    import subprocess, sys
    placeholder = st.empty()
    with placeholder.container():
        st.info("🔄 Running analytics pipeline for the first time. This takes 2-4 minutes…")
        progress = st.progress(0)
        status   = st.empty()

        steps = [
            ("Loading data …",          "from analytics.data_loader import build_master_dataset; loyalty, activity = build_master_dataset()"),
            ("Engineering features …",  "from analytics.feature_engineering import build_features; features = build_features()"),
            ("Training churn model …",  "from analytics.churn_model import train_model, predict_churn; r = train_model(features); features = predict_churn(features, r['model'], r['scaler'])"),
            ("Training segmentation …", "from analytics.segmentation import train_segmentation; sr = train_segmentation(features); features = sr['features']"),
            ("Applying retention rules …", "from analytics.retention import apply_retention_rules; features = apply_retention_rules(features); features.to_parquet('analytics/scored_customers.parquet', index=False)"),
        ]

        # Run in a single subprocess to avoid re-import issues
        result = subprocess.run(
            [sys.executable, "run_pipeline.py"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            st.error(f"Pipeline failed:\n{result.stderr[-2000:]}")
            st.stop()

        progress.progress(100)
        status.success("✅ Pipeline complete!")
    placeholder.empty()
    st.rerun()


# ---------------------------------------------------------------------------
# PLOTLY THEME
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e8eaf6"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(255,255,255,0.04)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)"),
)


# ---------------------------------------------------------------------------
# PAGE: EXECUTIVE OVERVIEW
# ---------------------------------------------------------------------------

def page_overview(df: pd.DataFrame):
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">✈️ AirLoyalty Intelligence Platform</div>
        <div class="hero-subtitle">Behavioral analytics for ~16,700 Canadian loyalty members · 2012–2018 · Powered by ML</div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Cards ──
    total      = len(df)
    active     = (df["churn_label"] == 0).sum()
    churn_rate = df["churn_label"].mean()
    avg_clv    = df["CLV"].mean()
    critical   = (df["churn_risk_tier"] == "Critical").sum()
    revenue_at_risk = df[df["churn_risk_tier"].isin(["Critical", "High"])]["CLV"].sum()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, f"{total:,}",         "Total Members"),
        (c2, f"{active:,}",        "Active Members"),
        (c3, f"{churn_rate:.1%}",  "Churn Rate"),
        (c4, f"${avg_clv:,.0f}",   "Avg CLV"),
        (c5, f"{critical:,}",      "Critical Risk"),
        (c6, f"${revenue_at_risk/1e6:.1f}M", "Revenue at Risk"),
    ]
    for col, val, label in kpis:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: CLV distribution + Risk donut ──
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-header">CLV Distribution by Card Tier</div>', unsafe_allow_html=True)
        fig = px.violin(
            df[df["CLV"] < df["CLV"].quantile(0.99)],
            x="Loyalty Card", y="CLV",
            color="Loyalty Card",
            box=True,
            color_discrete_map={"Star": "#A8A8B3", "Nova": "#4F8EF7", "Aurora": "#00C9A7"},
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Churn Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["churn_risk_tier"].value_counts().reindex(
            ["Critical", "High", "Medium", "Low"], fill_value=0
        )
        fig = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.62,
            marker=dict(colors=[RISK_COLORS.get(r, "#888") for r in risk_counts.index]),
            textinfo="label+percent",
            hovertemplate="<b>%{label}</b><br>%{value:,} members<extra></extra>",
        ))
        fig.add_annotation(text=f"<b>{churn_rate:.0%}</b><br>churn", x=0.5, y=0.5,
                           font=dict(size=18, color="#e8eaf6"), showarrow=False)
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Segment overview + Province heatmap ──
    col3, col4 = st.columns([2, 3])

    with col3:
        st.markdown('<div class="section-header">Segment Breakdown</div>', unsafe_allow_html=True)
        seg_data = df.groupby("segment").agg(
            count=("Loyalty Number", "count"),
            avg_clv=("CLV", "mean"),
            churn_rate=("churn_label", "mean"),
        ).reset_index()
        seg_data["icon"] = seg_data["segment"].map(SEGMENT_ICONS)
        seg_data["label"] = seg_data["icon"] + " " + seg_data["segment"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_data["count"],
            y=seg_data["label"],
            orientation="h",
            marker=dict(
                color=[SEGMENT_COLORS.get(s, "#888") for s in seg_data["segment"]],
                opacity=0.85,
            ),
            text=seg_data["count"].apply(lambda x: f"{x:,}"),
            textposition="inside",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>Avg CLV: $%{customdata[0]:,.0f}<br>Churn Rate: %{customdata[1]:.1%}<extra></extra>",
            customdata=seg_data[["avg_clv", "churn_rate"]].values,
        ))
        fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="Members", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Churn Risk by Province</div>', unsafe_allow_html=True)
        prov_data = df.groupby("Province").agg(
            count=("Loyalty Number", "count"),
            churn_rate=("churn_label", "mean"),
            avg_clv=("CLV", "mean"),
            critical_count=(
                "churn_risk_tier",
                lambda x: (x == "Critical").sum(),
            ),
        ).reset_index().sort_values("churn_rate", ascending=False).head(12)

        fig = px.bar(
            prov_data,
            x="Province", y="churn_rate",
            color="churn_rate",
            color_continuous_scale=["#00C9A7", "#FFD166", "#FF6B6B"],
            text=prov_data["churn_rate"].apply(lambda x: f"{x:.0%}"),
            hover_data={"count": True, "avg_clv": ":.0f", "critical_count": True},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                          xaxis_title="", yaxis_title="Churn Rate",
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Monthly flight trend ──
    st.markdown('<div class="section-header">Network-Wide Monthly Flight Activity (2012–2018)</div>', unsafe_allow_html=True)
    activity = load_flight_activity()
    monthly = activity.groupby("period_date")["Total Flights"].sum().reset_index()
    monthly.columns = ["Date", "Total Flights"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Date"], y=monthly["Total Flights"],
        mode="lines", fill="tozeroy",
        line=dict(color="#4F8EF7", width=2),
        fillcolor="rgba(79,142,247,0.1)",
        name="Total Flights",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, xaxis_title="", yaxis_title="Total Flights Booked",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: CHURN RISK DASHBOARD
# ---------------------------------------------------------------------------

def page_churn(df: pd.DataFrame):
    st.markdown('<div class="hero-title" style="font-size:1.8rem; margin-bottom:4px;">⚠️ Churn Risk Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="margin-bottom:24px;">Identify and act on members likely to disengage</div>', unsafe_allow_html=True)

    # ── Filters ──
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        risk_filter = st.multiselect(
            "Risk Tier", ["Critical", "High", "Medium", "Low"],
            default=["Critical", "High"],
        )
    with col_f2:
        seg_filter = st.multiselect(
            "Segment", df["segment"].unique().tolist(),
            default=df["segment"].unique().tolist(),
        )
    with col_f3:
        card_filter = st.multiselect(
            "Card Tier", ["Star", "Nova", "Aurora"],
            default=["Star", "Nova", "Aurora"],
        )
    with col_f4:
        top_n = st.slider("Show top N customers", 25, 500, 100, 25)

    filtered = df[
        (df["churn_risk_tier"].isin(risk_filter)) &
        (df["segment"].isin(seg_filter)) &
        (df["Loyalty Card"].isin(card_filter))
    ].sort_values("churn_probability", ascending=False).head(top_n)

    # ── Summary stats for filtered view ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(filtered):,}</div><div class="metric-label">Customers Shown</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{filtered['churn_probability'].mean():.0%}</div><div class="metric-label">Avg Churn Prob</div></div>""", unsafe_allow_html=True)
    with c3:
        rev_risk = filtered["CLV"].sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${rev_risk/1e6:.1f}M</div><div class="metric-label">CLV at Risk</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{filtered['churn_reason'].value_counts().index[0] if len(filtered) > 0 else 'N/A'}</div><div class="metric-label">Top Churn Reason</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Scatter: churn prob vs CLV ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">Churn Probability vs Customer Lifetime Value</div>', unsafe_allow_html=True)
        plot_df = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(
            plot_df,
            x="CLV", y="churn_probability",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.65,
            size_max=8,
            hover_data={"Loyalty Number": True, "Loyalty Card": True,
                        "churn_risk_tier": True, "CLV": ":.0f"},
            labels={"churn_probability": "Churn Probability", "CLV": "Customer Lifetime Value ($)"},
        )
        # Add quadrant lines
        fig.add_hline(y=0.55, line_dash="dash", line_color="rgba(255,100,100,0.4)",
                      annotation_text="High Risk Threshold", annotation_position="top right")
        fig.add_vline(x=df["CLV"].median(), line_dash="dash",
                      line_color="rgba(255,255,255,0.2)",
                      annotation_text="Median CLV", annotation_position="top left")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Risk × Card Tier Heatmap</div>', unsafe_allow_html=True)
        heat = df.groupby(["Loyalty Card", "churn_risk_tier"]).size().unstack(fill_value=0)
        heat = heat.reindex(columns=["Critical", "High", "Medium", "Low"], fill_value=0)
        heat = heat.reindex(["Aurora", "Nova", "Star"], fill_value=0)

        fig = px.imshow(
            heat,
            color_continuous_scale=["#001933", "#1a3a5c", "#FF9500", "#FF3B30"],
            text_auto=True,
            aspect="auto",
        )
        fig.update_layout(**PLOTLY_LAYOUT, coloraxis_showscale=False,
                          xaxis_title="Risk Tier", yaxis_title="Card Tier")
        st.plotly_chart(fig, use_container_width=True)

    # ── Customer Table ──
    st.markdown('<div class="section-header">High-Risk Customer List</div>', unsafe_allow_html=True)

    display_cols = {
        "Loyalty Number": "ID",
        "Loyalty Card": "Card",
        "segment": "Segment",
        "churn_risk_tier": "Risk",
        "churn_probability": "Churn Prob",
        "CLV": "CLV ($)",
        "recency_days": "Days Since Last Flight",
        "trailing_inactive_months": "Inactive Months",
        "churn_reason": "Churn Reason",
        "action_title": "Recommended Action",
    }

    table = filtered[list(display_cols.keys())].rename(columns=display_cols).copy()
    table["Churn Prob"] = table["Churn Prob"].apply(lambda x: f"{x:.1%}")
    table["CLV ($)"]    = table["CLV ($)"].apply(lambda x: f"${x:,.0f}")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=380,
    )

    # Download button
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered customer list (CSV)",
        csv, "churn_risk_customers.csv", "text/csv",
    )


# ---------------------------------------------------------------------------
# PAGE: CUSTOMER SEGMENTS
# ---------------------------------------------------------------------------

def page_segments(df: pd.DataFrame):
    st.markdown('<div class="hero-title" style="font-size:1.8rem; margin-bottom:4px;">👥 Customer Segments</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="margin-bottom:24px;">Five behaviorally distinct groups — each requiring a different strategy</div>', unsafe_allow_html=True)

    # ── Segment cards ──
    seg_stats = df.groupby("segment").agg(
        count=("Loyalty Number", "count"),
        avg_clv=("CLV", "mean"),
        churn_rate=("churn_label", "mean"),
        avg_flights=("flights_12m", "mean"),
        avg_tenure=("months_since_enrollment", "mean"),
    ).reset_index()

    cols = st.columns(5)
    for i, row in seg_stats.iterrows():
        seg = row["segment"]
        icon = SEGMENT_ICONS.get(seg, "")
        color = SEGMENT_COLORS.get(seg, "#888")
        with cols[i % 5]:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {color}40;">
                <div style="font-size:2rem; margin-bottom:6px;">{icon}</div>
                <div style="font-size:0.95rem; font-weight:700; color:{color}; margin-bottom:8px;">{seg}</div>
                <div style="font-size:1.4rem; font-weight:800; color:#e8eaf6;">{row['count']:,}</div>
                <div class="metric-label">members</div>
                <div style="margin-top:10px; font-size:0.78rem; color:rgba(232,234,246,0.6);">
                    Avg CLV: <b style="color:#e8eaf6;">${row['avg_clv']:,.0f}</b><br>
                    Churn Rate: <b style="color:{'#FF6B6B' if row['churn_rate'] > 0.3 else '#00C9A7'};">{row['churn_rate']:.0%}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PCA scatter ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<div class="section-header">Segment Map (PCA 2D Projection)</div>', unsafe_allow_html=True)
        plot_df = df.sample(min(4000, len(df)), random_state=42)
        fig = px.scatter(
            plot_df, x="pca_x", y="pca_y",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
            opacity=0.6,
            hover_data={"Loyalty Number": True, "CLV": ":.0f",
                        "churn_probability": ":.2f", "Loyalty Card": True},
            labels={"pca_x": "Component 1", "pca_y": "Component 2"},
        )
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">CLV by Segment (Box Plot)</div>', unsafe_allow_html=True)
        fig = px.box(
            df[df["CLV"] < df["CLV"].quantile(0.98)],
            x="segment", y="CLV",
            color="segment",
            color_discrete_map=SEGMENT_COLORS,
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False,
                          xaxis_title="", yaxis_title="CLV ($)")
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    # ── Radar chart ──
    st.markdown('<div class="section-header">Segment Behavioral Radar</div>', unsafe_allow_html=True)

    radar_metrics = {
        "Avg Monthly Flights": "avg_monthly_flights",
        "Recency Score": lambda d: 1 - (d["recency_days"].clip(0, 365) / 365),
        "Redemption Ratio": "redemption_ratio",
        "CLV Score": lambda d: (d["CLV"] / d["CLV"].max()).clip(0, 1),
        "Tenure Score": lambda d: (d["months_since_enrollment"] / d["months_since_enrollment"].max()).clip(0, 1),
        "6M Activity": lambda d: (d["flights_6m"] / d["flights_6m"].max().clip(1)).clip(0, 1),
    }

    # Compute normalized per-segment averages
    radar_data = {}
    for label, col in radar_metrics.items():
        if callable(col):
            seg_vals = df.copy()
            seg_vals["_computed"] = col(df)
            radar_data[label] = seg_vals.groupby("segment")["_computed"].mean()
        else:
            max_val = df[col].max()
            radar_data[label] = df.groupby("segment")[col].mean() / max(max_val, 1e-9)

    radar_df = pd.DataFrame(radar_data)
    categories = list(radar_metrics.keys())

    fig = go.Figure()
    for seg in radar_df.index:
        values = radar_df.loc[seg].tolist()
        values += [values[0]]  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill="toself",
            name=f"{SEGMENT_ICONS.get(seg,'')} {seg}",
            line=dict(color=SEGMENT_COLORS.get(seg, "#888"), width=2),
            fillcolor="rgba({},{},{},0.12)".format(
                int(SEGMENT_COLORS.get(seg, "#888888")[1:3], 16),
                int(SEGMENT_COLORS.get(seg, "#888888")[3:5], 16),
                int(SEGMENT_COLORS.get(seg, "#888888")[5:7], 16),
            ),
            opacity=0.85,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(255,255,255,0.08)",
                            tickfont=dict(color="rgba(232,234,246,0.4)", size=9)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Full segment comparison table ──
    st.markdown('<div class="section-header">Segment Comparison Table</div>', unsafe_allow_html=True)
    comparison = df.groupby("segment").agg(
        Members=("Loyalty Number", "count"),
        **{"Avg CLV ($)":       ("CLV", lambda x: f"${x.mean():,.0f}")},
        **{"Median CLV ($)":    ("CLV", lambda x: f"${x.median():,.0f}")},
        **{"Avg Flights/12m":   ("flights_12m", lambda x: f"{x.mean():.1f}")},
        **{"Avg Recency (days)":("recency_days", lambda x: f"{x.mean():.0f}")},
        **{"Churn Rate":        ("churn_label", lambda x: f"{x.mean():.0%}")},
        **{"Avg Tenure (mo)":   ("months_since_enrollment", lambda x: f"{x.mean():.0f}")},
        **{"Redemption Ratio":  ("redemption_ratio", lambda x: f"{x.mean():.2f}")},
    ).reset_index()
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# PAGE: RETENTION ACTIONS
# ---------------------------------------------------------------------------

def page_retention(df: pd.DataFrame):
    st.markdown('<div class="hero-title" style="font-size:1.8rem; margin-bottom:4px;">🎯 Retention Actions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle" style="margin-bottom:24px;">Specific, actionable interventions — ready to hand to the operations team</div>', unsafe_allow_html=True)

    # ── Filters ──
    col1, col2, col3 = st.columns(3)
    with col1:
        sel_seg  = st.selectbox("Segment", ["All"] + sorted(df["segment"].unique().tolist()))
    with col2:
        sel_risk = st.selectbox("Risk Tier", ["All", "Critical", "High", "Medium", "Low"])
    with col3:
        priority_only = st.checkbox("Priority 1 & 2 only (immediate action)", value=True)

    filt = df.copy()
    if sel_seg  != "All": filt = filt[filt["segment"] == sel_seg]
    if sel_risk != "All": filt = filt[filt["churn_risk_tier"] == sel_risk]
    if priority_only:     filt = filt[filt["action_priority"] <= 2]
    filt = filt.sort_values(["action_priority", "churn_probability"], ascending=[True, False])

    # ── Summary ──
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-value">{len(filt):,}</div><div class="metric-label">Customers Needing Action</div></div>""", unsafe_allow_html=True)
    with c2:
        clv_risk = filt["CLV"].sum()
        st.markdown(f"""<div class="metric-card"><div class="metric-value">${clv_risk/1e6:.1f}M</div><div class="metric-label">Total CLV at Stake</div></div>""", unsafe_allow_html=True)
    with c3:
        top_action = filt["action_title"].value_counts().index[0] if len(filt) > 0 else "N/A"
        st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size:1rem;">{top_action[:30]}…</div><div class="metric-label">Most Common Action</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action cards (top 20) ──
    st.markdown('<div class="section-header">Action Cards — Top 20 Priority Customers</div>', unsafe_allow_html=True)
    top20 = filt.head(20)
    for _, row in top20.iterrows():
        seg   = row.get("segment", "")
        risk  = str(row.get("churn_risk_tier", ""))
        risk_class = f"risk-{risk.lower()}"
        seg_color  = SEGMENT_COLORS.get(seg, "#888")
        icon  = SEGMENT_ICONS.get(seg, "")
        st.markdown(f"""
        <div class="action-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <div>
                    <span style="font-weight:700; color:#e8eaf6; font-size:1rem;">
                        #{row['Loyalty Number']} · {row.get('Loyalty Card','?')} Member
                    </span>
                    <span style="margin-left:12px; background:{seg_color}20; color:{seg_color};
                                 border:1px solid {seg_color}40; border-radius:20px;
                                 padding:2px 10px; font-size:0.72rem; font-weight:600;">
                        {icon} {seg}
                    </span>
                </div>
                <span class="{risk_class}">{risk}</span>
            </div>
            <div class="action-title">🎯 {row.get('action_title','')}</div>
            <div class="action-detail">{row.get('action_detail','')}</div>
            <div class="action-meta">
                <div class="action-meta-item">📅 Timing: <span>{row.get('action_timing','')}</span></div>
                <div class="action-meta-item">📣 Channel: <span>{row.get('action_channel','')}</span></div>
                <div class="action-meta-item">💰 CLV: <span>${row.get('CLV',0):,.0f}</span></div>
                <div class="action-meta-item">📊 Churn Prob: <span>{row.get('churn_probability',0):.0%}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Action summary by type ──
    st.markdown('<div class="section-header">Action Distribution</div>', unsafe_allow_html=True)
    action_counts = df.groupby(["action_title", "segment"]).size().reset_index(name="count")
    fig = px.bar(
        action_counts.sort_values("count", ascending=True).tail(20),
        x="count", y="action_title", color="segment",
        orientation="h",
        color_discrete_map=SEGMENT_COLORS,
        labels={"count": "Customers", "action_title": ""},
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Download
    csv = filt[[
        "Loyalty Number", "Loyalty Card", "segment", "churn_risk_tier",
        "churn_probability", "CLV", "action_title", "action_detail",
        "action_channel", "action_timing", "revenue_at_risk_label",
    ]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Retention Action Plan (CSV)",
        csv, "retention_actions.csv", "text/csv",
    )


# ---------------------------------------------------------------------------
# PAGE: INDIVIDUAL CUSTOMER LOOKUP
# ---------------------------------------------------------------------------

def page_lookup(df: pd.DataFrame, activity: pd.DataFrame):
    st.markdown('<div class="hero-title" style="font-size:1.8rem; margin-bottom:4px;">🔍 Individual Customer Lookup</div>', unsafe_allow_html=True)

    # ── Search ──
    search_col, _ = st.columns([2, 3])
    with search_col:
        loyalty_ids = sorted(df["Loyalty Number"].astype(str).tolist())
        selected_id = st.selectbox(
            "Search by Loyalty Number",
            loyalty_ids,
            index=0,
        )

    if not selected_id:
        st.info("Enter a loyalty number to view the customer profile.")
        return

    row = df[df["Loyalty Number"].astype(str) == str(selected_id)]
    if row.empty:
        st.error("Customer not found.")
        return
    row = row.iloc[0]

    seg   = row.get("segment", "Unknown")
    risk  = str(row.get("churn_risk_tier", "Unknown"))
    color = SEGMENT_COLORS.get(seg, "#888")
    icon  = SEGMENT_ICONS.get(seg, "")
    risk_color = RISK_COLORS.get(risk, "#888")

    # ── Profile card ──
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown(f"""
        <div class="profile-card">
            <div class="profile-name">#{row['Loyalty Number']}</div>
            <div style="color: rgba(232,234,246,0.5); font-size:0.85rem; margin-bottom:16px;">
                {row.get('City','?')}, {row.get('Province','?')}
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px;">
                <span style="background:{color}20; color:{color}; border:1px solid {color}40;
                             border-radius:20px; padding:4px 12px; font-size:0.78rem; font-weight:600;">
                    {icon} {seg}
                </span>
                <span style="background:{risk_color}20; color:{risk_color}; border:1px solid {risk_color}40;
                             border-radius:8px; padding:4px 12px; font-size:0.78rem; font-weight:700;">
                    {risk} Risk
                </span>
                <span style="background:rgba(255,255,255,0.06); color:rgba(232,234,246,0.8);
                             border:1px solid rgba(255,255,255,0.1);
                             border-radius:8px; padding:4px 12px; font-size:0.78rem; font-weight:600;">
                    {row.get('Loyalty Card','?')} Card
                </span>
            </div>
            <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Gender</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">{row.get('Gender','?')}</td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Education</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">{row.get('Education','?')}</td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Marital Status</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">{row.get('Marital Status','?')}</td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Salary</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">
                        {'$'+f"{row['Salary']:,.0f}" if pd.notna(row.get('Salary')) else 'Not disclosed'}
                    </td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">CLV</td>
                    <td style="color:#00C9A7; font-weight:700; text-align:right;">${row.get('CLV',0):,.2f}</td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Enrolled</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">
                        {int(row.get('Enrollment Month',0)):02d}/{int(row.get('Enrollment Year',0))}
                    </td></tr>
                <tr><td style="color:rgba(232,234,246,0.5); padding:4px 0;">Tenure</td>
                    <td style="color:#e8eaf6; font-weight:500; text-align:right;">
                        {row.get('months_since_enrollment',0):.0f} months
                    </td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # ── Churn probability gauge ──
        churn_prob = float(row.get("churn_probability", 0))
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=churn_prob * 100,
            title={"text": "Churn Probability (%)", "font": {"size": 16, "color": "#e8eaf6"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "rgba(232,234,246,0.4)"},
                "bar":  {"color": risk_color},
                "steps": [
                    {"range": [0, 30],  "color": "rgba(52,199,89,0.15)"},
                    {"range": [30, 55], "color": "rgba(255,204,0,0.15)"},
                    {"range": [55, 75], "color": "rgba(255,149,0,0.15)"},
                    {"range": [75, 100],"color": "rgba(255,59,48,0.15)"},
                ],
                "threshold": {
                    "line": {"color": risk_color, "width": 3},
                    "thickness": 0.75,
                    "value": churn_prob * 100,
                },
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(255,255,255,0.1)",
            },
            number={"suffix": "%", "font": {"color": risk_color, "size": 36}},
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e8eaf6"),
            height=260,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Key metrics ──
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size:1.5rem;">{row.get('total_flights_all',0):.0f}</div><div class="metric-label">Total Flights</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size:1.5rem;">{row.get('flights_12m',0):.0f}</div><div class="metric-label">Flights (12m)</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size:1.5rem;">{row.get('recency_days',0):.0f}</div><div class="metric-label">Days Since Last Flight</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card"><div class="metric-value" style="font-size:1.5rem;">{row.get('points_bank',0):,.0f}</div><div class="metric-label">Points Bank</div></div>""", unsafe_allow_html=True)

    # ── Flight history chart ──
    st.markdown('<div class="section-header">Flight History</div>', unsafe_allow_html=True)
    cust_act = activity[activity["Loyalty Number"] == int(selected_id)].sort_values("period_date")
    if not cust_act.empty:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=cust_act["period_date"], y=cust_act["Total Flights"],
            name="Flights", marker_color="#4F8EF7", opacity=0.7,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=cust_act["period_date"], y=cust_act["Points Accumulated"],
            name="Points Earned", line=dict(color="#00C9A7", width=2),
            mode="lines",
        ), secondary_y=True)
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=300,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1),
        )
        fig.update_yaxes(title_text="Flights Booked", secondary_y=False)
        fig.update_yaxes(title_text="Points Earned", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No flight activity records found for this customer.")

    # ── Retention Action ──
    st.markdown('<div class="section-header">Recommended Retention Action</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="action-card">
        <div class="action-title">🎯 {row.get('action_title','')}</div>
        <div class="action-detail">{row.get('action_detail','')}</div>
        <div class="action-meta">
            <div class="action-meta-item">📅 Timing: <span>{row.get('action_timing','')}</span></div>
            <div class="action-meta-item">📣 Channel: <span>{row.get('action_channel','')}</span></div>
            <div class="action-meta-item">⚡ Churn Reason: <span>{row.get('churn_reason','')}</span></div>
            <div class="action-meta-item">💰 Revenue at Risk: <span>{row.get('revenue_at_risk_label','')}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR & ROUTING
# ---------------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 10px 0;">
            <div style="font-size:2.5rem;">✈️</div>
            <div style="font-size:1.1rem; font-weight:800; 
                        background:linear-gradient(90deg,#4F8EF7,#00C9A7);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text;">
                AirLoyalty Intel
            </div>
            <div style="font-size:0.7rem; color:rgba(232,234,246,0.4); margin-top:4px;">
                Behavioral Intelligence Platform
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.06); margin:8px 0 20px 0;">
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Executive Overview",
             "⚠️ Churn Risk",
             "👥 Customer Segments",
             "🎯 Retention Actions",
             "🔍 Customer Lookup"],
            label_visibility="collapsed",
        )

        st.markdown("""
        <hr style="border-color:rgba(255,255,255,0.06); margin:20px 0 12px 0;">
        <div style="font-size:0.72rem; color:rgba(232,234,246,0.3); text-align:center; line-height:1.6;">
            📅 Data: 2012–2018<br>
            👥 16,737 Members<br>
            🤖 Random Forest · K-Means<br>
            ⏱️ Cutoff: Sep 2017
        </div>
        """, unsafe_allow_html=True)

    return page


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    page = sidebar()

    # ── Check if pipeline has been run ──
    if not os.path.exists(DATA_PATH):
        st.warning("⚙️ First-time setup: the analytics pipeline needs to run once to process the data.")
        if st.button("🚀 Run Analytics Pipeline Now", type="primary"):
            run_pipeline_with_progress()
        st.stop()

    with st.spinner("Loading data …"):
        df       = load_scored_data()
        activity = load_flight_activity()

    if df is None:
        st.error("Could not load data. Please run the pipeline first.")
        st.stop()

    if page == "🏠 Executive Overview":
        page_overview(df)
    elif page == "⚠️ Churn Risk":
        page_churn(df)
    elif page == "👥 Customer Segments":
        page_segments(df)
    elif page == "🎯 Retention Actions":
        page_retention(df)
    elif page == "🔍 Customer Lookup":
        page_lookup(df, activity)


if __name__ == "__main__":
    main()
