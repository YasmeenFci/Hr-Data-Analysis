"""
HR Analytics Intelligence Dashboard
Streamlit app with EDA, visualization, and embedded ML algorithms
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Streamlit 1.57, dark mode verified ─────────────────────────
# Color palette sourced directly from utils.D9m7Ykmm.js:
#   Dark bg  = gray100 #0e1117  |  Sidebar bg = gray90 #262730
#   Body txt = gray10  #fafafa  |  Faded txt  = rgba(250,250,250,0.6)
#   gray85   = #31333F          |  gray80     = #555867
#
# Key lesson: emotion CSS-in-JS renders widget internals (radio spans, selectbox
# divs, heading tags) OUTSIDE the normal cascade, so each must be targeted by its
# own data-testid or data-baseweb attribute — a generic `label` or `span` selector
# on the parent is NOT enough.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ═══════════════════════════════════════════════════════════════════
   1. GLOBAL FONT
   Target every layer: html root, emotion widget containers, baseweb.
   Do NOT use [class*="css"] — those are hashed and unstable.
═══════════════════════════════════════════════════════════════════ */
html, body, .stApp { font-family: 'Sora', sans-serif !important; }

/* Widget label text (the title above every widget) */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span { font-family: 'Sora', sans-serif !important; }

/* Markdown / headings / plain text */
.stMarkdown, .stMarkdown p, .stMarkdown li,
[data-testid="stHeading"], [data-testid="stHeadingWithActionElements"],
[data-testid="stText"] { font-family: 'Sora', sans-serif !important; }

/* Baseweb input controls */
[data-baseweb="input"] input,
[data-baseweb="select"] *,
[data-baseweb="textarea"] textarea,
[data-baseweb="radio"] *,
[data-baseweb="checkbox"] * { font-family: 'Sora', sans-serif !important; }


/* ═══════════════════════════════════════════════════════════════════
   2. APP BACKGROUND & HEADER
═══════════════════════════════════════════════════════════════════ */
.stApp { background-color: #0f1117 !important; }

[data-testid="stHeader"] {
    background-color: #0f1117 !important;
    border-bottom: 1px solid #1e2130 !important;
}
/* toolbar icons inside header */
[data-testid="stToolbar"], [data-testid="stToolbarActions"] {
    background: transparent !important;
}
[data-testid="stToolbarActionButton"] svg { fill: #8b8fa8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   3. SIDEBAR
   Sidebar bg = gray90 (#262730). We use a slightly deeper custom bg.
   All text inside must be explicitly set — emotion does NOT inherit
   from the sidebar container color.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13162a 0%, #10121f 100%) !important;
    border-right: 1px solid #2d3157 !important;
}
[data-testid="stSidebarContent"] { background: transparent !important; }

/* All text nodes in sidebar */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #c9cde8 !important; }

/* Sidebar section caption / dividers */
[data-testid="stSidebar"] .stMarkdown hr { border-color: #2d3157 !important; }

/* ── Sidebar RADIO (used as page nav) ──────────────────────────────
   Radio options render as: <div data-testid="stRadio">
     <div data-testid="stRadioGroup">
       <label>
         <div data-baseweb="radio">  ← the circle
         <div>
           <span>Option text</span>  ← THIS is what we need to color
   So `label` alone works, but `span` inside needs explicit targeting.
────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #c9cde8 !important;
    gap: 10px !important;
}
/* Option text span */
[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] ~ div span,
[data-testid="stSidebar"] [data-testid="stRadio"] label > div:last-child {
    color: #c9cde8 !important;
}
/* Radio circle — unselected */
[data-testid="stSidebar"] [data-baseweb="radio"] div {
    border-color: #555867 !important;
}
/* Radio circle — selected */
[data-testid="stSidebar"] [data-baseweb="radio"][data-checked="true"] div,
[data-testid="stSidebar"] [aria-checked="true"] [data-baseweb="radio"] div {
    border-color: #7c83ff !important;
    background-color: #7c83ff !important;
}
/* Selected option text */
[data-testid="stSidebar"] [aria-checked="true"] ~ div span,
[data-testid="stSidebar"] label:has([aria-checked="true"]) > div:last-child {
    color: #7c83ff !important;
    font-weight: 600 !important;
}

/* ── Sidebar Filters caption ── */
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #8b8fa8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}


/* ═══════════════════════════════════════════════════════════════════
   4. MAIN BLOCK CONTAINER
═══════════════════════════════════════════════════════════════════ */
.block-container, .stMainBlockContainer {
    padding-top: 1.5rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
}

/* Page-level headings (st.title / st.header / st.subheader / st.markdown "# H1") */
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stHeadingWithActionElements"] h1,
[data-testid="stHeadingWithActionElements"] h2,
[data-testid="stHeadingWithActionElements"] h3 {
    color: #e0e4ff !important;
    font-family: 'Sora', sans-serif !important;
}

/* Body markdown text */
.stMarkdown p, .stMarkdown li, .stMarkdown a { color: #c9cde8 !important; }
/* st.caption / small text */
[data-testid="stCaptionContainer"] p { color: #8b8fa8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   5. WIDGET LABELS (the title text ABOVE every widget)
   Emotion renders these as: <div data-testid="stWidgetLabel"><p>Label</p>
   The color is set inline by emotion as fadedText60 — we must override
   the <p> directly, not just the container div.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #8b8fa8 !important;
    font-size: 0.82rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}


/* ═══════════════════════════════════════════════════════════════════
   6. SELECTBOX
   Structure: <div data-testid="stSelectbox">
                <div data-testid="stWidgetLabel">…label…
                <div data-baseweb="select">
                  <div>  ← selected value text lives here
   The value text is NOT in a <label> — it's in a <div> with aria roles.
═══════════════════════════════════════════════════════════════════ */
[data-baseweb="select"] > div {
    background-color: #1a1d2e !important;
    border-color: #2d3157 !important;
    color: #e0e4ff !important;
}
[data-baseweb="select"] [data-baseweb="tag"] { background-color: #7c83ff22 !important; }
/* Dropdown menu */
[data-baseweb="menu"] { background-color: #1a1d2e !important; border-color: #2d3157 !important; }
[data-baseweb="menu"] li { color: #c9cde8 !important; }
[data-baseweb="menu"] li:hover { background-color: #7c83ff22 !important; }
/* Selected item in dropdown */
[data-baseweb="menu"] [aria-selected="true"] {
    background-color: #7c83ff33 !important;
    color: #7c83ff !important;
}
/* Placeholder text */
[data-baseweb="select"] [data-placeholder="true"] { color: #555867 !important; }
/* Arrow icon */
[data-baseweb="select"] svg { fill: #8b8fa8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   7. MULTISELECT
   Selected tags and option list use same baseweb patterns as selectbox.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #7c83ff22 !important;
    border-color: #7c83ff55 !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span { color: #7c83ff !important; }
[data-testid="stMultiSelect"] [data-baseweb="tag"] [role="presentation"] svg { fill: #7c83ff !important; }


/* ═══════════════════════════════════════════════════════════════════
   8. RADIO BUTTONS (outside sidebar — in main content area)
   Structure: <div data-testid="stRadio">
                <div data-testid="stWidgetLabel">  ← label (covered above)
                <div data-testid="stRadioGroup">
                  <label>
                    <div data-baseweb="radio">  ← visual circle
                    <div>
                      <span>  ← THE TEXT — not covered by `label { color }`
                              because emotion sets it inline on the <span>
═══════════════════════════════════════════════════════════════════ */
[data-testid="stRadio"] label { color: #c9cde8 !important; }
/* The actual option text span */
[data-testid="stRadioGroup"] label > div:last-child,
[data-testid="stRadioGroup"] label > div:last-child span { color: #c9cde8 !important; }
/* Unselected circle */
[data-testid="stRadioGroup"] [data-baseweb="radio"] div { border-color: #555867 !important; }
/* Selected circle */
[data-testid="stRadioGroup"] [aria-checked="true"] [data-baseweb="radio"] div {
    border-color: #7c83ff !important;
    background-color: #7c83ff !important;
}
/* Selected text */
[data-testid="stRadioGroup"] label:has([aria-checked="true"]) > div:last-child,
[data-testid="stRadioGroup"] label:has([aria-checked="true"]) > div:last-child span {
    color: #7c83ff !important;
    font-weight: 600 !important;
}


/* ═══════════════════════════════════════════════════════════════════
   9. SLIDER
   [data-testid="stSlider"] — label covered by stWidgetLabel above.
   Thumb value shown in [data-testid="stSliderThumbValue"] — a tooltip.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stSlider"] [role="slider"] { background: #7c83ff !important; }
[data-testid="stSliderThumbValue"] { color: #e0e4ff !important; }
/* Track */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]::before {
    background: #7c83ff !important;
}


/* ═══════════════════════════════════════════════════════════════════
   10. BUTTONS
═══════════════════════════════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, #7c83ff, #5a61e8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Sora', sans-serif !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
.stButton > button:hover  { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
/* Secondary / outline buttons (download, etc.) */
[data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a {
    border-color: #7c83ff !important;
    color: #7c83ff !important;
}


/* ═══════════════════════════════════════════════════════════════════
   11. TABS
   [data-baseweb="tab-list"] — the tab bar
   [data-baseweb="tab"]      — individual tab button
   Tab text lives inside a <span> inside the button.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1a1d2e !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 7px !important;
    border: none !important;
}
/* Tab text — emotion sets color on the span, not the button */
[data-testid="stTabs"] [data-baseweb="tab"] span,
[data-testid="stTabs"] [data-baseweb="tab"] p {
    color: #8b8fa8 !important;
    font-weight: 500 !important;
}
[data-testid="stTabs"] [aria-selected="true"] { background: #7c83ff22 !important; }
[data-testid="stTabs"] [aria-selected="true"] span,
[data-testid="stTabs"] [aria-selected="true"] p {
    color: #7c83ff !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }


/* ═══════════════════════════════════════════════════════════════════
   12. EXPANDER
   Summary text is in a <p> inside <summary>, not directly in summary.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: #1a1d2e !important;
    border: 1px solid #2d3157 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpanderDetails"] p {
    color: #c9cde8 !important;
}
[data-testid="stExpander"] summary svg { fill: #8b8fa8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   13. FILE UPLOADER
═══════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploader"] {
    background: #1a1d2e !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: #1a1d2e !important;
    border: 1px dashed #2d3157 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #8b8fa8 !important; }
[data-testid="stFileUploaderDropzone"] svg { fill: #8b8fa8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   14. ALERTS (info / success / warning / error)
   Streamlit renders these with colored left-border; text is in <p>.
═══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 4px !important;
}
[data-testid="stAlertContainer"] p { color: #c9cde8 !important; }


/* ═══════════════════════════════════════════════════════════════════
   15. METRIC WIDGETS
   [data-testid="stMetricValue"] — big number
   [data-testid="stMetricLabel"] — label above
   [data-testid="stMetricDelta"] — delta below
═══════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #1a1d2e;
    border: 1px solid #2d3157;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] {
    color: #7c83ff !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
    color: #8b8fa8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricDelta"] { color: #4caf8c !important; }


/* ═══════════════════════════════════════════════════════════════════
   16. TEXT INPUT / NUMBER INPUT / TEXT AREA
   Input fields use [data-baseweb="input"] wrapper.
   The actual <input> tag color must be set separately.
═══════════════════════════════════════════════════════════════════ */
[data-baseweb="input"], [data-baseweb="textarea"] {
    background-color: #1a1d2e !important;
    border-color: #2d3157 !important;
}
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    color: #e0e4ff !important;
    background-color: transparent !important;
}
[data-baseweb="input"] input::placeholder,
[data-baseweb="textarea"] textarea::placeholder { color: #555867 !important; }


/* ═══════════════════════════════════════════════════════════════════
   17. SCROLLBAR
═══════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0f1117; }
::-webkit-scrollbar-thumb { background: #2d3157; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #7c83ff; }
</style>
""", unsafe_allow_html=True)

# ─── Pure HTML component classes (no Streamlit internals, always safe) ────────
st.markdown("""<style>
.metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid #2d3157;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #7c83ff; font-family: 'JetBrains Mono', monospace; }
.metric-label { font-size: 0.78rem; color: #8b8fa8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

.section-header {
    background: linear-gradient(90deg, #1a1d2e, #252840);
    border-left: 4px solid #7c83ff;
    padding: 10px 18px;
    border-radius: 0 8px 8px 0;
    margin: 20px 0 12px 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: #e0e4ff;
}

.result-box {
    background: #1a1d2e;
    border: 1px solid #2d3157;
    border-radius: 10px;
    padding: 18px;
    margin: 10px 0;
}
.algo-tag {
    display: inline-block;
    background: #7c83ff22;
    color: #7c83ff;
    border: 1px solid #7c83ff55;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)


# ─── Data Loader ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data\hr_clean_data.csv", parse_dates=["hire_date"])
    # ensure numeric types
    for col in ["salary", "age", "years_at_company", "performance_score",
                "satisfaction_score", "training_hours", "projects_completed",
                "absences_days", "engagement_index"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["attrition"] = df["attrition"].astype(int)
    df["remote_work"] = df["remote_work"].astype(int)
    return df


df = load_data()

FEATURE_COLS = ["age", "salary", "years_at_company", "performance_score",
                "satisfaction_score", "training_hours", "projects_completed",
                "absences_days", "remote_work", "engagement_index"]

PLOTLY_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "#1a1d2e",
    "plot_bgcolor": "#1a1d2e",
    "font": {"family": "Sora, sans-serif", "color": "#c9cde8"},
}


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 HR Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Exploration", "🤖 ML Laboratory", "📁 Data View"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Filters")

    dept_filter = st.multiselect(
        "Department",
        options=sorted(df["department"].dropna().unique()),
        default=sorted(df["department"].dropna().unique()),
    )
    pos_filter = st.multiselect(
        "Position Level",
        options=sorted(df["position"].dropna().unique()),
        default=sorted(df["position"].dropna().unique()),
    )
    salary_range = st.slider(
        "Salary Range ($)",
        int(df["salary"].min()), int(df["salary"].max()),
        (int(df["salary"].min()), int(df["salary"].max())),
        step=5000,
    )

    st.markdown("---")
    st.caption("Dataset: HR Employee Analytics")
    st.caption(f"Records after filters: **{len(df):,}**")

# ─── Apply Filters ───────────────────────────────────────────────────────────
dff = df[
    df["department"].isin(dept_filter)
    & df["position"].isin(pos_filter)
    & df["salary"].between(*salary_range)
].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown("# 📊 HR Analytics Dashboard")
    st.markdown(f"*Showing **{len(dff):,}** employee records*")

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("Total Employees", f"{len(dff):,}", ""),
        ("Attrition Rate", f"{dff['attrition'].mean()*100:.1f}%", ""),
        ("Avg Salary", f"${dff['salary'].mean():,.0f}", ""),
        ("Avg Engagement", f"{dff['engagement_index'].mean():.2f} / 5", ""),
        ("Remote Workers", f"{dff['remote_work'].mean()*100:.0f}%", ""),
    ]
    for col, (label, val, _) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{val}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

    # Row 1
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-header">Attrition by Department</div>', unsafe_allow_html=True)
        att_dept = (
            dff.groupby("department")["attrition"].mean().mul(100).reset_index()
        )
        att_dept.columns = ["Department", "Attrition Rate (%)"]
        fig = px.bar(
            att_dept.sort_values("Attrition Rate (%)", ascending=True),
            x="Attrition Rate (%)", y="Department", orientation="h",
            color="Attrition Rate (%)", color_continuous_scale="RdYlGn_r",
            **{k: v for k, v in [("template", "plotly_dark")]},
        )
        fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(l=0, r=10, t=20, b=20),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Salary Distribution by Position</div>', unsafe_allow_html=True)
        fig = px.violin(
            dff, x="position", y="salary", color="position",
            box=True, template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(l=0, r=10, t=20, b=20),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns([1, 1])

    with col3:
        st.markdown('<div class="section-header">Engagement Index vs Salary</div>', unsafe_allow_html=True)
        fig = px.scatter(
            dff.sample(min(600, len(dff))),
            x="salary", y="engagement_index", color="attrition",
            size="training_hours", hover_data=["department", "position"],
            color_discrete_map={0: "#7c83ff", 1: "#ff6b6b"},
            template="plotly_dark",
        )
        fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(l=0, r=10, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-header">Headcount by Dept & Tenure</div>', unsafe_allow_html=True)
        pivot = dff.groupby(["department", "tenure_group"]).size().reset_index(name="Count")
        fig = px.bar(
            pivot, x="department", y="Count", color="tenure_group",
            barmode="stack", template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Viridis,
        )
        fig.update_layout(**PLOTLY_THEME, height=320, margin=dict(l=0, r=10, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Exploration":
    st.markdown("# 🔍 Deep Exploration")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔗 Correlations", "🎯 Attrition Analysis"])

    with tab1:
        col = st.selectbox("Select feature to explore", FEATURE_COLS)
        grp = st.selectbox("Group by", ["None", "department", "position", "gender", "education"])

        c1, c2 = st.columns(2)
        with c1:
            if grp == "None":
                fig = px.histogram(dff, x=col, nbins=30, template="plotly_dark",
                                   color_discrete_sequence=["#7c83ff"])
            else:
                fig = px.histogram(dff, x=col, color=grp, nbins=30,
                                   template="plotly_dark", barmode="overlay",
                                   opacity=0.75)
            fig.update_layout(**PLOTLY_THEME, height=350, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            if grp == "None":
                fig = px.box(dff, y=col, template="plotly_dark",
                             color_discrete_sequence=["#7c83ff"])
            else:
                fig = px.box(dff, x=grp, y=col, color=grp,
                             template="plotly_dark")
            fig.update_layout(**PLOTLY_THEME, height=350, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
        corr_cols = st.multiselect("Features for correlation", FEATURE_COLS, default=FEATURE_COLS)
        if len(corr_cols) >= 2:
            corr_matrix = dff[corr_cols].corr()
            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale="RdBu",
                zmin=-1, zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                showscale=True,
            ))
            fig.update_layout(**PLOTLY_THEME, height=500, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Attrition Rate by Education</div>', unsafe_allow_html=True)
            att_edu = dff.groupby("education")["attrition"].mean().mul(100).reset_index()
            fig = px.bar(att_edu, x="education", y="attrition",
                         color="attrition", color_continuous_scale="Reds",
                         labels={"attrition": "Attrition %"},
                         template="plotly_dark")
            fig.update_layout(**PLOTLY_THEME, height=300, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">Attrition vs Absences</div>', unsafe_allow_html=True)
            fig = px.violin(dff, x="attrition", y="absences_days",
                            color="attrition", box=True,
                            color_discrete_map={0: "#7c83ff", 1: "#ff6b6b"},
                            template="plotly_dark",
                            labels={"attrition": "Attrition (0=No, 1=Yes)"})
            fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Average Features by Attrition Status</div>', unsafe_allow_html=True)
        avg_comparison = dff.groupby("attrition")[FEATURE_COLS].mean().T.reset_index()
        avg_comparison.columns = ["Feature", "Retained (0)", "Left (1)"]
        avg_comparison["Difference %"] = (
            (avg_comparison["Left (1)"] - avg_comparison["Retained (0)"])
            / avg_comparison["Retained (0)"] * 100
        ).round(1)
        st.dataframe(
            avg_comparison.style.background_gradient(subset=["Difference %"], cmap="RdYlGn"),
            use_container_width=True, height=350,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — ML LABORATORY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Laboratory":
    st.markdown("# 🤖 ML Laboratory")
    st.markdown("*Train machine learning models on your HR data. Upload your own CSV or use the built-in dataset.*")

    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                  classification_report, confusion_matrix,
                                  mean_squared_error, r2_score, silhouette_score)
    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                   AdaBoostClassifier, RandomForestRegressor,
                                   GradientBoostingRegressor)
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.decomposition import PCA

    # ── Upload or use default ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Data Source</div>', unsafe_allow_html=True)
    upload = st.file_uploader("Upload your own CSV (optional — leave blank to use HR dataset)", type="csv")
    if upload:
        user_df = pd.read_csv(upload)
        st.success(f"✅ Uploaded: {user_df.shape[0]:,} rows × {user_df.shape[1]} columns")
    else:
        user_df = dff.copy()
        st.info("Using built-in HR dataset.")

    st.markdown("---")

    # ── Task & Algorithm Selection ─────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Configure Experiment</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        task = st.selectbox("Task Type", ["Classification", "Regression", "Clustering"])
    with c2:
        if task == "Classification":
            algo_choices = {
                "Logistic Regression": LogisticRegression(max_iter=500),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(n_estimators=100),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
                "AdaBoost": AdaBoostClassifier(n_estimators=100, algorithm='SAMME'),
                "SVM (RBF kernel)": SVC(probability=True),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
            }
        elif task == "Regression":
            algo_choices = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
                "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100),
                "SVR": SVR(),
            }
        else:
            algo_choices = {
                "K-Means": "kmeans",
                "Agglomerative Clustering": "agg",
                "DBSCAN": "dbscan",
            }
        algo_name = st.selectbox("Algorithm", list(algo_choices.keys()))

    with c3:
        numeric_cols = user_df.select_dtypes(include=[np.number]).columns.tolist()
        if task == "Classification":
            bool_or_bin = [c for c in numeric_cols if user_df[c].nunique() == 2]
            target_default = "attrition" if "attrition" in bool_or_bin else bool_or_bin[0] if bool_or_bin else numeric_cols[0]
            target = st.selectbox("Target Variable", options=[target_default] + [c for c in user_df.columns if c != target_default])
        elif task == "Regression":
            target_default = "salary" if "salary" in numeric_cols else numeric_cols[0]
            target = st.selectbox("Target Variable", options=[target_default] + [c for c in numeric_cols if c != target_default])
        else:
            target = None
            st.markdown("*Unsupervised — no target needed*")

    # Feature selection
    if target:
        avail_features = [c for c in numeric_cols if c != target]
    else:
        avail_features = numeric_cols

    # Default features always exclude the target to prevent accidental data leakage
    _safe_defaults = [f for f in FEATURE_COLS if f in avail_features and f != target][:8]
    selected_features = st.multiselect(
        "Select Features",
        avail_features,
        default=_safe_defaults,
        help="⚠️ Never include the target variable as a feature — that causes data leakage.",
    )

    # Hyperparameters
    with st.expander("🔧 Hyperparameter Tuning"):
        col_a, col_b, col_c = st.columns(3)
        test_size = col_a.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = col_b.slider("Cross-Val Folds", 2, 10, 5)
        if task == "Clustering":
            n_clusters = col_c.slider("Number of Clusters (K-Means / Agg)", 2, 10, 4)
            eps_val = col_a.slider("DBSCAN eps", 0.1, 5.0, 0.8, 0.1)
            min_s = col_b.slider("DBSCAN min_samples", 2, 20, 5)

    # ── Train ──────────────────────────────────────────────────────────────
    if st.button("🚀 Run Experiment", type="primary", use_container_width=True):
        if len(selected_features) < 2:
            st.error("Please select at least 2 features.")
            st.stop()

        model_data = user_df[selected_features].dropna()
        if target:
            target_data = user_df.loc[model_data.index, target]

        scaler = StandardScaler()
        X = scaler.fit_transform(model_data)

        # ── Classification ────────────────────────────────────────────────
        if task == "Classification":
            y = target_data.astype(int) if target_data.dtype != object else LabelEncoder().fit_transform(target_data)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
            model = algo_choices[algo_name]

            with st.spinner(f"Training {algo_name}…"):
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else None

                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average="weighted")
                auc = roc_auc_score(y_te, y_prob) if y_prob is not None else None
                cv = cross_val_score(model, X, y, cv=StratifiedKFold(cv_folds), scoring="accuracy")

            st.success("✅ Training complete!")
            st.markdown(f'<div class="algo-tag">{algo_name}</div>', unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("F1 Score", f"{f1:.4f}")
            m3.metric("ROC-AUC", f"{auc:.4f}" if auc else "N/A")
            m4.metric(f"CV Mean ({cv_folds}-fold)", f"{cv.mean():.4f} ± {cv.std():.3f}")

            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(y_te, y_pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                labels={"x": "Predicted", "y": "Actual"},
                                title="Confusion Matrix", template="plotly_dark")
                fig.update_layout(**PLOTLY_THEME, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if hasattr(model, "feature_importances_"):
                    fi = pd.DataFrame({"Feature": selected_features,
                                       "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                                 color="Importance", color_continuous_scale="Viridis",
                                 title="Feature Importances", template="plotly_dark")
                    fig.update_layout(**PLOTLY_THEME, height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                elif hasattr(model, "coef_"):
                    coefs = pd.DataFrame({"Feature": selected_features,
                                          "Coefficient": np.abs(model.coef_[0])}).sort_values("Coefficient", ascending=True)
                    fig = px.bar(coefs, x="Coefficient", y="Feature", orientation="h",
                                 color="Coefficient", color_continuous_scale="Plasma",
                                 title="Feature Coefficients (abs)", template="plotly_dark")
                    fig.update_layout(**PLOTLY_THEME, height=350, coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model.")

            with st.expander("📄 Full Classification Report"):
                report = classification_report(y_te, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).T.style.background_gradient(cmap="Blues"))

        # ── Regression ────────────────────────────────────────────────────
        elif task == "Regression":
            # ── Bug fix 1: guard against target leaking into features ──────
            if target in selected_features:
                st.error(
                    f"⚠️ **Data leakage detected**: `{target}` is both your target "
                    f"and a selected feature. Remove it from the feature list before training."
                )
                st.stop()

            y = target_data.values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
            model = algo_choices[algo_name]

            with st.spinner(f"Training {algo_name}…"):
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                r2 = r2_score(y_te, y_pred)
                # Bug fix 2: MAPE and RMSE-as-% give interpretable scale
                mape = np.mean(np.abs((y_te - y_pred) / np.where(y_te == 0, 1, y_te))) * 100
                rmse_pct = rmse / np.mean(y_te) * 100
                # Baseline: always predicting the training mean
                baseline_rmse = np.sqrt(mean_squared_error(y_te, np.full_like(y_te, y_tr.mean())))
                cv = cross_val_score(model, X, y, cv=cv_folds, scoring="r2")

            st.success("✅ Training complete!")
            st.markdown(f'<div class="algo-tag">{algo_name}</div>', unsafe_allow_html=True)

            # Bug fix 3: show RMSE in context so it doesn't look alarming
            st.info(
                f"**How to read these metrics** — RMSE is in the same unit as `{target}`. "
                f"A large absolute RMSE is normal for high-value targets (e.g. salary in dollars). "
                f"Use **RMSE %** and **MAPE** for scale-independent interpretation, "
                f"and compare to the **Baseline RMSE** (always predicting the mean)."
            )

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("RMSE", f"{rmse:,.0f}")
            m2.metric("RMSE % of Mean", f"{rmse_pct:.1f}%",
                      help="RMSE expressed as % of the target mean — scale-independent")
            m3.metric("MAPE", f"{mape:.1f}%",
                      help="Mean Absolute Percentage Error — average % off per prediction")
            m4.metric("R² Score", f"{r2:.4f}",
                      help="1.0 = perfect, 0.0 = predicts the mean, negative = worse than mean")
            m5.metric(f"CV R² ({cv_folds}-fold)", f"{cv.mean():.4f} ± {cv.std():.3f}")

            c1, c2 = st.columns(2)
            with c1:
                scatter_df = pd.DataFrame({"Actual": y_te, "Predicted": y_pred})
                fig = px.scatter(scatter_df, x="Actual", y="Predicted",
                                 trendline="ols", title="Actual vs Predicted",
                                 template="plotly_dark",
                                 color_discrete_sequence=["#7c83ff"])
                # Perfect prediction line
                mn, mx = float(y_te.min()), float(y_te.max())
                fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                              line=dict(color="#ff6b6b", dash="dash", width=1.5))
                fig.update_layout(**PLOTLY_THEME, height=350)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                residuals = y_te - y_pred
                fig = px.histogram(x=residuals, nbins=30,
                                   title="Residuals Distribution (should be ~normal, centered at 0)",
                                   template="plotly_dark",
                                   color_discrete_sequence=["#ff6b6b"])
                fig.add_vline(x=0, line_color="#7c83ff", line_dash="dash")
                fig.update_layout(**PLOTLY_THEME, height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Baseline comparison bar chart
            st.markdown('<div class="section-header">Model vs Baseline Comparison</div>',
                        unsafe_allow_html=True)
            compare_df = pd.DataFrame({
                "Model": ["Baseline\n(predict mean)", algo_name],
                "RMSE": [baseline_rmse, rmse],
                "Color": ["#8b8fa8", "#7c83ff"],
            })
            fig = px.bar(compare_df, x="Model", y="RMSE",
                         color="Color", color_discrete_map="identity",
                         text_auto=".0f",
                         title=f"Your model vs dumb baseline — lower RMSE is better",
                         template="plotly_dark")
            fig.update_layout(**PLOTLY_THEME, height=280, showlegend=False,
                              margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

            if hasattr(model, "feature_importances_"):
                fi = pd.DataFrame({"Feature": selected_features,
                                   "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
                fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Viridis",
                             title="Feature Importances", template="plotly_dark")
                fig.update_layout(**PLOTLY_THEME, height=350, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        # ── Clustering ────────────────────────────────────────────────────
        else:
            with st.spinner(f"Running {algo_name}…"):
                if algo_name == "K-Means":
                    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = model.fit_predict(X)
                    inertia = model.inertia_
                elif algo_name == "Agglomerative Clustering":
                    model = AgglomerativeClustering(n_clusters=n_clusters)
                    labels = model.fit_predict(X)
                    inertia = None
                else:
                    model = DBSCAN(eps=eps_val, min_samples=min_s)
                    labels = model.fit_predict(X)
                    inertia = None

                n_found = len(set(labels)) - (1 if -1 in labels else 0)
                sil = silhouette_score(X, labels) if n_found > 1 and len(set(labels)) > 1 else None

                # PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                cluster_df = pd.DataFrame({
                    "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                    "Cluster": labels.astype(str),
                })

            st.success("✅ Clustering complete!")
            st.markdown(f'<div class="algo-tag">{algo_name}</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Clusters Found", n_found)
            m2.metric("Silhouette Score", f"{sil:.4f}" if sil else "N/A")
            m3.metric("Inertia (KMeans)", f"{inertia:,.0f}" if inertia else "N/A")

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(cluster_df, x="PC1", y="PC2", color="Cluster",
                                 title="PCA Cluster Visualization",
                                 template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
                fig.update_layout(**PLOTLY_THEME, height=380)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                if algo_name == "K-Means":
                    # Elbow curve
                    inertias = []
                    k_range = range(2, 11)
                    for k in k_range:
                        km = KMeans(n_clusters=k, random_state=42, n_init=10)
                        km.fit(X)
                        inertias.append(km.inertia_)
                    fig = px.line(x=list(k_range), y=inertias, markers=True,
                                  title="Elbow Curve (Inertia vs K)",
                                  labels={"x": "K", "y": "Inertia"},
                                  template="plotly_dark",
                                  color_discrete_sequence=["#7c83ff"])
                    fig.update_layout(**PLOTLY_THEME, height=380)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    counts = pd.Series(labels).value_counts().reset_index()
                    counts.columns = ["Cluster", "Count"]
                    fig = px.bar(counts, x="Cluster", y="Count",
                                 color="Cluster", title="Cluster Sizes",
                                 template="plotly_dark",
                                 color_discrete_sequence=px.colors.qualitative.Vivid)
                    fig.update_layout(**PLOTLY_THEME, height=380, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Cluster profile
            profile_df = model_data.copy()
            profile_df["Cluster"] = labels
            profile = profile_df.groupby("Cluster")[selected_features].mean().T
            st.markdown('<div class="section-header">Cluster Feature Profiles</div>', unsafe_allow_html=True)
            st.dataframe(profile.style.background_gradient(cmap="coolwarm", axis=1), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — DATA VIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📁 Data View":
    st.markdown("# 📁 Dataset Explorer")

    st.markdown('<div class="section-header">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(dff.head(200), use_container_width=True, height=400)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(dff.describe(include='number').T.style.background_gradient(subset=["mean"], cmap="Blues"),
                     use_container_width=True)
    with c2:
        st.markdown('<div class="section-header">Missing Values</div>', unsafe_allow_html=True)
        miss = dff.isnull().sum().reset_index()
        miss.columns = ["Column", "Missing"]
        miss["% Missing"] = (miss["Missing"] / len(dff) * 100).round(2)
        st.dataframe(miss.style.background_gradient(subset=["% Missing"], cmap="Reds"),
                     use_container_width=True)

    st.markdown('<div class="section-header">Download Filtered Dataset</div>', unsafe_allow_html=True)
    csv = dff.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV", csv, "hr_filtered_data.csv", "text/csv", use_container_width=True)
