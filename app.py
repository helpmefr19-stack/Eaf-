import base64
import html
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from openai import OpenAI


st.set_page_config(
    page_title="EAF Pre-Heat Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --app-bg: #f4f7fb;
        --panel: #ffffff;
        --panel-soft: #f8fafc;
        --text: #0f172a;
        --muted: #64748b;
        --line: #e2e8f0;
        --blue: #2388ff;
        --cyan: #10b8c7;
        --green: #10b981;
        --amber: #f59e0b;
        --red: #ef4444;
        --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(35, 136, 255, 0.12), transparent 34rem),
            linear-gradient(180deg, #f8fbff 0%, var(--app-bg) 46%, #eef4fb 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(248, 250, 252, 0.94));
        border-right: 1px solid rgba(226, 232, 240, 0.85);
        box-shadow: 16px 0 44px rgba(15, 23, 42, 0.05);
        backdrop-filter: blur(18px);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: var(--text) !important;
    }

    [data-testid="stSidebar"] .stNumberInput input {
        min-height: 42px;
        border: 0 !important;
        border-radius: 9px;
        background: linear-gradient(180deg, #ffffff, #f8fafc);
        color: var(--text) !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput input:focus {
        border-color: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput button {
        min-width: 42px;
        border-radius: 9px !important;
        color: #334155 !important;
        background: linear-gradient(180deg, #ffffff, #f3f7fb) !important;
        border: 1px solid #d9e2ee !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput button:hover {
        background: #eef6ff !important;
        border-color: #bfdbfe !important;
        color: #1d4ed8 !important;
    }

    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"],
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"] > div {
        border: 1px solid #d9e2ee !important;
        border-radius: 10px !important;
        background: linear-gradient(180deg, #ffffff, #f8fafc) !important;
        box-shadow: 0 9px 22px rgba(15, 23, 42, 0.045) !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput div,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stNumberInput button {
        outline: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput div[style*="border"],
    [data-testid="stSidebar"] .stNumberInput div[class*="st-"] {
        border-color: #d9e2ee !important;
    }

    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"]:focus-within,
    [data-testid="stSidebar"] .stNumberInput div[data-baseweb="input"]:focus-within > div {
        border-color: #93c5fd !important;
        box-shadow: 0 0 0 3px rgba(35, 136, 255, 0.13), 0 12px 26px rgba(15, 23, 42, 0.06) !important;
        outline: none !important;
    }

    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] .stNumberInput [data-testid="stNumberInputStepUp"] {
        border-color: #d9e2ee !important;
        background: linear-gradient(180deg, #ffffff, #f3f7fb) !important;
        color: #334155 !important;
        outline: none !important;
        box-shadow: none !important;
    }

    [data-testid="stAppViewContainer"] > .main .block-container {
        max-width: 1280px;
        padding-top: 2.1rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--text);
        letter-spacing: 0;
    }

    hr {
        margin: 1rem 0 1.25rem;
        border-color: var(--line);
    }

    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.2rem 0 1rem;
    }

    .brand-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        color: #ffffff;
        font-weight: 800;
        background: linear-gradient(135deg, #2388ff, #10b8c7);
        box-shadow: 0 12px 26px rgba(35, 136, 255, 0.28);
    }

    .brand-title {
        margin: 0;
        font-size: 1.02rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .brand-subtitle {
        margin: 0.15rem 0 0;
        color: var(--muted);
        font-size: 0.78rem;
    }

    .sidebar-section {
        margin-top: 1rem;
        padding: 0.95rem;
        border: 1px solid rgba(219, 227, 238, 0.95);
        border-radius: 8px;
        background: linear-gradient(180deg, #ffffff, #f7fafc);
        box-shadow: 0 12px 26px rgba(15, 23, 42, 0.055);
    }

    .sidebar-label {
        margin: 0 0 0.2rem;
        color: var(--muted);
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
    }

    .sidebar-value {
        margin: 0;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 850;
    }

    .topbar {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-end;
        margin-bottom: 1.25rem;
    }

    .eyebrow {
        margin: 0 0 0.35rem;
        color: var(--blue);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .page-title {
        margin: 0;
        color: var(--text);
        font-size: clamp(2rem, 4vw, 3.8rem);
        line-height: 1.02;
        font-weight: 900;
    }

    .page-caption {
        max-width: 760px;
        margin: 0.8rem 0 0;
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        white-space: nowrap;
        padding: 0.62rem 0.82rem;
        border: 1px solid #bfdbfe;
        border-radius: 8px;
        background: #eff6ff;
        color: #1d4ed8;
        font-size: 0.82rem;
        font-weight: 800;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--green);
        box-shadow: 0 0 0 4px rgba(16, 185, 129, 0.14);
    }

    .metric-card,
    .chart-card,
    .copilot-card {
        border: 1px solid rgba(226, 232, 240, 0.92);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: var(--shadow);
    }

    .metric-card {
        min-height: 152px;
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .metric-card:after {
        content: "";
        position: absolute;
        inset: auto -2rem -3.2rem auto;
        width: 8rem;
        height: 8rem;
        border-radius: 999px;
        background: var(--accent-soft);
    }

    .metric-label {
        margin: 0;
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .metric-value {
        margin: 0.55rem 0 0.15rem;
        color: var(--text);
        font-size: 2.12rem;
        line-height: 1;
        font-weight: 900;
    }

    .metric-help {
        margin: 0;
        color: var(--muted);
        font-size: 0.86rem;
    }

    .metric-chip {
        display: inline-flex;
        margin-top: 0.9rem;
        padding: 0.35rem 0.55rem;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 850;
    }

    .chart-card {
        padding: 1rem 1rem 0.35rem;
        min-height: 390px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(226, 232, 240, 0.92);
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: var(--shadow);
    }

    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    .panel-title {
        margin: 0;
        color: var(--text);
        font-size: 1.15rem;
        font-weight: 900;
    }

    .panel-subtitle {
        margin: 0.25rem 0 0.75rem;
        color: var(--muted);
        font-size: 0.86rem;
    }

    .copilot-card {
        padding: 1.1rem;
        margin-top: 1rem;
    }

    .copilot-answer {
        padding: 1rem;
        border: 1px solid #bfdbfe;
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        background: #f8fbff;
        color: #0f172a !important;
        line-height: 1.65;
    }

    .copilot-answer,
    .copilot-answer * {
        color: #0f172a !important;
    }

    .stTabs [data-baseweb="tab-list"],
    .stTabs [role="tablist"] {
        gap: 0.45rem;
        padding: 0.42rem !important;
        border: 1px solid #dbe3ee;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.9);
        width: fit-content;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.065);
    }

    .stTabs [data-baseweb="tab"],
    .stTabs [role="tab"] {
        min-height: 44px !important;
        height: 44px !important;
        border-radius: 11px !important;
        padding: 0 1.2rem !important;
        color: #334155 !important;
        font-weight: 800;
        outline: none !important;
        box-shadow: none !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .stTabs [data-baseweb="tab"] *,
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [role="tab"] *,
    .stTabs [role="tab"] p,
    .stTabs [role="tab"] span {
        color: #334155 !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    .stTabs [data-baseweb="tab"]:focus,
    .stTabs [data-baseweb="tab"]:focus-visible,
    .stTabs [data-baseweb="tab"][tabindex="0"],
    .stTabs [role="tab"]:focus,
    .stTabs [role="tab"]:focus-visible,
    .stTabs [role="tab"][tabindex="0"] {
        outline: none !important;
        box-shadow: none !important;
    }

    .stTabs [data-baseweb="tab-border"],
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .stTabs [data-baseweb="tab"]:hover,
    .stTabs [role="tab"]:hover {
        background: #f8fbff !important;
        color: #1d4ed8 !important;
    }

    .stTabs [data-baseweb="tab"]:hover *,
    .stTabs [role="tab"]:hover * {
        color: #1d4ed8 !important;
    }

    .stTabs [aria-selected="true"],
    .stTabs [role="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, #ffffff, #eef6ff) !important;
        color: var(--blue) !important;
        border-color: #bfdbfe !important;
        box-shadow: 0 10px 24px rgba(35, 136, 255, 0.14) !important;
    }

    .stTabs [aria-selected="true"] *,
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span,
    .stTabs [role="tab"][aria-selected="true"] *,
    .stTabs [role="tab"][aria-selected="true"] p,
    .stTabs [role="tab"][aria-selected="true"] span {
        color: var(--blue) !important;
    }

    .stTextInput input,
    .stFileUploader section {
        min-height: 42px;
        border-radius: 8px;
        border: 1px solid #dbe3ee !important;
        background: linear-gradient(180deg, #ffffff, #f8fafc) !important;
        color: #0f172a !important;
        caret-color: #0f172a;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9), 0 8px 18px rgba(15, 23, 42, 0.04);
    }

    .stTextInput input:focus,
    .stFileUploader section:focus-within {
        border-color: #93c5fd !important;
        box-shadow: 0 0 0 3px rgba(35, 136, 255, 0.14), 0 12px 26px rgba(15, 23, 42, 0.06);
    }

    .stTextInput input::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    .stFileUploader,
    .stFileUploader *,
    .stFileUploader label,
    .stFileUploader span,
    .stFileUploader small {
        color: #0f172a !important;
    }

    .stFileUploader button,
    .stFileUploader [data-testid="stBaseButton-secondary"],
    .stFileUploader [data-testid="stBaseButton-secondary"] * {
        opacity: 1 !important;
        visibility: visible !important;
        color: #ffffff !important;
    }

    .stFileUploader button,
    .stFileUploader [data-testid="stBaseButton-secondary"] {
        border: 1px solid #1d7ff0 !important;
        border-radius: 8px !important;
        background: linear-gradient(135deg, #2388ff, #10b8c7) !important;
        box-shadow: 0 12px 26px rgba(35, 136, 255, 0.18) !important;
    }

    .stButton > button {
        border-radius: 8px;
        min-height: 42px;
        font-weight: 850;
        border: 1px solid #1d7ff0;
        background: linear-gradient(135deg, #2388ff, #10b8c7);
        color: #ffffff;
        box-shadow: 0 14px 30px rgba(35, 136, 255, 0.23);
    }

    .stButton > button:hover {
        border-color: #0969d9;
        color: #ffffff;
        transform: translateY(-1px);
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
    }

    @media (max-width: 900px) {
        .topbar {
            display: block;
        }

        .status-pill {
            margin-top: 1rem;
        }

        .metric-value {
            font-size: 1.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "output")

DEFAULTS = {
    "ocr_hm": 58.0,
    "ocr_dri": 53.0,
    "ocr_lime": 11.0,
    "ocr_dolo": 1.8,
    "ocr_ps": 0.0,
    "ocr_tc": 122.0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


@st.cache_resource
def load_production_models():
    try:
        p = joblib.load(os.path.join(MODEL_DIR, "preheat_power_model.pkl"))
        y = joblib.load(os.path.join(MODEL_DIR, "preheat_yield_model.pkl"))
        t = joblib.load(os.path.join(MODEL_DIR, "preheat_ttt_model.pkl"))
        return p, y, t
    except Exception:
        return None, None, None


def render_metric_card(label, value, unit, helper, chip, accent, accent_soft):
    st.markdown(
        f"""
        <div class="metric-card" style="--accent:{accent}; --accent-soft:{accent_soft};">
            <p class="metric-label">{label}</p>
            <p class="metric-value">{value}<span style="font-size:1rem;color:#64748b;"> {unit}</span></p>
            <p class="metric-help">{helper}</p>
            <span class="metric-chip">{chip}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=320):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a", family="Inter, Segoe UI, sans-serif"),
        margin=dict(t=18, b=24, l=12, r=12),
        legend=dict(orientation="h", y=-0.16, x=0),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e2e8f0", tickfont=dict(color="#475569"))
    fig.update_yaxes(gridcolor="#e8eef6", zeroline=False, tickfont=dict(color="#475569"))
    return fig


m_power, m_yield, m_ttt = load_production_models()

# Gemini 3.5 Flash is the current stable, free-tier-capable Flash model
# (subject to Gemini API quota and regional availability).
GEMINI_FLASH_MODEL = os.environ.get("GEMINI_FLASH_MODEL", "gemini-3.5-flash")
api_key = os.environ.get("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
client = (
    OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=api_key,
    )
    if api_key
    else None
)


with st.sidebar:
    st.markdown(
        """
        <div class="brand-wrap">
            <div class="brand-mark">EA</div>
            <div>
                <p class="brand-title">Pre-Heat AI</p>
                <p class="brand-subtitle">Recipe optimization suite</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<p class="sidebar-label">Charge Setup</p>', unsafe_allow_html=True)
    hm = st.number_input("Liquid Hot Metal (Tons)", min_value=0.0, max_value=150.0, key="ocr_hm", step=1.0)
    dri = st.number_input("Direct Reduced Iron (Tons)", min_value=0.0, max_value=150.0, key="ocr_dri", step=1.0)
    lime = st.number_input("Lime Flux Additive (Tons)", min_value=0.0, max_value=25.0, key="ocr_lime", step=0.5)
    dolo = st.number_input("Dolomite Flux Additive (Tons)", min_value=0.0, max_value=15.0, key="ocr_dolo", step=0.1)
    ps = st.number_input("Panther Shot / Clean Scrap (Tons)", min_value=0.0, max_value=25.0, key="ocr_ps", step=1.0)
    tc = st.number_input("Total Charge Target (Tons)", min_value=1.0, max_value=250.0, key="ocr_tc", step=1.0)

    hm_frac = hm / tc if tc > 0 else 0
    dri_frac = dri / tc if tc > 0 else 0
    basicity = lime / dolo if dolo > 0 else lime / 0.1
    total_inputs = hm + dri + lime + dolo + ps
    charge_gap = tc - total_inputs

    st.markdown(
        f"""
        <div class="sidebar-section">
            <p class="sidebar-label">Recipe Health</p>
            <p class="sidebar-value">{total_inputs:,.1f} / {tc:,.1f} T</p>
            <p style="margin:0.25rem 0 0;color:#64748b;font-size:0.82rem;">Charge gap: {charge_gap:+.1f} T</p>
        </div>
        <div class="sidebar-section">
            <p class="sidebar-label">Mix Split</p>
            <p style="margin:0;color:#0f172a;font-weight:800;">HM {hm_frac:.1%} Â· DRI {dri_frac:.1%}</p>
            <p style="margin:0.25rem 0 0;color:#64748b;font-size:0.82rem;">Basicity index: {basicity:.2f}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

live_features = np.array([[hm, dri, lime, dolo, ps, tc, hm_frac, dri_frac, basicity]])


st.markdown(
    """
    <div class="topbar">
        <div>
            <p class="eyebrow">Steelmaking Decision Intelligence</p>
            <h1 class="page-title">EAF Pre-Heat Yield & Power Dashboard</h1>
            <p class="page-caption">
                Tune the charge recipe, compare operating scenarios, and extract production values from shop-floor images in one clean workspace.
            </p>
        </div>
        <div class="status-pill"><span class="status-dot"></span> Model workspace active</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_sim, tab_ocr = st.tabs(["Simulator", "Vision OCR"])


with tab_sim:
    if m_power is None:
        st.error("Model pickles are missing inside the output folder. Run the training scripts before launching the tool.")
    else:
        pred_power = m_power.predict(live_features)[0]
        pred_yield = m_yield.predict(live_features)[0]
        pred_ttt = m_ttt.predict(live_features)[0]

        k1, k2, k3 = st.columns(3)
        with k1:
            render_metric_card(
                "Power Requirement",
                f"{pred_power:,.0f}",
                "kWh",
                "Thermal demand forecast for the selected charge.",
                "Energy forecast",
                "#2388ff",
                "rgba(35, 136, 255, 0.12)",
            )
        with k2:
            render_metric_card(
                "Liquid Steel Yield",
                f"{pred_yield:.2f}",
                "%",
                "Projected mass recovery efficiency.",
                "Yield target",
                "#10b981",
                "rgba(16, 185, 129, 0.13)",
            )
        with k3:
            render_metric_card(
                "Tap-To-Tap Time",
                f"{pred_ttt:.1f}",
                "min",
                "Expected process duration for the current plan.",
                "Cycle estimate",
                "#f59e0b",
                "rgba(245, 158, 11, 0.15)",
            )

        st.write("")

        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(border=True):
                st.markdown(
                    """
                    <p class="panel-title">Charge Basket Distribution</p>
                    <p class="panel-subtitle">Material mix by tonnage for the active recipe.</p>
                    """,
                    unsafe_allow_html=True,
                )
                charge_data = pd.DataFrame(
                    {
                        "Material": ["Hot Metal", "DRI", "Lime", "Dolomite", "Panther Shot"],
                        "Tons": [hm, dri, lime, dolo, ps],
                    }
                )
                fig_pie = px.pie(
                    charge_data,
                    values="Tons",
                    names="Material",
                    hole=0.58,
                    color_discrete_sequence=["#12b8c4", "#2388ff", "#7dd3fc", "#cbd5e1", "#94a3b8"],
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#ffffff", width=3)))
                chart_layout(fig_pie, height=305)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})

        with c2:
            with st.container(border=True):
                st.markdown(
                    """
                    <p class="panel-title">Operating Scenario Comparison</p>
                    <p class="panel-subtitle">Power variance against two recipe alternatives.</p>
                    """,
                    unsafe_allow_html=True,
                )
                sim_df = pd.DataFrame(
                    {
                        "Strategy": ["Current", "High HM", "High DRI"],
                        "Power Target (kWh)": [pred_power, pred_power * 0.86, pred_power * 1.12],
                        "Yield Recovery (%)": [pred_yield, pred_yield + 1.8, pred_yield - 2.5],
                    }
                )
                fig_bar = px.bar(
                    sim_df,
                    x="Strategy",
                    y="Power Target (kWh)",
                    color="Yield Recovery (%)",
                    color_continuous_scale=["#dbeafe", "#2388ff", "#10b8c7"],
                    text_auto=".0f",
                )
                fig_bar.update_traces(marker_line_width=0, textfont=dict(color="#0f172a", size=12), cliponaxis=False)
                chart_layout(fig_bar, height=305)
                st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

        c3, c4 = st.columns([0.9, 1.1])
        with c3:
            with st.container(border=True):
                st.markdown(
                    """
                    <p class="panel-title">Charge Target Fit</p>
                    <p class="panel-subtitle">Actual entered material against the total charge target.</p>
                    """,
                    unsafe_allow_html=True,
                )
                gauge_value = min(max(total_inputs / tc * 100, 0), 140) if tc else 0
                fig_gauge = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=gauge_value,
                        number={"suffix": "%", "font": {"size": 34, "color": "#0f172a"}},
                        gauge={
                            "axis": {"range": [0, 140], "tickcolor": "#94a3b8"},
                            "bar": {"color": "#2388ff"},
                            "bgcolor": "#f8fafc",
                            "borderwidth": 0,
                            "steps": [
                                {"range": [0, 90], "color": "#e0f2fe"},
                                {"range": [90, 110], "color": "#dcfce7"},
                                {"range": [110, 140], "color": "#fee2e2"},
                            ],
                            "threshold": {"line": {"color": "#0f172a", "width": 3}, "thickness": 0.75, "value": 100},
                        },
                    )
                )
                chart_layout(fig_gauge, height=270)
                st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        with c4:
            with st.container(border=True):
                st.markdown(
                    """
                    <p class="panel-title">Metallurgical Copilot</p>
                    <p class="panel-subtitle">Ask for short operational guidance based on the active prediction state.</p>
                    """,
                    unsafe_allow_html=True,
                )
                user_query = st.text_input(
                    "Ask the agent",
                    placeholder="Example: What should I adjust to optimize yield right now?",
                    label_visibility="collapsed",
                )

                if user_query:
                    if client is None:
                        st.error("Gemini API key is not configured. Set GEMINI_API_KEY in .streamlit/secrets.toml or your environment.")
                        st.stop()
                    system_context = (
                        "You are a Senior Chief Metallurgical Engineer. "
                        f"Context: Mix Plan has HM={hm}T, DRI={dri}T, LIME={lime}T, DOLO={dolo}T, PS={ps}T, TC={tc}T. "
                        f"Model Predictions: Power={pred_power:,.0f}kWh, Yield={pred_yield:.2f}%, TTT={pred_ttt:.1f} mins. "
                        "Provide short, executive operational insights directly."
                    )
                    with st.spinner("Analyzing metallurgy configuration..."):
                        try:
                            response = client.chat.completions.create(
                                model=GEMINI_FLASH_MODEL,
                                messages=[
                                    {"role": "system", "content": system_context},
                                    {"role": "user", "content": user_query},
                                ],
                                temperature=0.3,
                            )
                            answer = html.escape(response.choices[0].message.content)
                            st.markdown(f'<div class="copilot-answer">{answer}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Failed to communicate with API: {e}")


with tab_ocr:
    left, right = st.columns([1.05, 0.95])

    with left:
        with st.container(border=True):
            st.markdown(
                """
                <p class="panel-title">Vision Data Extraction</p>
                <p class="panel-subtitle">Upload a production sheet, handwritten log card, or application screenshot.</p>
                """,
                unsafe_allow_html=True,
            )
            uploaded_img = st.file_uploader(
                "Upload operational image",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )
            if uploaded_img is not None:
                st.image(uploaded_img, caption="Uploaded production source", use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown(
                """
                <p class="panel-title">Extracted Recipe Sync</p>
                <p class="panel-subtitle">Detected values are written back to the sidebar controls automatically.</p>
                """,
                unsafe_allow_html=True,
            )

            ocr_btn = st.button("Run Vision OCR Parsing", type="primary", use_container_width=True, disabled=uploaded_img is None)

            if uploaded_img is None:
                st.info("Upload an image to enable OCR parsing.")

            if uploaded_img is not None and ocr_btn:
                if client is None:
                    st.error("Gemini API key is not configured. Set GEMINI_API_KEY in .streamlit/secrets.toml or your environment.")
                    st.stop()
                with st.spinner("Running vision extraction..."):
                    try:
                        img_bytes = uploaded_img.read()
                        base64_image = base64.b64encode(img_bytes).decode("utf-8")

                        ocr_prompt = """
                        Look closely at the document/image provided. Identify and extract the numbers corresponding to these operational EAF targets:
                        1. Hot Metal Mass (HM)
                        2. DRI Mass (DRI)
                        3. Lime Tonnage (LIME)
                        4. Dolomite Tonnage (DOLO)
                        5. Panther Shot / Scrap Tonnage (PS)
                        6. Total Charge Target (TC)

                        Return ONLY a raw clean JSON dictionary block exactly containing these string format keys, nothing else, no markdown wrappers, no introductory chat text:
                        {"HM": 58.0, "DRI": 53.0, "LIME": 11.0, "DOLO": 1.8, "PS": 0.0, "TC": 122.0}
                        """

                        response = client.chat.completions.create(
                            model=GEMINI_FLASH_MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": ocr_prompt},
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                    ],
                                }
                            ],
                        )

                        raw_res = response.choices[0].message.content.strip()
                        match = re.search(r"\{[\s\S]*\}", raw_res)
                        if match:
                            parsed_data = json.loads(match.group(0))
                            parsed_data = {k.upper().strip(): float(v) for k, v in parsed_data.items()}
                        else:
                            raise ValueError(f"Could not identify a valid JSON block inside the model response: {raw_res}")

                        st.session_state["ocr_hm"] = float(parsed_data.get("HM", DEFAULTS["ocr_hm"]))
                        st.session_state["ocr_dri"] = float(parsed_data.get("DRI", DEFAULTS["ocr_dri"]))
                        st.session_state["ocr_lime"] = float(parsed_data.get("LIME", DEFAULTS["ocr_lime"]))
                        st.session_state["ocr_dolo"] = float(parsed_data.get("DOLO", DEFAULTS["ocr_dolo"]))
                        st.session_state["ocr_ps"] = float(parsed_data.get("PS", DEFAULTS["ocr_ps"]))
                        st.session_state["ocr_tc"] = float(parsed_data.get("TC", DEFAULTS["ocr_tc"]))

                        st.success("Extraction complete. Sidebar recipe values were updated.")
                        st.json(parsed_data)
                        st.rerun()

                    except Exception as err:
                        st.error(f"Vision engine failed to isolate variables cleanly: {err}")



