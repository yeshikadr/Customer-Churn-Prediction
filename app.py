import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---------------------------------------------------------------- PATHS
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "cleaned_churn_data.csv"
MODEL_PATH = BASE_DIR / "outputs" / "churn_model.pkl"

# Dark base theme so native widgets (tables, inputs) match the dashboard. Restart once after the first run.
_CFG = BASE_DIR / ".streamlit" / "config.toml"
if not _CFG.exists():
    try:
        _CFG.parent.mkdir(exist_ok=True)
        _CFG.write_text('[theme]\nbase = "dark"\nprimaryColor = "#6366F1"\nbackgroundColor = "#0B0F1A"\n'
                        'secondaryBackgroundColor = "#121827"\ntextColor = "#F1F4FB"\n', encoding="utf-8")
    except OSError:
        pass

st.set_page_config(
    page_title="Customer Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()

# ---------------------------------------------------------------- PALETTE
NAVY, PANEL = "#0B0F1A", "#121827"          # page background, card surface
INK = "#F1F4FB"
BLUE, CYAN = "#818CF8", "#38BDF8"           # primary indigo, secondary sky
CORAL, GREEN, AMBER = "#FB7185", "#34D399", "#FBBF24"
TEXT, MUTED = "#F1F4FB", "#8B93A7"
TEAL = BLUE
STATUS_COLORS = {"Retained": BLUE, "Churned": CORAL}

# ---------------------------------------------------------------- CSS
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1000px 520px at 88% -10%, rgba(99,102,241,.16), transparent 60%),
                    radial-gradient(800px 460px at -8% 12%, rgba(56,189,248,.06), transparent 55%), #0B0F1A;
         color: #F1F4FB; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 3rem 3rem; max-width: 1500px; }
h1, h2, h3, h4, h5, p, li, label, .stMarkdown { color: #F1F4FB; }
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700; }
[data-testid="stCaptionContainer"], .stCaption { color: #8B93A7 !important; }
hr { border-color: rgba(255,255,255,.09); }

/* Hero */
.hero { padding: 30px 34px; border-radius: 20px; margin-bottom: 22px;
        background: linear-gradient(115deg, #1E1B4B 0%, #3730A3 55%, #4F46E5 100%);
        box-shadow: 0 18px 40px rgba(55,48,163,.25); }
.hero h1 { margin: 0 0 8px; font-size: 32px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
.hero p { margin: 0; color: #C7D2FE; font-size: 15px; max-width: 760px; }
.chip { display: inline-block; padding: 4px 12px; margin: 16px 8px 0 0; border-radius: 999px; font-size: 12px;
        font-weight: 600; color: #E0E7FF; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); }

/* Cards */
.kpi, .glass { background: #121827; border: 1px solid rgba(255,255,255,.09); border-radius: 16px;
               box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.35);
               transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease; }
.kpi { padding: 20px 22px; margin-bottom: 16px; }
.kpi:hover { transform: translateY(-2px); box-shadow: 0 14px 30px rgba(0,0,0,.35); border-color: var(--accent); }
.kpi .icon { width: 40px; height: 40px; border-radius: 11px; display: flex; align-items: center;
             justify-content: center; margin-bottom: 14px;
             background: color-mix(in srgb, var(--accent) 16%, transparent); }
.kpi .label { color: #8B93A7; font-size: 13px; font-weight: 600; }
.kpi .value { color: #F1F4FB; font-size: 30px; font-weight: 700; letter-spacing: -0.02em;
              line-height: 1.2; font-variant-numeric: tabular-nums; }
.kpi .sub { color: #94A3B8; font-size: 12px; margin-top: 2px; }
.glass { padding: 20px 24px; margin-bottom: 16px; }
.glass:hover { border-color: rgba(129,140,248,.45); }
.glass p, .glass li { color: #C3CAD9; font-size: 14.5px; line-height: 1.65; }
.glass b { color: #F1F4FB; }
.pill { display: inline-block; padding: 6px 16px; border-radius: 999px; font-weight: 700; font-size: 14px; }
.section-title { font-size: 20px; font-weight: 700; letter-spacing: -0.01em; margin: 14px 0 12px; color: #F1F4FB; }
.crumb { color: #94A3B8; font-size: 13px; margin-bottom: 10px; }
.crumb b { color: #C3CAD9; font-weight: 600; }
.note { margin: -2px 0 16px; padding: 10px 14px; border-radius: 10px; font-size: 13px; color: #C3CAD9;
        background: rgba(99,102,241,.12); border-left: 3px solid #818CF8; }
.note b { color: #F1F4FB; }
.result-banner { padding: 18px 22px; border-radius: 14px; margin-bottom: 14px; font-weight: 700; font-size: 17px; }
.rep-table { width: 100%; border-collapse: collapse; font-size: 13.5px; background: #121827;
             border: 1px solid rgba(255,255,255,.09); border-radius: 14px; overflow: hidden; }
.rep-table th { text-align: right; color: #8B93A7; font-weight: 600; padding: 10px 14px;
                background: rgba(255,255,255,.04); border-bottom: 1px solid rgba(255,255,255,.09); }
.rep-table td { text-align: right; color: #F1F4FB; padding: 10px 14px; border-bottom: 1px solid rgba(255,255,255,.06); }
.rep-table th:first-child, .rep-table td:first-child { text-align: left; font-weight: 600; }
.rep-table tr:hover td { background: rgba(129,140,248,.10); }

/* Widgets */
[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
    background-color: #121827 !important; color: #F1F4FB !important;
    border: 1px solid rgba(255,255,255,.16) !important; border-radius: 10px !important; }
[data-baseweb="select"] span, [data-baseweb="select"] div { color: #F1F4FB; }
[data-baseweb="popover"] * { color: #F1F4FB; }
[data-baseweb="popover"] > div, [data-baseweb="menu"] { background-color: #121827 !important; }
[data-baseweb="tag"] { background-color: #4F46E5 !important; }
[data-baseweb="tag"] span { color: #FFFFFF !important; }
div[data-baseweb="slider"] div[role="slider"] { background-color: #4F46E5 !important; box-shadow: none !important; }
[data-testid="stForm"] { background: #121827; border: 1px solid rgba(255,255,255,.09); border-radius: 18px; padding: 24px;
                         box-shadow: 0 8px 24px rgba(0,0,0,.35); }
[data-testid="stMetric"] { background: #121827; border: 1px solid rgba(255,255,255,.09); padding: 18px 20px; border-radius: 14px; }
[data-testid="stMetricLabel"] p { color: #8B93A7 !important; }
[data-testid="stMetricValue"] { color: #F1F4FB !important; font-weight: 700; }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(255,255,255,.09); }
[data-testid="stSpinner"] * { color: #8B93A7 !important; }
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
    background: #4F46E5; color: #FFFFFF !important; border: none; border-radius: 10px; padding: 10px 20px;
    font-weight: 600; transition: background .2s ease, transform .2s ease, box-shadow .2s ease;
    box-shadow: 0 6px 16px rgba(79,70,229,.28); }
.stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover {
    background: #4338CA; transform: translateY(-1px); box-shadow: 0 10px 22px rgba(79,70,229,.36);
    color: #FFFFFF !important; }

/* Charts */
[data-testid="stPlotlyChart"] { border-radius: 16px; overflow: hidden; border: 1px solid rgba(255,255,255,.09); background: #121827;
    box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.35);
    transition: box-shadow .25s ease, border-color .25s ease; }
[data-testid="stPlotlyChart"]:hover { border-color: rgba(129,140,248,.45); box-shadow: 0 14px 30px rgba(0,0,0,.35); }

/* Hide default Streamlit chrome (sidebar toggle stays) */
#MainMenu, footer, [data-testid="stDeployButton"], [data-testid="stDecoration"] { display: none !important; }

/* Smooth page transition: each page is a keyed container, so it fades in once per page change */
@keyframes pageIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
[class*="st-key-page-"] { animation: pageIn .6s cubic-bezier(.22,.61,.36,1) both; }
[data-stale="true"], .stale-element { opacity: 1 !important; transition: none !important; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* Sidebar */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0F1526 0%, #090D18 100%);
                                   border-right: 1px solid rgba(255,255,255,.06); }
section[data-testid="stSidebar"][aria-expanded="true"] { width: 264px !important; min-width: 264px !important; max-width: 264px !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 1.4rem 1rem 1rem; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .4rem; }
.brand { display: flex; align-items: center; gap: 12px; padding: 4px 6px 20px;
         border-bottom: 1px solid rgba(255,255,255,.08); margin-bottom: 14px; }
.brand-logo { width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
              flex-shrink: 0; background: linear-gradient(135deg, #6366F1, #38BDF8); box-shadow: 0 6px 18px rgba(99,102,241,.40); }
.brand-title { color: #FFFFFF; font-size: 14.5px; font-weight: 800; letter-spacing: .08em; line-height: 1.2; }
.brand-sub { color: #94A3B8; font-size: 11.5px; margin-top: 3px; }
section[data-testid="stSidebar"] .stButton { width: 100%; }
section[data-testid="stSidebar"] .stButton > button {
    width: 100%; min-height: 0; padding: 11px 14px; border-radius: 10px; gap: 10px;
    display: flex; justify-content: flex-start; align-items: center; text-align: left;
    background: transparent !important; border: 1px solid transparent !important;
    box-shadow: none !important; transform: none !important;
    color: #94A3B8 !important; font-weight: 500; transition: background .2s ease, color .2s ease; }
section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button div {
    margin: 0; text-align: left !important; justify-content: flex-start; color: inherit !important;
    font-size: 14.5px; white-space: nowrap; }
section[data-testid="stSidebar"] .stButton > button div { flex: 1 1 auto; }
section[data-testid="stSidebar"] .stButton > button span[data-testid="stIconMaterial"] {
    font-size: 20px; color: inherit !important; flex: 0 0 auto; }
section[data-testid="stSidebar"] .stButton > button:hover,
section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
    background: rgba(255,255,255,.06) !important; color: #E2E8F0 !important;
    border-color: transparent !important; transform: none !important; box-shadow: none !important; }
section[data-testid="stSidebar"] .stButton > button:focus-visible { outline: 2px solid #818CF8 !important; outline-offset: 2px; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"],
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover,
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:focus:not(:active),
section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:focus:not(:active) {
    background: rgba(99,102,241,.18) !important; color: #C7D2FE !important;
    font-weight: 650; box-shadow: inset 3px 0 0 #818CF8 !important; }
.glance { display: flex; gap: 10px; margin: 16px 0 12px; }
.glance div { flex: 1; padding: 10px 12px; border-radius: 10px; background: rgba(255,255,255,.04);
              border: 1px solid rgba(255,255,255,.08); }
.glance b { display: block; color: #FFFFFF; font-size: 17px; font-weight: 700; }
.glance span { color: #8B93A7; font-size: 11px; }
.side-footer { padding: 14px 14px 12px; border-radius: 12px; background: rgba(255,255,255,.03);
               border: 1px solid rgba(255,255,255,.08); }
.side-row { display: flex; justify-content: space-between; gap: 10px; padding: 5px 0; font-size: 12px; }
.side-row span:first-child { color: #8B93A7; }
.side-row span:last-child { color: #CBD5E1; font-weight: 600; text-align: right; }
.side-status { display: flex; align-items: center; gap: 8px; margin-top: 10px; padding-top: 10px;
               border-top: 1px solid rgba(255,255,255,.08); font-size: 12px; font-weight: 600; color: #34D399; }
.side-dot { width: 8px; height: 8px; border-radius: 50%; background: #34D399; }
.side-spacer { height: calc(100vh - 640px); min-height: 24px; }
.footer { text-align: center; color: #94A3B8; font-size: 12.5px; margin-top: 40px; padding-top: 18px;
          border-top: 1px solid rgba(255,255,255,.09); }

@media (max-width: 900px) {
    .block-container { padding: 1.2rem 1rem 2rem !important; }
    .hero { padding: 20px; } .hero h1 { font-size: 25px; }
    .kpi .value { font-size: 24px; }
    .side-spacer { height: 24px; }
}
"""
st.markdown("<style>" + APP_CSS + "</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------- HELPERS
def style_chart(fig, height=380):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#121827",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial", color=TEXT, size=13),
        title_font=dict(size=17, color=INK),
        margin=dict(l=30, r=25, t=60, b=35),
        height=height,
        legend=dict(title_font_color=MUTED, font_color=TEXT, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#1B2236", font_color=INK),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.07)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.07)", zerolinecolor="rgba(255,255,255,0.07)")
    return fig


def show(fig, height=380):
    st.plotly_chart(style_chart(fig, height), use_container_width=True,
                    config={"displayModeBar": False})


def _svg(body):
    return ("<svg width='22' height='22' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' "
            "stroke-linecap='round' stroke-linejoin='round'>" + body + "</svg>")


KPI_ICONS = {
    "\U0001F465": _svg("<circle cx='9' cy='8' r='3.5'/><path d='M2 20c0-3.6 3.1-6 7-6s7 2.4 7 6'/><path d='M16 4.5a3.5 3.5 0 010 7M18 14.4c2.3.6 4 2.3 4 5.6'/>"),
    "\u26a0": _svg("<path d='M12 3l10 18H2z'/><path d='M12 10v5M12 18v.5'/>"),
    "\u2705": _svg("<circle cx='12' cy='12' r='9'/><path d='M8 12.5l3 3 5-6'/>"),
    "\U0001F4C9": _svg("<path d='M3 7l6 6 4-4 8 8'/><path d='M15 17h6v-6'/>"),
    "\U0001F6E1": _svg("<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/>"),
    "\U0001F4B3": _svg("<rect x='2' y='5' width='20' height='14' rx='2.5'/><path d='M2 10h20'/>"),
    "\U0001F3AF": _svg("<circle cx='12' cy='12' r='9'/><circle cx='12' cy='12' r='5'/><circle cx='12' cy='12' r='1.2'/>"),
    "\U0001F52C": _svg("<circle cx='11' cy='11' r='7'/><path d='M21 21l-5-5'/>"),
    "\U0001F4E1": _svg("<path d='M3 12h4l3-8 4 16 3-8h4'/>"),
    "\u2696": _svg("<path d='M12 3v18M5 7h14M5 7l-3 8a4 4 0 006 0zM19 7l-3 8a4 4 0 006 0z'/>"),
}


def kpi(col, icon, label, value, accent, sub=""):
    icon = KPI_ICONS.get(icon.replace("\ufe0f", ""), icon)
    col.markdown(
        f'<div class="kpi" style="--accent:{accent}"><div class="icon" style="color:{accent}">{icon}</div>'
        f'<div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def hero(title, text, chips=None):
    chip_html = "".join(f'<span class="chip">{c}</span>' for c in (chips or []))
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{text}</p>{chip_html}</div>', unsafe_allow_html=True)


def note(html):
    st.markdown(f'<div class="note">{html}</div>', unsafe_allow_html=True)


def top_rate_note(d, col, label):
    r = rate_by(d, col)
    if len(r) < 2:
        return ""
    return (f"<b>{r.iloc[0][col]}</b> has the highest churn among {label} groups "
            f"(<b>{r.iloc[0]['Churn Rate (%)']:.1f}%</b>), versus <b>{r.iloc[-1]['Churn Rate (%)']:.1f}%</b> "
            f"for <b>{r.iloc[-1][col]}</b>.")


def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def card(html):
    st.markdown(f'<div class="glass">{html}</div>', unsafe_allow_html=True)


def with_status(d):
    d = d.copy()
    d["Status"] = d["Churn"].map({0: "Retained", 1: "Churned"})
    return d


def count_by(d, col):
    out = d.groupby([col, "Churn"]).size().reset_index(name="Customers")
    out["Status"] = out["Churn"].map({0: "Retained", 1: "Churned"})
    return out


def rate_by(d, col):
    out = (d.groupby(col, observed=True)["Churn"].mean() * 100).reset_index(name="Churn Rate (%)")
    return out.sort_values("Churn Rate (%)", ascending=False)


def risk_category(p):
    if p < 0.30:
        return "Low", GREEN
    if p < 0.60:
        return "Medium", AMBER
    return "High", CORAL


def make_insights(d):
    """Automatically generated findings from the (filtered) dataset."""
    if d.empty:
        return []
    out = []
    rate = d["Churn"].mean() * 100
    out.append(f"The overall churn rate in this view is <b>{rate:.1f}%</b> across <b>{len(d):,}</b> customers.")
    for col, label in [("Contract", "contract type"), ("InternetService", "internet service"),
                       ("PaymentMethod", "payment method")]:
        r = rate_by(d, col)
        if len(r) > 1:
            top, low = r.iloc[0], r.iloc[-1]
            out.append(
                f"By {label}, <b>{top[col]}</b> customers churn the most (<b>{top['Churn Rate (%)']:.1f}%</b>), "
                f"while <b>{low[col]}</b> customers churn the least (<b>{low['Churn Rate (%)']:.1f}%</b>)."
            )
    if d["Churn"].nunique() > 1:
        t = d.groupby("Churn")["tenure"].mean()
        m = d.groupby("Churn")["MonthlyCharges"].mean()
        out.append(f"Churned customers stay <b>{t[1]:.1f}</b> months on average versus <b>{t[0]:.1f}</b> months "
                   f"for retained customers.")
        out.append(f"Churned customers pay <b>${m[1]:.2f}</b> per month on average versus <b>${m[0]:.2f}</b> "
                   f"for retained customers.")
    return out


RECOMMENDATIONS = [
    "Offer incentives to move month-to-month customers onto one-year or two-year contracts.",
    "Run onboarding and check-in campaigns for customers in their first 12 months.",
    "Promote automatic payment methods to customers who pay by electronic check.",
    "Review pricing and service quality for high-spending customers on fiber optic plans.",
    "Bundle security and tech support add-ons to improve engagement and loyalty.",
]


def build_report(d):
    lines = [
        "CUSTOMER CHURN ANALYSIS REPORT",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        "=" * 50,
        f"Customers analysed : {len(d):,}",
        f"Churned            : {int(d['Churn'].sum()):,}",
        f"Retained           : {int((d['Churn'] == 0).sum()):,}",
        f"Churn rate         : {d['Churn'].mean() * 100:.2f}%",
        f"Avg monthly charges: ${d['MonthlyCharges'].mean():.2f}",
        "", "KEY INSIGHTS", "-" * 50,
    ]
    import re
    lines += [f"- {re.sub('<[^>]+>', '', i)}" for i in make_insights(d)]
    lines += ["", "RECOMMENDATIONS", "-" * 50] + [f"- {r}" for r in RECOMMENDATIONS]
    return "\n".join(lines)


# ---------------------------------------------------------------- SIDEBAR
NAV_ITEMS = [
    ("Dashboard Overview", "▦", ":material/dashboard:"),
    ("Churn Analysis", "◫", ":material/monitoring:"),
    ("Model Performance", "◈", ":material/psychology:"),
    ("Predict Churn", "✧", ":material/online_prediction:"),
]

if "page" not in st.session_state:
    st.session_state["page"] = NAV_ITEMS[0][0]


def set_page(name):
    st.session_state["page"] = name


SIDEBAR_CSS = ""

st.markdown("<style>" + SIDEBAR_CSS + "</style>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
<div class="brand">
  <div class="brand-logo">
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.4"
         stroke-linecap="round"><path d="M5 20V11M12 20V4M19 20v-6"/></svg>
  </div>
  <div>
    <div class="brand-title">CHURN ANALYTICS</div>
    <div class="brand-sub">Customer Intelligence Platform</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

for _name, _glyph, _mat in NAV_ITEMS:
    _kw = dict(
        key=f"nav_{_name}",
        type="primary" if st.session_state["page"] == _name else "secondary",
        use_container_width=True,
        on_click=set_page,
        args=(_name,),
    )
    try:
        st.sidebar.button(_name, icon=_mat, **_kw)
    except Exception:  # older Streamlit without the icon parameter
        st.sidebar.button(f"{_glyph}\u2003{_name}", **_kw)

page = st.session_state["page"]

st.sidebar.markdown(
    """
<div class="glance">
  <div><b>__CUST__</b><span>Customers</span></div>
  <div><b>__RATE__</b><span>Churn rate</span></div>
</div>
<div class="side-spacer"></div>
<div class="side-footer">
  <div class="side-row"><span>Project</span><span>Customer Churn Prediction</span></div>
  <div class="side-row"><span>Model</span><span>Logistic Regression</span></div>
  <div class="side-row"><span>Dataset</span><span>IBM Telco Customer Churn</span></div>
  <div class="side-status"><span class="side-dot"></span>Model Ready</div>
</div>
""".replace("__CUST__", f"{len(df):,}").replace("__RATE__", f"{df['Churn'].mean() * 100:.1f}%"),
    unsafe_allow_html=True,
)

st.markdown(f'<div class="crumb">Churn Analytics &nbsp;/&nbsp; <b>{page}</b></div>', unsafe_allow_html=True)


try:
    page_ctx = st.container(key="page-" + page.lower().replace(" ", "-"))
except TypeError:  # older Streamlit without container keys
    page_ctx = st.container()

with page_ctx:
    # ================================================================ OVERVIEW
    if page.endswith("Dashboard Overview"):
        hero("Customer Intelligence Dashboard",
             "Monitor customer health, uncover churn patterns and predict who is likely to leave, "
             "all powered by a Logistic Regression model trained on your customer data.",
             chips=["Logistic Regression", f"{len(df):,} customers analysed", "Model ready"])

        total = len(df)
        churned = int(df["Churn"].sum())
        retained = total - churned
        churn_rate = churned / total * 100
        retention_rate = retained / total * 100
        avg_charges = df["MonthlyCharges"].mean()

        c1, c2, c3 = st.columns(3)
        kpi(c1, "👥", "Total Customers", f"{total:,}", BLUE, "All customers in the dataset")
        kpi(c2, "⚠️", "Churned Customers", f"{churned:,}", CORAL, "Customers who left")
        kpi(c3, "✅", "Retained Customers", f"{retained:,}", GREEN, "Customers who stayed")
        c4, c5, c6 = st.columns(3)
        kpi(c4, "📉", "Overall Churn Rate", f"{churn_rate:.2f}%", AMBER, "Share of customers lost")
        kpi(c5, "🛡️", "Customer Retention Rate", f"{retention_rate:.2f}%", CYAN, "Share of customers kept")
        kpi(c6, "💳", "Average Monthly Charges", f"${avg_charges:.2f}", "#A78BFA", "Per customer, per month")

        section("Customer Insights")
        col1, col2 = st.columns(2)

        pie = px.pie(
            pd.DataFrame({"Status": ["Retained", "Churned"], "Customers": [retained, churned]}),
            names="Status", values="Customers", hole=0.62, title="Customer Churn Distribution",
            color="Status", color_discrete_map=STATUS_COLORS,
        )
        pie.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#121827", width=3)))
        with col1:
            show(pie)

        bar = px.bar(count_by(df, "Contract"), x="Contract", y="Customers", color="Status", barmode="group",
                     title="Churn by Contract Type", color_discrete_map=STATUS_COLORS)
        with col2:
            show(bar)

        col1, col2 = st.columns(2)
        internet = px.bar(count_by(df, "InternetService"), x="InternetService", y="Customers", color="Status",
                          barmode="group", title="Churn by Internet Service", color_discrete_map=STATUS_COLORS)
        with col1:
            show(internet)

        tenure = px.histogram(with_status(df), x="tenure", color="Status", nbins=30, barmode="overlay",
                              opacity=0.8, title="Customer Tenure Distribution",
                              labels={"tenure": "Tenure (months)"}, color_discrete_map=STATUS_COLORS)
        with col2:
            show(tenure)

        charges = px.histogram(with_status(df), x="MonthlyCharges", color="Status", nbins=35, barmode="overlay",
                               opacity=0.75, title="Monthly Charges vs Churn",
                               labels={"MonthlyCharges": "Monthly charges ($)"}, color_discrete_map=STATUS_COLORS)
        show(charges)

        col1, col2 = st.columns(2)
        pay = rate_by(df, "PaymentMethod")
        pay_fig = px.bar(pay, x="Churn Rate (%)", y="PaymentMethod", orientation="h", text_auto=".1f",
                         title="Payment Method Analysis: Churn Rate", color="Churn Rate (%)",
                         color_continuous_scale=["#2A2F5C", BLUE, "#C7D2FE"])
        pay_fig.update_layout(coloraxis_showscale=False, yaxis_title=None)
        with col1:
            show(pay_fig)

        ret = df.copy()
        ret["Tenure Group"] = pd.cut(ret["tenure"], [-1, 12, 24, 48, 72],
                                     labels=["0-12 mo", "13-24 mo", "25-48 mo", "49-72 mo"])
        ret_rate = (1 - ret.groupby("Tenure Group", observed=True)["Churn"].mean()) * 100
        ret_fig = px.line(ret_rate.reset_index(name="Retention Rate (%)"), x="Tenure Group",
                          y="Retention Rate (%)", markers=True, title="Customer Retention Analysis by Tenure")
        ret_fig.update_traces(line=dict(color=CYAN, width=4), marker=dict(size=11, color=BLUE))
        ret_fig.update_yaxes(range=[0, 100])
        with col2:
            show(ret_fig)


    # ================================================================ ANALYSIS
    elif page == "Churn Analysis":
        hero("Churn Analysis",
             "Slice the customer base with filters to see who churns, when they churn and why it matters.")

        section("Filters")
        f1, f2, f3, f4 = st.columns(4)
        contracts = f1.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
        internets = f2.multiselect("Internet Service", sorted(df["InternetService"].unique()),
                                   default=sorted(df["InternetService"].unique()))
        payments = f3.multiselect("Payment Method", sorted(df["PaymentMethod"].unique()),
                                  default=sorted(df["PaymentMethod"].unique()))
        t_min, t_max = int(df["tenure"].min()), int(df["tenure"].max())
        tenure_range = f4.slider("Tenure (months)", t_min, t_max, (t_min, t_max))

        fdf = df[
            df["Contract"].isin(contracts)
            & df["InternetService"].isin(internets)
            & df["PaymentMethod"].isin(payments)
            & df["tenure"].between(*tenure_range)
        ]

        if fdf.empty:
            card("<p><b>No customers match these filters.</b><br>Widen the tenure range or select more "
                 "options above to see results.</p>")
            st.stop()

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "👥", "Customers in View", f"{len(fdf):,}", BLUE)
        kpi(k2, "⚠️", "Churned", f"{int(fdf['Churn'].sum()):,}", CORAL)
        kpi(k3, "📉", "Churn Rate", f"{fdf['Churn'].mean() * 100:.2f}%", AMBER)
        kpi(k4, "💳", "Avg Monthly Charges", f"${fdf['MonthlyCharges'].mean():.2f}", "#A78BFA")

        section("Business Insights")
        items = "".join(f"<li>{i}</li>" for i in make_insights(fdf))
        card(f"<ul style='margin:0;padding-left:20px'>{items}</ul>")

        section("Churn Patterns")
        col1, col2 = st.columns(2)
        cr = px.bar(rate_by(fdf, "Contract"), x="Contract", y="Churn Rate (%)", text_auto=".1f",
                    title="Contract-wise Churn Rate", color="Churn Rate (%)",
                    color_continuous_scale=["#2A2F5C", BLUE, "#C7D2FE"])
        cr.update_layout(coloraxis_showscale=False)
        with col1:
            show(cr)
            note(top_rate_note(fdf, 'Contract', 'contract'))

        tn = px.histogram(with_status(fdf), x="tenure", color="Status", nbins=30, barmode="overlay", opacity=0.8,
                          title="Tenure-based Churn Pattern", labels={"tenure": "Tenure (months)"},
                          color_discrete_map=STATUS_COLORS)
        with col2:
            show(tn)
            _early, _late = fdf[fdf["tenure"] <= 12], fdf[fdf["tenure"] > 12]
            if len(_early) and len(_late):
                note(f"Customers in their first year churn at <b>{_early['Churn'].mean() * 100:.1f}%</b>, "
                     f"compared with <b>{_late['Churn'].mean() * 100:.1f}%</b> after 12 months.")

        col1, col2 = st.columns(2)
        ib = px.bar(count_by(fdf, "InternetService"), x="InternetService", y="Customers", color="Status",
                    barmode="group", title="Internet Service Analysis", color_discrete_map=STATUS_COLORS)
        with col1:
            show(ib)
            note(top_rate_note(fdf, 'InternetService', 'internet service'))

        pm = px.bar(count_by(fdf, "PaymentMethod"), x="PaymentMethod", y="Customers", color="Status",
                    barmode="group", title="Payment Method Comparison", color_discrete_map=STATUS_COLORS)
        with col2:
            show(pm)
            note(top_rate_note(fdf, 'PaymentMethod', 'payment method'))

        section("Recommendations")
        recs = "".join(f"<li>{r}</li>" for r in RECOMMENDATIONS)
        card(f"<ul style='margin:0;padding-left:20px'>{recs}</ul>")

        section("Customer Records and Downloads")
        st.dataframe(fdf.head(200), use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download filtered dataset (CSV)", fdf.to_csv(index=False).encode("utf-8"),
                           "filtered_customers.csv", "text/csv", use_container_width=True)
        d2.download_button("Download analysis report (TXT)", build_report(fdf).encode("utf-8"),
                           "churn_analysis_report.txt", "text/plain", use_container_width=True)


    # ================================================================ MODEL PERFORMANCE
    elif page.endswith("Model Performance"):
        hero("Model Performance",
             "Evaluation of the Logistic Regression model on a held-out 20% test set, "
             "calculated live from your trained model.")

        with st.spinner("Evaluating the model on the held-out test set..."):
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        c1, c2, c3, c4 = st.columns(4)
        kpi(c1, "🎯", "Accuracy", f"{accuracy * 100:.2f}%", BLUE, "Correct predictions overall")
        kpi(c2, "🔬", "Precision", f"{precision * 100:.2f}%", CYAN, "Predicted churners who left")
        kpi(c3, "📡", "Recall", f"{recall * 100:.2f}%", GREEN, "Actual churners we caught")
        kpi(c4, "⚖️", "F1 Score", f"{f1 * 100:.2f}%", AMBER, "Balance of precision and recall")

        card("<p><b>What these metrics mean.</b> <b>Accuracy</b> is the share of all customers the model classified "
             "correctly. <b>Precision</b> answers: when the model says a customer will churn, how often is it right? "
             "<b>Recall</b> answers: of all customers who actually churned, how many did the model find? "
             "<b>F1</b> combines precision and recall into one number. For churn work, recall matters most when "
             "missing a leaving customer is costly.</p>")

        col1, col2 = st.columns(2)
        cm = confusion_matrix(y_test, predictions)
        cm_fig = px.imshow(cm, text_auto=True, x=["Retained", "Churned"], y=["Retained", "Churned"],
                           labels={"x": "Predicted", "y": "Actual", "color": "Customers"},
                           title="Confusion Matrix", color_continuous_scale=["#161C2E", "#4F46E5", "#C7D2FE"])
        with col1:
            show(cm_fig, 420)

        report = classification_report(y_test, predictions, target_names=["Retained", "Churned"],
                                       output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(3)
        with col2:
            section("Classification Report")
            disp = report_df.copy().astype(object)
            for _i in disp.index:
                disp.loc[_i, "support"] = "" if _i == "accuracy" else f"{int(report_df.loc[_i, 'support']):,}"
            if "accuracy" in disp.index:
                disp.loc["accuracy", ["precision", "recall"]] = ""
            st.markdown(disp.to_html(classes="rep-table", border=0), unsafe_allow_html=True)
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"Of {tp + fn:,} customers who actually churned in the test set, the model found {tp:,} "
                       f"and missed {fn:,}. It also flagged {fp:,} customers who stayed.")

        section("Logistic Regression Feature Coefficients")
        try:
            classifier = model.named_steps["classifier"]
            preprocessor = model.named_steps["preprocessor"]
            imp = pd.DataFrame({"Feature": preprocessor.get_feature_names_out(),
                                "Coefficient": classifier.coef_[0]})
            imp["Abs"] = imp["Coefficient"].abs()
            imp = imp.sort_values("Abs", ascending=False).head(15).sort_values("Coefficient")
            imp["Effect"] = imp["Coefficient"].apply(lambda v: "Raises churn risk" if v > 0 else "Lowers churn risk")
            fig = px.bar(imp, x="Coefficient", y="Feature", orientation="h", color="Effect",
                         title="Top 15 Factors Influencing Customer Churn",
                         color_discrete_map={"Raises churn risk": CORAL, "Lowers churn risk": TEAL})
            show(fig, 520)
            st.caption("Positive coefficients increase the estimated likelihood of churn and negative coefficients "
                       "decrease it. These are model associations, not causal effects.")
        except Exception as e:
            st.warning(f"Feature coefficients could not be displayed for this model pipeline: {e}")


    # ================================================================ PREDICT
    elif page.endswith("Predict Churn"):
        hero("Predict Customer Churn",
             "Enter a customer's details to estimate their probability of leaving and see how risky the account is.")

        with st.form("prediction_form"):
            st.markdown("### Personal Information")
            col1, col2, col3 = st.columns(3)
            gender = col1.selectbox("Gender", ["Male", "Female"])
            senior = col2.selectbox("Senior Citizen", [0, 1])
            partner = col3.selectbox("Partner", ["Yes", "No"])
            dependents = col1.selectbox("Dependents", ["Yes", "No"])

            st.markdown("### Customer Services")
            col1, col2, col3 = st.columns(3)
            phone = col1.selectbox("Phone Service", ["Yes", "No"])
            multiple = col2.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = col3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = col1.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = col2.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device = col3.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech = col1.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            tv = col2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            movies = col3.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

            st.markdown("### Account Information")
            col1, col2, col3 = st.columns(3)
            tenure = col1.slider("Tenure (Months)", 0, 72, 12)
            contract = col2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = col3.selectbox("Paperless Billing", ["Yes", "No"])
            payment = col1.selectbox("Payment Method", ["Electronic check", "Mailed check",
                                                        "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly = col2.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
            total = col3.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0)

            submitted = st.form_submit_button("Predict Churn", use_container_width=True)

        if not submitted:
            card("<p><b>No prediction yet.</b><br>Fill in the customer details above and select "
                 "<b>Predict Churn</b> to see the probability, risk level and explanation.</p>")
        else:
            customer = pd.DataFrame([{
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": security,
                "OnlineBackup": backup,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }])

            prediction = model.predict(customer)[0]
            probability = float(model.predict_proba(customer)[0][1])
            risk, risk_color = risk_category(probability)

            st.divider()
            section("Prediction Results")
            col1, col2 = st.columns([1.1, 1])

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number=dict(suffix="%", font=dict(color=INK, size=48)),
                title=dict(text="Churn Probability", font=dict(color=MUTED, size=16)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)),
                    bar=dict(color=risk_color, thickness=0.28),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[dict(range=[0, 30], color="rgba(123,200,164,.22)"),
                           dict(range=[30, 60], color="rgba(240,176,64,.22)"),
                           dict(range=[60, 100], color="rgba(229,112,126,.25)")],
                ),
            ))
            with col1:
                show(gauge, 340)

            with col2:
                verdict = "Likely to Churn" if prediction == 1 else "Likely to Stay"
                card(f"<p style='margin:0 0 6px;color:{MUTED}'>Prediction</p>"
                     f"<p style='margin:0 0 14px;font-size:26px;font-weight:800;color:#F1F4FB'>{verdict}</p>"
                     f"<p style='margin:0 0 6px;color:{MUTED}'>Risk category</p>"
                     f"<span class='pill' style='background:{risk_color}33;color:{risk_color};"
                     f"border:1px solid {risk_color}'>{risk} risk</span>")

                if prediction == 1:
                    st.markdown(f'<div class="result-banner" style="background:{CORAL}22;border:1px solid {CORAL};'
                                f'color:{CORAL}">Churn expected: this customer is predicted to leave.</div>',
                                unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="result-banner" style="background:{GREEN}22;border:1px solid {GREEN};'
                                f'color:{GREEN}">No churn expected: this customer is predicted to stay.</div>',
                                unsafe_allow_html=True)
                st.caption(f"Probability of staying: {(1 - probability) * 100:.2f}%")

            explanations = {
                "Low": "The model sees few warning signs. Keep service quality high and consider "
                       "loyalty rewards to maintain the relationship.",
                "Medium": "The model sees mixed signals. A proactive check-in or a contract upgrade "
                          "offer could tip this customer toward staying.",
                "High": "The model sees strong churn signals. Prioritise this customer for retention outreach, "
                        "such as a tailored discount, a longer-term contract offer or a support call.",
            }
            card(f"<p><b>What this means.</b> The model estimates a <b>{probability * 100:.2f}%</b> chance that this "
                 f"customer will churn, which falls in the <b>{risk.lower()}-risk</b> band "
                 f"(low below 30%, medium 30-60%, high above 60%). {explanations[risk]}</p>")
            st.caption("This is a model-based estimate and not a guarantee of future customer behaviour.")


# ---------------------------------------------------------------- FOOTER
st.markdown(
    '<div class="footer">Customer Churn Prediction · Logistic Regression · '
    'Built with Python, Pandas, Scikit-learn, Plotly and Streamlit</div>',
    unsafe_allow_html=True,
)