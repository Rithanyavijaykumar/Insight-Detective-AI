"""
app.py  —  Insight Detective AI  |  Dark + Colorful + Interactive
"""
import os, json
import streamlit as st
import pandas as pd

from dotenv import load_dotenv
_base = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_base, ".env"), override=True)

from data_processor import load_dataset, clean_dataset, perform_eda
from visualizer     import (plot_missing_values, plot_numeric_distributions,
                             plot_correlation_heatmap, plot_boxplots,
                             plot_categorical_bars, plot_time_series,
                             plot_anomalies, plot_pairplot_sample)
from ml_analyzer    import (run_full_analysis, detect_anomalies,
                             find_strong_correlations)
from ai_reporter    import generate_report, generate_quick_summary

try:    API_KEY = st.secrets["GROQ_API_KEY"]
except: API_KEY = os.getenv("GROQ_API_KEY", "")

st.set_page_config(page_title="Insight Detective AI",
                   page_icon="🔍", layout="wide",
                   initial_sidebar_state="expanded")

# ═══════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*  { font-family:'Inter',sans-serif; box-sizing:border-box; }
.stApp { background:#0d0d0d; }
#MainMenu,footer,header { visibility:hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:#111 !important;
    border-right:1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color:#aaa !important; font-size:.84rem !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
    background:#161616;
    border-radius:12px;
    padding:5px 6px;
    gap:4px;
    border:1px solid #222;
}
.stTabs [data-baseweb="tab"]{
    background:transparent !important;
    border:none !important;
    border-radius:8px !important;
    color:#555 !important;
    font-size:.85rem !important;
    font-weight:600 !important;
    padding:.45rem 1.1rem !important;
    transition: all .2s !important;
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:#fff !important;
    box-shadow:0 2px 12px rgba(99,102,241,.4) !important;
}

/* ── Primary button ── */
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important;
    font-weight:700 !important; font-size:.92rem !important;
    padding:.7rem 2rem !important;
    box-shadow:0 4px 18px rgba(99,102,241,.35) !important;
    transition:all .2s !important;
}
.stButton>button[kind="primary"]:hover{
    transform:translateY(-2px) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.55) !important;
}

/* ── Download buttons ── */
.stDownloadButton>button{
    background:#1a1a1a !important; color:#ccc !important;
    border:1px solid #2a2a2a !important; border-radius:9px !important;
    font-size:.84rem !important; font-weight:500 !important;
    transition:all .2s !important;
}
.stDownloadButton>button:hover{
    border-color:#6366f1 !important; color:#fff !important;
    background:#1e1b3a !important;
}

/* ── Metrics ── */
[data-testid="metric-container"]{
    background:#161616 !important;
    border:1px solid #222 !important;
    border-radius:12px !important;
    padding:1.1rem !important;
}
[data-testid="stMetricValue"]{ color:#fff !important; font-weight:700 !important; font-size:1.6rem !important; }
[data-testid="stMetricLabel"]{ color:#555 !important; font-size:.75rem !important; text-transform:uppercase; letter-spacing:.06em; }

/* ── Expander ── */
.streamlit-expanderHeader{
    background:#161616 !important; border:1px solid #222 !important;
    border-radius:10px !important; color:#888 !important;
    font-size:.84rem !important; font-weight:600 !important;
}
.streamlit-expanderContent{
    background:#111 !important; border:1px solid #222 !important;
    border-top:none !important; border-radius:0 0 10px 10px !important;
}

/* ── Dataframe ── */
.stDataFrame{ border:1px solid #1e1e1e !important; border-radius:10px !important; }

/* ── Alerts ── */
.stSuccess{background:#0a1f0e !important; border:1px solid #14532d !important; border-radius:9px !important; color:#4ade80 !important;}
.stInfo   {background:#0c1a2e !important; border:1px solid #1e3a5f !important; border-radius:9px !important; color:#60a5fa !important;}
.stWarning{background:#1a1400 !important; border:1px solid #3a2e00 !important; border-radius:9px !important; color:#fbbf24 !important;}
.stError  {background:#1f0a0a !important; border:1px solid #5a1a1a !important; border-radius:9px !important; color:#f87171 !important;}

/* ── Spinner ── */
.stSpinner>div{ border-top-color:#6366f1 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px;}
::-webkit-scrollbar-track{background:#0d0d0d;}
::-webkit-scrollbar-thumb{background:#2a2a2a;border-radius:5px;}
::-webkit-scrollbar-thumb:hover{background:#444;}

/* ── Upload ── */
[data-testid="stFileUploader"]{
    background:#161616 !important; border:1px dashed #2e2e2e !important;
    border-radius:12px !important;
}

/* ── Stat cards ── */
.kpi-wrap { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin:1.2rem 0; }
.kpi {
    background:#161616; border:1px solid #222; border-radius:14px;
    padding:1.2rem 1rem; text-align:center; position:relative; overflow:hidden;
    transition:transform .2s, border-color .2s;
}
.kpi:hover { transform:translateY(-3px); border-color:#6366f1; }
.kpi-accent {
    position:absolute; top:0; left:0; right:0; height:3px;
    border-radius:14px 14px 0 0;
}
.kpi-icon  { font-size:1.4rem; margin-bottom:.3rem; }
.kpi-val   { font-size:1.8rem; font-weight:800; color:#fff; line-height:1.1; }
.kpi-label { font-size:.7rem; color:#555; text-transform:uppercase; letter-spacing:.07em; margin-top:.3rem; }

/* ── Insight strip ── */
.iq-wrap { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:.8rem 0 1.5rem; }
.iq {
    background:#161616; border:1px solid #222; border-radius:12px;
    padding:1.1rem 1.3rem; font-size:.88rem; color:#bbb; line-height:1.7;
    border-left:3px solid #6366f1; transition:border-color .2s, background .2s;
}
.iq:hover { background:#1a1a2e; border-left-color:#a78bfa; }

/* ── Section label ── */
.slabel {
    font-size:.68rem; letter-spacing:.14em; text-transform:uppercase;
    color:#555; margin:2rem 0 .7rem; display:flex; align-items:center; gap:.6rem;
}
.slabel::after { content:''; flex:1; height:1px; background:#1e1e1e; }

/* ── Anomaly cards ── */
.am-wrap { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:.8rem 0 1.4rem; }
.am { border-radius:12px; padding:1.2rem 1rem; text-align:center; border:1px solid; }
.am-a { background:#0f1629; border-color:#1e3a5f; }
.am-b { background:#0a1f0e; border-color:#14532d; }
.am-c { background:#1f0a0a; border-color:#5a1a1a; }
.am-d { background:#1a1200; border-color:#3a2800; }
.am-val { font-size:1.9rem; font-weight:800; line-height:1.1; }
.am-lbl { font-size:.68rem; text-transform:uppercase; letter-spacing:.07em; color:#555; margin-top:.3rem; }
.blue   { color:#60a5fa; } .green { color:#4ade80; }
.red    { color:#f87171; } .amber { color:#fbbf24; }

/* ── Report box ── */
.rbox {
    background:#111; border:1px solid #222; border-radius:14px;
    padding:2rem 2.5rem; line-height:1.9; color:#ccc; font-size:.9rem;
    border-top:3px solid #6366f1; margin-top:1rem;
}
.rbox h2 { color:#a78bfa; border-bottom:1px solid #1e1e1e; padding-bottom:.4rem; font-size:1.1rem; }
.rbox h3 { color:#60a5fa; font-size:1rem; }
.rbox li { margin-bottom:.4rem; }
.rbox strong { color:#f0abfc; }
.rbox p { color:#999; }

/* ── Export cards ── */
.ex-card {
    background:#161616; border:1px solid #222; border-radius:12px;
    padding:1.3rem 1.4rem; margin-bottom:1rem;
}
.ex-title { font-size:.78rem; text-transform:uppercase; letter-spacing:.09em; color:#555; margin-bottom:.8rem; }

/* ── Hero ── */
.hero {
    background:linear-gradient(135deg,#111 0%,#161629 100%);
    border:1px solid #222; border-radius:18px;
    padding:3rem 2.5rem 2.5rem; margin-bottom:1.8rem; position:relative; overflow:hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:300px; height:300px; border-radius:50%;
    background:radial-gradient(circle,rgba(99,102,241,.15),transparent 70%);
}
.hero::after {
    content:''; position:absolute; bottom:-40px; left:30%;
    width:200px; height:200px; border-radius:50%;
    background:radial-gradient(circle,rgba(139,92,246,.1),transparent 70%);
}
.hero-eyebrow {
    display:inline-block; background:rgba(99,102,241,.15);
    border:1px solid rgba(99,102,241,.3); border-radius:50px;
    padding:.25rem .9rem; font-size:.72rem; font-weight:600;
    color:#818cf8; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.9rem;
}
.hero-title {
    font-size:2.6rem; font-weight:800; color:#fff;
    letter-spacing:-.03em; line-height:1.1; margin-bottom:.7rem;
}
.hero-title span { color:#818cf8; }
.hero-sub { color:#555; font-size:.95rem; max-width:500px; line-height:1.6; margin-bottom:1.5rem; }
.hero-chips { display:flex; flex-wrap:wrap; gap:.5rem; }
.chip {
    background:#1a1a1a; border:1px solid #2a2a2a;
    border-radius:50px; padding:.3rem .85rem;
    font-size:.75rem; color:#777; font-weight:500;
    transition:all .2s;
}
.chip:hover { border-color:#6366f1; color:#a78bfa; background:#1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:1.5rem 0 1rem;">
        <div style="font-size:2rem; margin-bottom:.4rem;">🔍</div>
        <div style="font-size:1rem; font-weight:700; color:#fff;">Insight Detective</div>
        <div style="font-size:.75rem; color:#444; margin-top:.2rem;">AI · v2.0</div>
    </div>
    <div style="height:1px;background:#1e1e1e;margin-bottom:1.2rem;"></div>
    <div style="font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;
                color:#444;margin-bottom:.9rem;">⚙️ Analysis Settings</div>
    """, unsafe_allow_html=True)

    contamination    = st.slider("Anomaly sensitivity",   0.01, 0.20, 0.05, 0.01)
    corr_threshold   = st.slider("Correlation threshold", 0.50, 0.99, 0.70, 0.05)
    top_n_categories = st.slider("Top N categories",       3,   20,   10,   1)

    st.markdown("""
    <div style="height:1px;background:#1e1e1e;margin:1.2rem 0;"></div>
    <div style="font-size:.72rem;color:#333;line-height:2.1;">
        <span style="color:#6366f1;">■</span> Pandas · NumPy · SciPy<br>
        <span style="color:#8b5cf6;">■</span> Plotly · Matplotlib<br>
        <span style="color:#06b6d4;">■</span> Scikit-Learn · PCA<br>
        <span style="color:#10b981;">■</span> LLaMA 3.3 70B · Groq<br>
        <span style="color:#f59e0b;">■</span> Streamlit
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🔬 AI-Powered Data Intelligence</div>
    <div class="hero-title">Insight <span>Detective</span> AI</div>
    <div class="hero-sub">
        Upload any dataset — get automated EDA, anomaly detection,
        trend analysis and a full AI executive report in seconds.
    </div>
    <div class="hero-chips">
        <span class="chip">📊 Auto EDA</span>
        <span class="chip">🧹 Smart Cleaning</span>
        <span class="chip">🚨 Anomaly Detection</span>
        <span class="chip">📈 Trend Analysis</span>
        <span class="chip">🔗 Correlations</span>
        <span class="chip">🤖 LLM Report</span>
        <span class="chip">📥 Export</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  UPLOAD
# ═══════════════════════════════════════════════════════════
uploaded_file = st.file_uploader(
    "📁  Upload dataset — CSV or Excel (.xlsx / .xls)",
    type=["csv","xlsx","xls"],
)

if not uploaded_file:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#2a2a2a;font-size:.9rem;">
        ☝️  Drop a CSV above to begin analysis
        <br><span style="font-size:.8rem;color:#222;">
        Try: <code style="color:#6366f1;">sample_data/customer_data.csv</code></span>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════
#  LOAD + CLEAN
# ═══════════════════════════════════════════════════════════
with st.spinner("🧹 Loading and cleaning..."):
    try:
        df_raw = load_dataset(uploaded_file)
        df, cleaning_report = clean_dataset(df_raw.copy())
    except Exception as e:
        st.error(f"Failed to load: {e}"); st.stop()

# ═══════════════════════════════════════════════════════════
#  KPI CARDS
# ═══════════════════════════════════════════════════════════
kpis = [
    ("📊", f"{df.shape[0]:,}",  "Total Rows",        "linear-gradient(90deg,#6366f1,#8b5cf6)"),
    ("📋", str(df.shape[1]),     "Columns",           "linear-gradient(90deg,#06b6d4,#3b82f6)"),
    ("🔢", str(len(df.select_dtypes(include="number").columns)), "Numeric",
                                                       "linear-gradient(90deg,#10b981,#059669)"),
    ("🏷️", str(len(df.select_dtypes(include="object").columns)), "Categorical",
                                                       "linear-gradient(90deg,#f59e0b,#ef4444)"),
    ("🧹", str(cleaning_report.get("duplicate_rows_removed",0)), "Dupes Removed",
                                                       "linear-gradient(90deg,#8b5cf6,#ec4899)"),
]
html_kpis = '<div class="kpi-wrap">' + "".join(f"""
<div class="kpi">
  <div class="kpi-accent" style="background:{grad};"></div>
  <div class="kpi-icon">{ic}</div>
  <div class="kpi-val">{val}</div>
  <div class="kpi-label">{lbl}</div>
</div>""" for ic,val,lbl,grad in kpis) + '</div>'
st.markdown(html_kpis, unsafe_allow_html=True)

with st.expander("🧹 Data cleaning details"):
    d1, d2 = st.columns(2)
    with d1:
        st.write(f"Original: `{cleaning_report.get('original_shape')}`")
        st.write(f"Cleaned:  `{cleaning_report.get('final_shape')}`")
    with d2:
        miss = cleaning_report.get("missing_values_before",{})
        if miss:
            for k,v in miss.items(): st.write(f"`{k}` — {v} filled")
        else: st.success("No missing values")

# ═══════════════════════════════════════════════════════════
#  EDA + ML
# ═══════════════════════════════════════════════════════════
with st.spinner("🔬 Running EDA + ML analysis..."):
    eda        = perform_eda(df)
    ml_results = run_full_analysis(df)
    ml_results["anomalies"]           = detect_anomalies(df, contamination)
    ml_results["strong_correlations"] = find_strong_correlations(df, corr_threshold)

# ═══════════════════════════════════════════════════════════
#  QUICK INSIGHTS
# ═══════════════════════════════════════════════════════════
with st.spinner("✨ Generating AI insights..."):
    try:
        raw   = generate_quick_summary(df.shape, eda, API_KEY)
        lines = [l.strip() for l in raw.strip().split("\n") if l.strip()][:3]
        st.markdown('<div class="slabel">✨ Quick AI Insights</div>', unsafe_allow_html=True)
        cards = "".join(f'<div class="iq">{l}</div>' for l in lines)
        st.markdown(f'<div class="iq-wrap">{cards}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Quick insights unavailable: {e}")

st.markdown("<div style='height:1px;background:#1e1e1e;margin:1rem 0 1.5rem;'></div>",
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════
def slabel(t):
    st.markdown(f'<div class="slabel">{t}</div>', unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "📋 Data Preview","📈 Visualizations","🤖 ML Analysis","📝 AI Report","📥 Export"
])

# ── TAB 1 ──────────────────────────────────────────────────
with tab1:
    slabel("Raw data — first 100 rows")
    st.dataframe(df.head(100), use_container_width=True, height=300)

    slabel("Column overview")
    st.dataframe(pd.DataFrame({
        "Column":  df.columns,
        "Type":    df.dtypes.values.astype(str),
        "Non-Null":df.notnull().sum().values,
        "Nulls":   df.isnull().sum().values,
        "Unique":  [df[c].nunique() for c in df.columns],
        "Sample":  [str(df[c].dropna().iloc[0])
                    if df[c].dropna().shape[0]>0 else "—" for c in df.columns],
    }), use_container_width=True)

    slabel("Descriptive statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────────
with tab2:
    fig_miss = plot_missing_values(df)
    if fig_miss:
        slabel("Missing values"); st.plotly_chart(fig_miss, use_container_width=True)
    else:
        st.success("✅ No missing values")

    fig_dist = plot_numeric_distributions(df)
    if fig_dist:
        slabel("Distributions"); st.plotly_chart(fig_dist, use_container_width=True)

    v1,v2 = st.columns(2)
    with v1:
        fig_box = plot_boxplots(df)
        if fig_box: slabel("Box plots"); st.plotly_chart(fig_box, use_container_width=True)
    with v2:
        fig_corr = plot_correlation_heatmap(df)
        if fig_corr: slabel("Correlation heatmap"); st.plotly_chart(fig_corr, use_container_width=True)
        else: st.info("Need ≥2 numeric columns")

    fig_pair = plot_pairplot_sample(df)
    if fig_pair:
        slabel("Pair plot"); st.plotly_chart(fig_pair, use_container_width=True)

    cat_figs = plot_categorical_bars(df, top_n=top_n_categories)
    if cat_figs:
        slabel("Categorical distributions")
        for i in range(0, len(cat_figs), 2):
            cc = st.columns(2)
            for j, fig in enumerate(cat_figs[i:i+2]):
                with cc[j]: st.plotly_chart(fig, use_container_width=True)

    ts_figs = plot_time_series(df)
    if ts_figs:
        slabel("Time series")
        for fig in ts_figs: st.plotly_chart(fig, use_container_width=True)

# ── TAB 3 ──────────────────────────────────────────────────
with tab3:
    ai    = ml_results["anomalies"]
    total = len(df)
    n_bad = ai.get("anomaly_count",0)
    pct   = ai.get("anomaly_pct",0.0)

    slabel("Anomaly detection — Isolation Forest")
    st.markdown(f"""
    <div class="am-wrap">
      <div class="am am-a"><div class="am-val blue">{total:,}</div>
        <div class="am-lbl">Rows Analyzed</div></div>
      <div class="am am-b"><div class="am-val green">{total-n_bad:,}</div>
        <div class="am-lbl">Normal</div></div>
      <div class="am am-c"><div class="am-val red">{n_bad:,}</div>
        <div class="am-lbl">Anomalies</div></div>
      <div class="am am-d"><div class="am-val amber">{pct}%</div>
        <div class="am-lbl">Anomaly Rate</div></div>
    </div>""", unsafe_allow_html=True)

    labels      = ai.get("labels")
    primary_col = ai.get("primary_column")
    if labels is not None and primary_col:
        fig_a = plot_anomalies(df, primary_col, labels)
        if fig_a: st.plotly_chart(fig_a, use_container_width=True)
        with st.expander("🔎 View anomalous rows"):
            st.dataframe(df[labels==-1], use_container_width=True)

    m1,m2 = st.columns(2)
    with m1:
        slabel("Trend analysis")
        trends = ml_results.get("trends",{})
        if trends:
            st.dataframe(pd.DataFrame([
                {"Column":c,"Direction":v["direction"],
                 "Slope":v["slope"],"R²":v["r_squared"],"p":v["p_value"]}
                for c,v in trends.items()
            ]), use_container_width=True)
        else: st.info("No numeric columns")
    with m2:
        slabel("Strong correlations")
        sc = ml_results.get("strong_correlations",[])
        if sc: st.dataframe(pd.DataFrame(sc), use_container_width=True)
        else:  st.info(f"None above {corr_threshold}")

    m3,m4 = st.columns(2)
    with m3:
        slabel("Distribution analysis")
        di = ml_results.get("distributions",{})
        if di:
            st.dataframe(pd.DataFrame([
                {"Column":c,"Skewness":v["skewness"],"Shape":v["skew_label"],
                 "Normal":"Yes" if v["is_normal"] else "No",
                 "p":v["normality_p_value"]}
                for c,v in di.items()
            ]), use_container_width=True)
    with m4:
        pca_d = ml_results.get("pca",{})
        if pca_d:
            slabel("PCA explained variance")
            st.dataframe(pd.DataFrame(list(pca_d.items()),
                         columns=["Component","Variance %"]),
                         use_container_width=True)

# ── TAB 4 ──────────────────────────────────────────────────
with tab4:
    st.markdown("""
    <div style="background:#161616;border:1px solid #222;border-radius:14px;
                padding:1.4rem 1.8rem;margin-bottom:1.5rem;border-left:4px solid #6366f1;">
        <div style="font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;
                    color:#555;margin-bottom:.3rem;">AI Report</div>
        <div style="font-size:1.15rem;font-weight:700;color:#fff;">
            AI-Generated Investigation Report
        </div>
        <div style="font-size:.8rem;color:#444;margin-top:.3rem;">
            LLaMA 3.3 70B via Groq · Executive-level analysis
        </div>
    </div>""", unsafe_allow_html=True)

    if "report_text" not in st.session_state:
        st.session_state.report_text = None

    if st.button("🚀  Generate Full AI Report", type="primary", use_container_width=True):
        with st.spinner("🧠  Writing your investigation report… (15–30 sec)"):
            try:
                st.session_state.report_text = generate_report(
                    eda, ml_results, uploaded_file.name, API_KEY)
            except Exception as e:
                st.error(f"Failed: {e}")
                st.session_state.report_text = None

    if st.session_state.report_text:
        st.markdown(
            f'<div class="rbox">{st.session_state.report_text}</div>',
            unsafe_allow_html=True)

# ── TAB 5 ──────────────────────────────────────────────────
with tab5:
    slabel("Downloads")
    e1,e2 = st.columns(2)

    with e1:
        st.markdown('<div class="ex-card"><div class="ex-title">🧹 Cleaned Dataset</div>',
                    unsafe_allow_html=True)
        st.download_button("⬇ Download cleaned CSV",
            data=df.to_csv(index=False).encode(),
            file_name=f"cleaned_{uploaded_file.name.replace('.xlsx','.csv').replace('.xls','.csv')}",
            mime="text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ex-card"><div class="ex-title">📊 EDA Statistics (JSON)</div>',
                    unsafe_allow_html=True)
        eda_safe = {k:(list(v) if isinstance(v,tuple) else v) for k,v in eda.items()}
        st.download_button("⬇ Download EDA JSON",
            data=json.dumps(eda_safe,indent=2,default=str).encode(),
            file_name="eda_stats.json", mime="application/json",
            use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with e2:
        st.markdown('<div class="ex-card"><div class="ex-title">🚨 Anomaly-Flagged Data</div>',
                    unsafe_allow_html=True)
        if ml_results["anomalies"].get("labels") is not None:
            df_a = df.copy()
            df_a["anomaly_flag"] = ml_results["anomalies"]["labels"].values
            df_a["anomaly_flag"] = df_a["anomaly_flag"].map({1:"Normal",-1:"Anomaly"})
            st.download_button("⬇ Download anomaly CSV",
                data=df_a.to_csv(index=False).encode(),
                file_name=f"anomalies_{uploaded_file.name.replace('.xlsx','.csv').replace('.xls','.csv')}",
                mime="text/csv", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="ex-card"><div class="ex-title">📝 AI Report (Markdown)</div>',
                    unsafe_allow_html=True)
        if st.session_state.get("report_text"):
            st.download_button("⬇ Download report",
                data=st.session_state.report_text.encode(),
                file_name="insight_report.md", mime="text/markdown",
                use_container_width=True)
        else:
            st.markdown("<div style='font-size:.82rem;color:#333;'>"
                        "Generate the report first (AI Report tab)</div>",
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
