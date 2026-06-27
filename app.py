"""
app.py  —  Noon Academy Intervention Dashboard
Premium Streamlit UI matching Noon Academy brand identity.

Architecture
────────────
• No dependency on pre-existing output files or main.py.
• Upload CSVs in-browser OR fall back to config.DATA_DIR automatically.
• Full pipeline runs in session_state — zero disk writes required.
• Yellow tier: deterministic messages generated instantly after pipeline.
• Red tier: on-demand or small-batch LLM generation with live progress.
• Quota-safe: QuotaExhaustedError caught gracefully with clear UI feedback.
• Approved messages exported via browser download (no disk path needed).

Run:  streamlit run app.py
"""

import io, json, logging, math, tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

import config
from data_pipeline import build_master_dataframe
from risk_engine import get_tier, TIER_RED, TIER_YELLOW, TIER_GREEN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Noon Academy — Intervention",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# NOON ACADEMY BRAND CSS
# Primary teal  : #00ab77  (Noon green)
# Accent gold   : #f5a623  (alert/warning)
# Dark bg       : #0d1117  (near-black)
# Card bg       : #161b22
# Border        : #21262d
# Text primary  : #e6edf3
# Text muted    : #8b949e
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
}
.stApp { background-color: #0d1117; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #00ab77;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 2px;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown small {
    color: #8b949e;
    font-size: 12px;
}
[data-testid="stSidebar"] hr { border-color: #21262d; }

/* ── Page title ── */
.noon-title {
    font-size: 28px;
    font-weight: 800;
    color: #e6edf3;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.noon-title span { color: #00ab77; }
.noon-subtitle {
    font-size: 13px;
    color: #8b949e;
    margin-top: 4px;
    margin-bottom: 0;
}

/* ── Section headers ── */
.section-header {
    font-size: 15px;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #21262d;
}

/* ── KPI scoreboard ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 10px;
    margin: 14px 0 20px 0;
}
.kpi-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 14px 12px 12px 12px;
    text-align: center;
}
.kpi-card.red   { border-top: 3px solid #e53e3e; }
.kpi-card.yellow{ border-top: 3px solid #f5a623; }
.kpi-card.green { border-top: 3px solid #00ab77; }
.kpi-card.teal  { border-top: 3px solid #00ab77; }
.kpi-card.white { border-top: 3px solid #30363d; }
.kpi-label {
    font-size: 11px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 30px;
    font-weight: 700;
    color: #e6edf3;
    line-height: 1;
}
.kpi-value.red    { color: #e53e3e; }
.kpi-value.yellow { color: #f5a623; }
.kpi-value.green  { color: #00ab77; }
.kpi-delta {
    font-size: 11px;
    color: #8b949e;
    margin-top: 4px;
}

/* ── Metric badges ── */
.badge {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
}
.badge-critical { background: #3d1212; color: #ff6b6b; border: 1px solid #e53e3e; }
.badge-warning  { background: #2d2200; color: #f5a623; border: 1px solid #d48806; }
.badge-ok       { background: #0d2818; color: #00ab77; border: 1px solid #00ab77; }
.badge-info     { background: #0c1e35; color: #58a6ff; border: 1px solid #1f6feb; }

/* ── Student cards ── */
.student-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.student-card.red-card    { border-left: 4px solid #e53e3e; }
.student-card.yellow-card { border-left: 4px solid #f5a623; }
.card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
}
.card-name {
    font-size: 15px;
    font-weight: 700;
    color: #e6edf3;
}
.card-id {
    font-size: 11px;
    color: #8b949e;
    font-family: monospace;
}
.metric-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.metric-chip {
    display: flex;
    flex-direction: column;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 7px;
    padding: 7px 13px;
    min-width: 110px;
}
.metric-chip-label {
    font-size: 10px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 3px;
}
.metric-chip-value {
    font-size: 17px;
    font-weight: 700;
    color: #e6edf3;
}
.metric-chip-value.critical { color: #ff6b6b; }
.metric-chip-value.warning  { color: #f5a623; }
.metric-chip-value.ok       { color: #00ab77; }

/* ── Arabic message area ── */
.arabic-label {
    font-size: 12px;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin-bottom: 5px;
}
[data-testid="stTextArea"] textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    direction: rtl;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #00ab77 !important;
    box-shadow: 0 0 0 2px rgba(0,171,119,0.15) !important;
}

/* ── Guardrail warning ── */
.guardrail-warn {
    background: #2d1f00;
    border: 1px solid #d48806;
    border-radius: 7px;
    padding: 9px 14px;
    font-size: 13px;
    color: #f5a623;
    margin-bottom: 10px;
}
.guardrail-error {
    background: #1f0d0d;
    border: 1px solid #e53e3e;
    border-radius: 7px;
    padding: 9px 14px;
    font-size: 13px;
    color: #ff6b6b;
    margin-bottom: 10px;
}

/* ── Quota warning banner ── */
.quota-banner {
    background: #1a1500;
    border: 1px solid #d48806;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 14px;
    color: #f5a623;
    font-size: 13px;
}

/* ── Pipeline success banner ── */
.pipeline-success {
    background: #0a1f14;
    border: 1px solid #00ab77;
    border-radius: 8px;
    padding: 11px 16px;
    color: #00ab77;
    font-size: 13px;
    font-weight: 600;
}

/* ── Buttons ── */
.stButton > button {
    background: #00ab77;
    color: #fff;
    border: none;
    border-radius: 7px;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 18px;
    transition: background 0.18s;
}
.stButton > button:hover { background: #009966; }
.stButton > button:disabled {
    background: #21262d !important;
    color: #8b949e !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    color: #8b949e;
    font-size: 13px;
    font-weight: 600;
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #00ab77;
    border-bottom: 2px solid #00ab77;
    background: transparent;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    margin-bottom: 8px;
}
[data-testid="stExpander"] summary {
    font-weight: 600;
    color: #e6edf3;
    padding: 10px 14px;
}
[data-testid="stExpander"] summary:hover { background: #1c2128; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
.stDataFrame thead tr th {
    background: #161b22 !important;
    color: #8b949e !important;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}
.stDataFrame tbody tr:hover td { background: #1c2128 !important; }

/* ── File uploaders ── */
[data-testid="stFileUploader"] {
    background: #161b22;
    border: 1px dashed #30363d;
    border-radius: 8px;
    padding: 4px;
}
[data-testid="stFileUploader"]:hover { border-color: #00ab77; }

/* ── Metrics (native) ── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px 14px;
}
[data-testid="stMetricValue"] { color: #e6edf3 !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Divider ── */
hr { border-color: #21262d; margin: 18px 0; }

/* ── Checkbox approve ── */
[data-testid="stCheckbox"] label {
    font-size: 13px;
    font-weight: 600;
    color: #00ab77;
}

/* ── Info / warning / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px;
    font-size: 13px;
}

/* ── Slider ── */
[data-testid="stSlider"] [role="slider"] { background: #00ab77 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC YELLOW MESSAGE TEMPLATE
# ═══════════════════════════════════════════════════════════════════════════════

YELLOW_TEMPLATE = (
    "مرحباً {name}! 👋 تذكير ودي: اختبار Quiz 2 بعد 6 أيام فقط. "
    "حضرت في المتوسط {att:.0f} دقيقة/جلسة وأكملت {pq:.0f} سؤال تدريبي يومياً. "
    "حاول الوصول إلى 90 دقيقة + 10 أسئلة يومياً — أنت قادر على ذلك! 💪\n\n"
    "Hi {name}! Quiz 2 is in 6 days. You've averaged {att:.0f} min/session and "
    "{pq:.0f} practice Q/day. Aim for 90 min + 10 questions daily. You've got this!"
)


def _build_yellow_messages(yellow_df: pd.DataFrame) -> pd.DataFrame:
    out = yellow_df.copy()
    out["message"] = out.apply(
        lambda r: YELLOW_TEMPLATE.format(
            name=r.get("student_name", r["student_id"]),
            att=float(r.get("avg_minutes_per_session") or 0),
            pq=float(r.get("avg_practice_per_day") or 0),
        ),
        axis=1,
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "tiered_df":    None,
        "red_df":       None,
        "yellow_df":    None,   # already has 'message' column after pipeline
        "messages":     {},     # {sid: {message, guardrail_passed, guardrail_reason}}
        "approved":     {},     # {sid: bool}
        "quota_hit":    False,
        "pipeline_ran": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _write_tmp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)


def _render_upload_section() -> dict[str, Optional[Path]]:
    st.markdown('<div class="section-header">📂 Data Source</div>', unsafe_allow_html=True)
    st.caption(
        f"Upload your three CSVs — or leave empty to read from "
        f"`{config.DATA_DIR}` (default data folder)."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        meta    = st.file_uploader("student_metadata.csv",      type="csv", key="up_meta")
    with col2:
        metrics = st.file_uploader("student_daily_metrics.csv", type="csv", key="up_metrics")
    with col3:
        notes   = st.file_uploader("facilitator_notes.csv",     type="csv", key="up_notes")

    return {
        "meta":    _write_tmp(meta)    if meta    else None,
        "metrics": _write_tmp(metrics) if metrics else None,
        "notes":   _write_tmp(notes)   if notes   else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(paths: dict[str, Optional[Path]]):
    orig = {
        "STUDENT_METADATA_PATH":  config.STUDENT_METADATA_PATH,
        "STUDENT_METRICS_PATH":   config.STUDENT_METRICS_PATH,
        "FACILITATOR_NOTES_PATH": config.FACILITATOR_NOTES_PATH,
    }
    if paths["meta"]:    config.STUDENT_METADATA_PATH  = paths["meta"]
    if paths["metrics"]: config.STUDENT_METRICS_PATH   = paths["metrics"]
    if paths["notes"]:   config.FACILITATOR_NOTES_PATH = paths["notes"]

    try:
        with st.spinner("⚙️ Cleaning data and computing risk tiers …"):
            tiered   = build_master_dataframe()
            red_df   = get_tier(tiered, TIER_RED)
            yellow_df = get_tier(tiered, TIER_YELLOW)
            yellow_df = _build_yellow_messages(yellow_df)   # instant — no LLM

        st.session_state.update({
            "tiered_df":    tiered,
            "red_df":       red_df,
            "yellow_df":    yellow_df,
            "pipeline_ran": True,
            "messages":     {},
            "approved":     {},
            "quota_hit":    False,
        })

    except FileNotFoundError as exc:
        st.error(
            f"**File not found:** {exc}\n\n"
            f"Upload all three CSVs above, or place them in `{config.DATA_DIR}`."
        )
        st.session_state["pipeline_ran"] = False
    except Exception as exc:
        st.error(f"**Pipeline error:** {exc}")
        logger.exception("Pipeline crashed")
        st.session_state["pipeline_ran"] = False
    finally:
        for attr, val in orig.items():
            setattr(config, attr, val)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            '<h2 style="color:#00ab77;font-size:20px;font-weight:800;margin-bottom:0">🎓 Noon Academy</h2>',
            unsafe_allow_html=True,
        )
        days_left = config.QUIZ2_DAY - config.CURRENT_DAY
        st.markdown(
            f'<p style="color:#8b949e;font-size:12px;margin-top:2px">'
            f'Day {config.CURRENT_DAY} &nbsp;·&nbsp; Quiz 2 in '
            f'<b style="color:#f5a623">{days_left} days</b></p>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="border-color:#21262d;margin:12px 0">', unsafe_allow_html=True)

        campus = "All"
        if st.session_state["pipeline_ran"] and st.session_state["tiered_df"] is not None:
            campuses = sorted(
                st.session_state["tiered_df"]["campus_id"].dropna().unique().tolist()
            )
            campus = st.selectbox("📍 Campus", ["All"] + campuses)
            st.markdown('<hr style="border-color:#21262d;margin:10px 0">', unsafe_allow_html=True)

        st.markdown(
            '<p style="color:#8b949e;font-size:11px;text-transform:uppercase;'
            'letter-spacing:1px;font-weight:600;margin-bottom:8px">⚙️ LLM Settings</p>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "Gemini API Key",
            value=config.GEMINI_API_KEY or "",
            type="password",
            help="Free key at https://aistudio.google.com/apikey",
            key="sidebar_api_key",
        )
        if api_key.strip():
            config.GEMINI_API_KEY = api_key.strip()

        batch_cap = st.slider(
            "Max per batch",
            min_value=1, max_value=15, value=5,
            help="Free tier: ~20 req/day. Keep batches small.",
            key="batch_cap",
        )
        st.markdown(
            '<p style="color:#8b949e;font-size:11px;margin-top:4px">'
            'Free tier: ~20 req/day. Use per-student buttons or small batches.</p>',
            unsafe_allow_html=True,
        )

        if st.session_state["quota_hit"]:
            st.markdown(
                '<div class="quota-banner">⚠️ <b>Quota exhausted.</b><br>'
                'Remaining messages can be generated tomorrow.</div>',
                unsafe_allow_html=True,
            )

        if st.session_state["pipeline_ran"]:
            st.markdown('<hr style="border-color:#21262d;margin:12px 0">', unsafe_allow_html=True)
            tiered = st.session_state["tiered_df"]
            counts = tiered["risk_tier"].value_counts()
            r, y, g = counts.get(TIER_RED,0), counts.get(TIER_YELLOW,0), counts.get(TIER_GREEN,0)
            st.markdown(
                f'<div style="font-size:12px;color:#8b949e;line-height:1.9">'
                f'<span style="color:#e53e3e">●</span> Red &nbsp;&nbsp;<b style="color:#e6edf3">{r}</b><br>'
                f'<span style="color:#f5a623">●</span> Yellow <b style="color:#e6edf3">{y}</b><br>'
                f'<span style="color:#00ab77">●</span> Green &nbsp;<b style="color:#e6edf3">{g}</b></div>',
                unsafe_allow_html=True,
            )

    return campus


# ═══════════════════════════════════════════════════════════════════════════════
# KPI SCOREBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def _render_kpi(tiered_df: pd.DataFrame):
    counts = tiered_df["risk_tier"].value_counts()
    total  = len(tiered_df)
    red_n  = counts.get(TIER_RED, 0)
    yel_n  = counts.get(TIER_YELLOW, 0)
    grn_n  = counts.get(TIER_GREEN, 0)
    drafts = len(st.session_state["messages"])
    appr   = sum(1 for v in st.session_state["approved"].values() if v)
    red_pct = round(100 * red_n / max(total, 1))

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card white">
        <div class="kpi-label">Total Students</div>
        <div class="kpi-value">{total}</div>
      </div>
      <div class="kpi-card red">
        <div class="kpi-label">🔴 High Risk</div>
        <div class="kpi-value red">{red_n}</div>
        <div class="kpi-delta">{red_pct}% of cohort</div>
      </div>
      <div class="kpi-card yellow">
        <div class="kpi-label">🟡 Medium Risk</div>
        <div class="kpi-value yellow">{yel_n}</div>
      </div>
      <div class="kpi-card green">
        <div class="kpi-label">🟢 On Track</div>
        <div class="kpi-value green">{grn_n}</div>
      </div>
      <div class="kpi-card teal">
        <div class="kpi-label">✉️ Drafts Ready</div>
        <div class="kpi-value">{drafts}</div>
      </div>
      <div class="kpi-card white">
        <div class="kpi-label">✅ Approved</div>
        <div class="kpi-value">{appr}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC CHIP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _att_class(val: float) -> str:
    if val < config.ATTENDANCE_THRESHOLD_RED_MIN:    return "critical"
    if val < config.ATTENDANCE_THRESHOLD_YELLOW_MIN: return "warning"
    return "ok"

def _pq_class(val: float) -> str:
    if val < config.PRACTICE_RED_THRESHOLD:    return "critical"
    if val < config.PRACTICE_YELLOW_THRESHOLD: return "warning"
    return "ok"

def _quiz_class(val: float) -> str:
    if val < config.PASSING_GRADE:      return "critical"
    if val < config.PASSING_GRADE + 10: return "warning"
    return "ok"

def _miss_class(val: int) -> str:
    if val >= config.CONSECUTIVE_MISS_RED:    return "critical"
    if val >= config.CONSECUTIVE_MISS_YELLOW: return "warning"
    return "ok"

def _metric_chip(label: str, value: str, cls: str) -> str:
    return (
        f'<div class="metric-chip">'
        f'  <div class="metric-chip-label">{label}</div>'
        f'  <div class="metric-chip-value {cls}">{value}</div>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LLM GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_single(student: dict):
    from llm_service import generate_outreach_message, QuotaExhaustedError
    sid = student["student_id"]
    try:
        result = generate_outreach_message(dict(student), retries=1)
        st.session_state["messages"][sid] = result
    except QuotaExhaustedError:
        st.session_state["quota_hit"] = True
        st.session_state["messages"][sid] = {
            "student_id": sid, "student_name": student.get("student_name", sid),
            "message": "[Quota exhausted — try again tomorrow or use a paid API key]",
            "guardrail_passed": False, "guardrail_reason": "QuotaExhaustedError",
        }


def _generate_batch(red_df: pd.DataFrame, cap: int):
    from llm_service import generate_outreach_message, QuotaExhaustedError

    pending = [
        dict(row) for _, row in red_df.iterrows()
        if row["student_id"] not in st.session_state["messages"]
    ][:cap]

    if not pending:
        st.info("All visible students already have drafts.")
        return

    bar   = st.progress(0, text="Starting …")
    total = len(pending)

    for i, student in enumerate(pending, 1):
        name = student.get("student_name", student.get("student_id"))
        bar.progress(i / total, text=f"Generating {i}/{total} — {name} …")
        sid = student["student_id"]
        try:
            result = generate_outreach_message(student, retries=1)
            st.session_state["messages"][sid] = result
        except QuotaExhaustedError:
            st.session_state["quota_hit"] = True
            st.session_state["messages"][sid] = {
                "student_id": sid, "student_name": name,
                "message": "[Quota exhausted — try again tomorrow]",
                "guardrail_passed": False, "guardrail_reason": "QuotaExhaustedError",
            }
            bar.empty()
            st.markdown(
                f'<div class="quota-banner">⚠️ <b>Daily quota exhausted</b> after {i-1} message(s). '
                'Generated drafts are preserved above. Remaining students can be processed tomorrow.</div>',
                unsafe_allow_html=True,
            )
            return

    bar.empty()
    passed = sum(
        1 for sid in st.session_state["messages"]
        if st.session_state["messages"][sid].get("guardrail_passed")
    )
    st.success(f"✅ Generated {total} draft(s) — {passed} passed guardrail.")


# ═══════════════════════════════════════════════════════════════════════════════
# RED TIER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_red_tab(red_df: pd.DataFrame, campus: str):
    st.markdown('<div class="section-header">🔴 High-Risk Students — Parent Outreach</div>',
                unsafe_allow_html=True)

    if campus != "All":
        red_df = red_df[red_df["campus_id"] == campus].reset_index(drop=True)

    if len(red_df) == 0:
        st.success("No high-risk students for this campus. 🎉")
        return

    # ── Quota banner ──────────────────────────────────────────────────────
    if st.session_state["quota_hit"]:
        st.markdown(
            '<div class="quota-banner">⚠️ <b>Daily Gemini quota exhausted.</b> '
            'Review and approve already-generated drafts below. '
            'New generations available tomorrow.</div>',
            unsafe_allow_html=True,
        )

    # ── Batch controls ────────────────────────────────────────────────────
    cap          = st.session_state.get("batch_cap", 5)
    has_key      = bool(config.GEMINI_API_KEY)
    pending_sids = [r for r in red_df["student_id"] if r not in st.session_state["messages"]]
    pending_n    = len(pending_sids)
    draft_n      = len(red_df) - pending_n

    ctrl_col, info_col = st.columns([2, 3])
    with ctrl_col:
        if st.button(
            f"✉️  Generate Next {min(cap, pending_n)} Draft{'s' if cap>1 else ''}",
            disabled=(not has_key or pending_n == 0 or st.session_state["quota_hit"]),
            key="btn_batch",
        ):
            _generate_batch(red_df, cap)
            st.rerun()
    with info_col:
        st.markdown(
            f'<p style="color:#8b949e;font-size:12px;margin-top:10px">'
            f'{draft_n} / {len(red_df)} drafts ready &nbsp;·&nbsp; '
            f'{pending_n} pending &nbsp;·&nbsp; batch = {cap}</p>',
            unsafe_allow_html=True,
        )

    if not has_key:
        st.info("💡 Enter your Gemini API key in the sidebar to enable generation.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Per-student cards ─────────────────────────────────────────────────
    for _, row in red_df.iterrows():
        sid     = str(row["student_id"])
        name    = str(row.get("student_name", sid))
        campus_id = str(row.get("campus_id", ""))
        track   = str(row.get("learning_track", ""))
        reasons = str(row.get("risk_reasons", ""))
        att     = float(row.get("avg_minutes_per_session") or 0)
        pq      = float(row.get("avg_practice_per_day") or 0)
        misses  = int(row.get("recent_consec_misses") or 0)
        notes   = str(row.get("facilitator_notes_combined") or "")

        # Quiz score handling
        quiz_html = ""
        try:
            qs = float(row.get("last_quiz_score") or "nan")
            if not math.isnan(qs):
                quiz_html = _metric_chip("Quiz 1 Score", f"{qs:.0f} / 100", _quiz_class(qs))
        except (TypeError, ValueError):
            pass

        msg_data    = st.session_state["messages"].get(sid)
        has_message = msg_data is not None
        is_approved = st.session_state["approved"].get(sid, False)

        # Card border colour
        border_cls = "red-card" if not is_approved else "green-card"

        expander_title = (
            f"{'✅ ' if is_approved else '🔴 '}"
            f"**{name}** &nbsp; `{sid}` &nbsp;·&nbsp; {campus_id}"
            f"{' — ' + reasons[:70] + '…' if len(reasons) > 70 else (' — ' + reasons if reasons else '')}"
        )

        with st.expander(expander_title, expanded=has_message and not is_approved):

            # ── Metric chips row ──────────────────────────────────────────
            chips_html = (
                '<div class="metric-row">'
                + _metric_chip("Attendance", f"{att:.0f} min/sess", _att_class(att))
                + _metric_chip("Practice",   f"{pq:.1f} Q/day",    _pq_class(pq))
                + _metric_chip("Consec. Misses", str(misses),       _miss_class(misses))
                + quiz_html
                + f'<div class="metric-chip"><div class="metric-chip-label">Track</div>'
                  f'<div class="metric-chip-value" style="font-size:13px">{track}</div></div>'
                + '</div>'
            )
            st.markdown(chips_html, unsafe_allow_html=True)

            # Facilitator notes (collapsed)
            if notes.strip():
                with st.expander("📝 Facilitator notes", expanded=False):
                    st.markdown(
                        f'<div style="direction:rtl;text-align:right;color:#c9d1d9;'
                        f'font-size:13px;line-height:1.8">{notes}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<hr style='border-color:#21262d;margin:10px 0'>", unsafe_allow_html=True)

            # ── Message area ──────────────────────────────────────────────
            if not has_message:
                gen_col, _ = st.columns([1, 3])
                with gen_col:
                    if st.button(
                        "✉️  Generate Draft",
                        key=f"gen_{sid}",
                        disabled=(not has_key or st.session_state["quota_hit"]),
                    ):
                        with st.spinner(f"Generating for {name} …"):
                            _generate_single(dict(row))
                        st.rerun()
                if not has_key:
                    st.caption("Add Gemini API key in sidebar to generate.")
            else:
                msg_text   = msg_data["message"]
                g_passed   = msg_data.get("guardrail_passed", True)
                g_reason   = msg_data.get("guardrail_reason", "")
                is_error   = msg_text.startswith("[") and "Error" in msg_text or "failed" in msg_text.lower()[:30]

                # Guardrail / error banners
                if is_error:
                    st.markdown(
                        f'<div class="guardrail-error">❌ <b>Generation error:</b> {g_reason}</div>',
                        unsafe_allow_html=True,
                    )
                elif not g_passed:
                    st.markdown(
                        f'<div class="guardrail-warn">⚠️ <b>Guardrail note:</b> {g_reason} — '
                        'Please review before approving.</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="arabic-label">WhatsApp Message (editable · Arabic RTL)</div>',
                    unsafe_allow_html=True,
                )
                edited = st.text_area(
                    label="msg_area",
                    value=msg_text,
                    height=160,
                    key=f"msg_{sid}",
                    label_visibility="collapsed",
                )
                st.session_state["messages"][sid]["message"] = edited

                # Approve checkbox + regen
                ap_col, regen_col = st.columns([3, 1])
                with ap_col:
                    approved = st.checkbox(
                        "✅  Approve & queue for sending",
                        value=st.session_state["approved"].get(sid, False),
                        key=f"approve_{sid}",
                    )
                    st.session_state["approved"][sid] = approved
                with regen_col:
                    if st.button("🔄 Regenerate", key=f"regen_{sid}",
                                 disabled=(not has_key or st.session_state["quota_hit"])):
                        with st.spinner("Regenerating …"):
                            _generate_single(dict(row))
                        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Bulk approve + download ────────────────────────────────────────────
    approved_sids = [sid for sid, ok in st.session_state["approved"].items() if ok]
    bulk_col, dl_col = st.columns([2, 1])

    with bulk_col:
        if st.button(
            f"✅  Bulk-Approve All {len(st.session_state['messages'])} Generated Drafts",
            type="primary",
            disabled=len(st.session_state["messages"]) == 0,
            key="btn_bulk",
        ):
            for sid in st.session_state["messages"]:
                st.session_state["approved"][sid] = True
            st.success("All generated drafts approved. Use Download to export.")
            st.rerun()

    if approved_sids:
        records = [
            {
                "student_id":   sid,
                "student_name": st.session_state["messages"][sid]["student_name"],
                "campus_id":    str(red_df.loc[red_df["student_id"] == sid, "campus_id"].values[0])
                                if sid in red_df["student_id"].values else "",
                "message":      st.session_state["messages"][sid]["message"],
                "approved":     True,
            }
            for sid in approved_sids
            if sid in st.session_state["messages"]
        ]
        with dl_col:
            st.download_button(
                label=f"⬇️  Download {len(records)} Approved",
                data=json.dumps(records, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="approved_messages.json",
                mime="application/json",
                key="btn_dl",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# YELLOW TIER TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_yellow_tab(yellow_df: pd.DataFrame, campus: str):
    st.markdown('<div class="section-header">🟡 Medium-Risk Students — Automated Reminders</div>',
                unsafe_allow_html=True)
    st.caption(
        "Deterministic bilingual reminders generated instantly — no API call needed. "
        "Messages are ready to review and export below."
    )

    if campus != "All":
        yellow_df = yellow_df[yellow_df["campus_id"] == campus].reset_index(drop=True)

    if len(yellow_df) == 0:
        st.info("No medium-risk students for this campus.")
        return

    # ── Quick stats ───────────────────────────────────────────────────────
    st.markdown(
        f'<div style="color:#8b949e;font-size:12px;margin-bottom:10px">'
        f'{len(yellow_df)} students · messages pre-generated · no API quota used</div>',
        unsafe_allow_html=True,
    )

    # ── Per-student cards (lightweight, no LLM) ───────────────────────────
    for _, row in yellow_df.iterrows():
        sid    = str(row["student_id"])
        name   = str(row.get("student_name", sid))
        campus_id = str(row.get("campus_id", ""))
        att    = float(row.get("avg_minutes_per_session") or 0)
        pq     = float(row.get("avg_practice_per_day") or 0)
        misses = int(row.get("recent_consec_misses") or 0)
        msg    = str(row.get("message", ""))
        reasons = str(row.get("risk_reasons", ""))

        label = f"🟡 **{name}** `{sid}` · {campus_id}"

        with st.expander(label, expanded=False):
            chips = (
                '<div class="metric-row">'
                + _metric_chip("Attendance",    f"{att:.0f} min/sess", _att_class(att))
                + _metric_chip("Practice",      f"{pq:.1f} Q/day",     _pq_class(pq))
                + _metric_chip("Consec. Misses", str(misses),           _miss_class(misses))
                + '</div>'
            )
            st.markdown(chips, unsafe_allow_html=True)
            st.caption(f"Risk reasons: {reasons}")
            st.markdown('<div class="arabic-label">Automated Reminder (editable)</div>',
                        unsafe_allow_html=True)
            st.text_area(
                label="yellow_msg",
                value=msg,
                height=130,
                key=f"ymsg_{sid}",
                label_visibility="collapsed",
            )

    # ── Download yellow messages ───────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    export = yellow_df[["student_id", "student_name", "campus_id",
                         "avg_minutes_per_session", "avg_practice_per_day", "message"]].copy()
    export.columns = ["student_id", "student_name", "campus_id",
                      "avg_att_min", "avg_practice_qd", "message"]
    st.download_button(
        label=f"⬇️  Export {len(yellow_df)} Yellow Reminders (JSON)",
        data=json.dumps(export.to_dict(orient="records"), ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="yellow_reminders.json",
        mime="application/json",
        key="btn_dl_yellow",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════════════

def _render_overview_tab(tiered_df: pd.DataFrame, campus: str):
    st.markdown('<div class="section-header">📊 Campus Risk Breakdown</div>',
                unsafe_allow_html=True)

    if campus != "All":
        tiered_df = tiered_df[tiered_df["campus_id"] == campus].reset_index(drop=True)

    # Campus table
    campus_tbl = (
        tiered_df.groupby(["campus_id", "risk_tier"])
                 .size().unstack(fill_value=0).reset_index()
    )
    for col in [TIER_RED, TIER_YELLOW, TIER_GREEN]:
        if col not in campus_tbl.columns:
            campus_tbl[col] = 0
    st.dataframe(campus_tbl, use_container_width=True, hide_index=True)

    # Charts
    c_chart, c_table = st.columns([3, 1])

    with c_chart:
        st.markdown(
            '<p style="color:#8b949e;font-size:12px;text-transform:uppercase;'
            'letter-spacing:0.8px;margin-bottom:6px">Risk Distribution</p>',
            unsafe_allow_html=True,
        )
        tier_counts = (
            tiered_df["risk_tier"].value_counts()
                      .reindex([TIER_RED, TIER_YELLOW, TIER_GREEN], fill_value=0)
                      .reset_index()
        )
        tier_counts.columns = ["Tier", "Count"]

        # Use Streamlit native bar chart with teal color override via config
        chart_data = tier_counts.set_index("Tier")
        st.bar_chart(chart_data, color="#00ab77")

    with c_table:
        st.markdown(
            '<p style="color:#8b949e;font-size:12px;text-transform:uppercase;'
            'letter-spacing:0.8px;margin-bottom:6px">Counts</p>',
            unsafe_allow_html=True,
        )
        st.dataframe(tier_counts, use_container_width=True, hide_index=True)

    # Quiz score distribution
    if "last_quiz_score" in tiered_df.columns:
        scores = tiered_df["last_quiz_score"].dropna()
        if len(scores) > 0:
            st.markdown(
                '<p style="color:#8b949e;font-size:12px;text-transform:uppercase;'
                'letter-spacing:0.8px;margin:16px 0 6px 0">Quiz 1 Score Distribution</p>',
                unsafe_allow_html=True,
            )
            bins      = pd.cut(scores, bins=[0, 40, 50, 60, 70, 80, 100], right=False)
            score_dist = (
                bins.value_counts().sort_index()
                    .reset_index()
            )
            score_dist.columns = ["Score Band", "Count"]
            score_dist["Score Band"] = score_dist["Score Band"].astype(str)
            st.bar_chart(score_dist.set_index("Score Band"), color="#00ab77")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _init_state()

    # Page header
    st.markdown(
        '<div class="noon-title">🎓 Noon Academy <span>Intervention Dashboard</span></div>',
        unsafe_allow_html=True,
    )
    days_left = config.QUIZ2_DAY - config.CURRENT_DAY
    st.markdown(
        f'<div class="noon-subtitle">Day {config.CURRENT_DAY} &nbsp;·&nbsp; '
        f'Quiz 2 in <b style="color:#f5a623">{days_left} days</b> &nbsp;·&nbsp; '
        f'Intervention window closing fast.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)

    campus = _render_sidebar()

    # ── Upload + Run ──────────────────────────────────────────────────────
    paths = _render_upload_section()

    run_col, status_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("🚀  Run Analysis & Clean Data", type="primary")
    with status_col:
        if st.session_state["pipeline_ran"] and st.session_state["tiered_df"] is not None:
            counts = st.session_state["tiered_df"]["risk_tier"].value_counts()
            total  = len(st.session_state["tiered_df"])
            st.markdown(
                f'<div class="pipeline-success">'
                f'Pipeline complete &nbsp;·&nbsp; '
                f'🔴 {counts.get(TIER_RED,0)} Red &nbsp; '
                f'🟡 {counts.get(TIER_YELLOW,0)} Yellow &nbsp; '
                f'🟢 {counts.get(TIER_GREEN,0)} Green &nbsp; '
                f'({total} total students)</div>',
                unsafe_allow_html=True,
            )

    if run_clicked:
        _run_pipeline(paths)
        st.rerun()

    if not st.session_state["pipeline_ran"]:
        st.markdown(
            '<div style="color:#8b949e;font-size:13px;margin-top:20px;padding:20px;'
            'background:#161b22;border:1px dashed #30363d;border-radius:8px;text-align:center">'
            '👆 Upload your CSVs (or use the default data folder) and click '
            '<b style="color:#00ab77">Run Analysis & Clean Data</b> to begin.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── KPI scoreboard ────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)

    # Apply campus filter for KPI
    kpi_df = st.session_state["tiered_df"]
    if campus != "All":
        kpi_df = kpi_df[kpi_df["campus_id"] == campus]
    _render_kpi(kpi_df)

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_red, tab_yellow, tab_overview = st.tabs([
        "🔴  High Risk (Parent Outreach)",
        "🟡  Medium Risk (Reminders)",
        "📊  Overview",
    ])

    with tab_red:
        _render_red_tab(st.session_state["red_df"], campus)

    with tab_yellow:
        _render_yellow_tab(st.session_state["yellow_df"], campus)

    with tab_overview:
        _render_overview_tab(st.session_state["tiered_df"], campus)


if __name__ == "__main__":
    main()