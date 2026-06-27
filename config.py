"""
config.py
All configuration via environment variables with documented defaults.
No hardcoded secrets or absolute paths.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional: pip install python-dotenv (listed in requirements.txt)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.getenv("NOON_BASE_DIR", Path(__file__).parent))
DATA_DIR = Path(os.getenv("NOON_DATA_DIR", BASE_DIR / "data"))
OUTPUTS_DIR = Path(os.getenv("NOON_OUTPUTS_DIR", BASE_DIR / "outputs"))

STUDENT_METADATA_PATH = Path(os.getenv("NOON_METADATA_CSV", DATA_DIR / "student_metadata.csv"))
STUDENT_METRICS_PATH  = Path(os.getenv("NOON_METRICS_CSV",  DATA_DIR / "student_daily_metrics.csv"))
FACILITATOR_NOTES_PATH = Path(os.getenv("NOON_NOTES_CSV",  DATA_DIR / "facilitator_notes.csv"))

ALL_STUDENTS_TIERED_PATH = Path(
    os.getenv("NOON_OUTPUT_ALL_TIERED", OUTPUTS_DIR / "all_students_tiered.csv")
)
RED_TIER_STUDENTS_PATH = Path(
    os.getenv("NOON_OUTPUT_RED_TIER", OUTPUTS_DIR / "red_tier_students.csv")
)
YELLOW_TIER_STUDENTS_PATH = Path(
    os.getenv("NOON_OUTPUT_YELLOW_TIER", OUTPUTS_DIR / "yellow_tier_students.csv")
)

# ── API Keys ────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
LLM_MODEL      = os.getenv("NOON_LLM_MODEL", "gemini-2.5-flash")

# ── Business-logic thresholds ───────────────────────────────────────────────
# Day context
CURRENT_DAY = int(os.getenv("NOON_CURRENT_DAY", "14"))
QUIZ1_DAY   = int(os.getenv("NOON_QUIZ1_DAY",   "10"))
QUIZ2_DAY   = int(os.getenv("NOON_QUIZ2_DAY",   "20"))

# Attendance (minutes per session; 90-min session)
ATTENDANCE_THRESHOLD_RED_MIN    = int(os.getenv("NOON_ATT_RED_MIN",    "45"))   # < 50 % of session
ATTENDANCE_THRESHOLD_YELLOW_MIN = int(os.getenv("NOON_ATT_YELLOW_MIN", "67"))   # < 75 %

# Practice questions per day
PRACTICE_RED_THRESHOLD    = int(os.getenv("NOON_PRACTICE_RED",    "5"))
PRACTICE_YELLOW_THRESHOLD = int(os.getenv("NOON_PRACTICE_YELLOW", "10"))

# Consecutive missed sessions before escalation
CONSECUTIVE_MISS_RED    = int(os.getenv("NOON_CONSEC_MISS_RED",    "3"))
CONSECUTIVE_MISS_YELLOW = int(os.getenv("NOON_CONSEC_MISS_YELLOW", "2"))

# Quiz / score thresholds
PASSING_GRADE             = int(os.getenv("NOON_PASSING_GRADE",    "60"))
HIGH_RISK_SCORE_THRESHOLD = int(os.getenv("NOON_HIGH_RISK_SCORE",  "50"))

# Post-quiz engagement drop that triggers Yellow flag (fraction, e.g. 0.4 = 40 % drop)
POST_QUIZ_DROP_THRESHOLD = float(os.getenv("NOON_POST_QUIZ_DROP", "0.40"))

# ── LLM generation settings ─────────────────────────────────────────────────
LLM_MAX_TOKENS  = int(os.getenv("NOON_LLM_MAX_TOKENS",  "400"))
LLM_TEMPERATURE = float(os.getenv("NOON_LLM_TEMPERATURE", "0.4"))

# ── Facilitator note healing ────────────────────────────────────────────────
# auto = names always + LLM grammar when GEMINI_API_KEY is set
# names = deterministic name swap only | llm = require API key | off = disabled
NOTE_HEALING_MODE = os.getenv("NOON_NOTE_HEALING_MODE", "auto").strip().lower()
NOTE_HEALING_LLM_MAX_TOKENS = int(os.getenv("NOON_NOTE_HEALING_MAX_TOKENS", "2048"))
NOTE_HEALING_CACHE_PATH = Path(
    os.getenv("NOON_NOTE_HEALING_CACHE", OUTPUTS_DIR / "note_healing_cache.json")
)

# ── App settings ────────────────────────────────────────────────────────────
APP_TITLE = "Noon Academy — Intervention Dashboard"
APP_PORT  = int(os.getenv("NOON_APP_PORT", "8501"))
