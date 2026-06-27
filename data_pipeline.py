"""
data_pipeline.py
════════════════
Single responsibility: raw ingestion, cleaning, aggregation, and master merge.

Pipeline stages
───────────────
1. load_student_metadata()    — parse and clean student_metadata.csv
2. load_daily_metrics()       — parse, clip, and window-filter metrics CSV
3. load_facilitator_notes()   — load + Mojibake-fix + Arabic name correction
                                + aggregate facilitator notes
4. _aggregate_metrics()       — overall stats + pre/post-quiz windows
5. build_master_dataframe()   — left-join spine: metadata ← metrics ← notes

Public API
──────────
    build_master_dataframe() → pd.DataFrame   (tiered, one row per student;
                                                also writes outputs/*.csv)
"""

import logging
import os
import re
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SHARED LOW-LEVEL UTILITIES
# ════════════════════════════════════════════════════════════════════════════

def _load_csv(
    path: Path,
    description: str,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load a CSV as all-string columns with a clear FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at '{path}'. "
            "Set the correct environment variable or place the file there."
        )
    df = pd.read_csv(path, dtype=str, encoding=encoding, encoding_errors="replace")
    logger.info(
        "Loaded %s: %d rows x %d cols from '%s'",
        description, *df.shape, path,
    )
    return df


def _fix_arabic_mojibake(text: str) -> str:
    """
    Reverse latin-1-as-utf-8 Mojibake corruption.
    Falls back silently when the text is already valid UTF-8.
    """
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def _looks_like_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]{2,}", text))


def _is_mojibake(text: str) -> bool:
    """Detect UTF-8 Arabic mis-read as Latin-1 (common in Excel exports)."""
    return bool(re.search(r"[ØÙÃÂ][\x80-\xbfØÙ]", text))


def _repair_text_encoding(text: str) -> str:
    """Return decoded Arabic text when the string is mojibake garbage."""
    if not text or not isinstance(text, str):
        return text or ""
    if _looks_like_arabic(text):
        return text
    if _is_mojibake(text):
        fixed = _fix_arabic_mojibake(text)
        if _looks_like_arabic(fixed):
            return fixed
    return text


# Excel on Windows opens CSV correctly when written with a UTF-8 BOM.
_CSV_ENCODING = "utf-8-sig"


def _parse_dates(series: pd.Series) -> pd.Series:
    """
    Robust date parser.
    Handles Excel '##' overflow artefacts (leading hashes) and mixed formats.
    Returns NaT for any value that cannot be parsed.
    """
    cleaned = series.astype(str).str.strip().str.replace(r"^#+", "", regex=True)
    return pd.to_datetime(cleaned, errors="coerce")


def _coerce_numeric(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convert a column to numeric.
    Logs a WARNING for non-parseable values and replaces them with NaN.
    """
    coerced = pd.to_numeric(series, errors="coerce")
    bad = coerced.isna() & series.notna() & (series.astype(str).str.strip() != "")
    if bad.sum():
        logger.warning(
            "%s: coerced %d non-numeric value(s) to NaN (samples: %s)",
            col_name, bad.sum(), series[bad].head(3).tolist(),
        )
    return coerced


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CSV LOADERS
# ════════════════════════════════════════════════════════════════════════════

def load_student_metadata() -> pd.DataFrame:
    """
    Load and clean student_metadata.csv.

    Cleaning applied
    ────────────────
    - Column names: lowercased, stripped.
    - student_id:   uppercased, stripped; duplicate rows dropped with warning.
    - target_score: coerced to float.
    - parent_phone: empty strings replaced with pd.NA.
    """
    df = _load_csv(config.STUDENT_METADATA_PATH, "student_metadata")
    df.columns       = df.columns.str.strip().str.lower()
    df["student_id"] = df["student_id"].str.strip().str.upper()

    before = len(df)
    df = df.drop_duplicates(subset=["student_id"])
    if len(df) < before:
        logger.warning(
            "Dropped %d duplicate student_id row(s) from metadata.", before - len(df)
        )

    df["target_score"] = _coerce_numeric(
        df.get("target_score", pd.Series(dtype=str)), "target_score"
    )
    if "parent_phone" in df.columns:
        df["parent_phone"] = df["parent_phone"].str.strip().replace(
            {"": pd.NA, "nan": pd.NA}
        )
    return df


def load_daily_metrics() -> pd.DataFrame:
    """
    Load and clean student_daily_metrics.csv.

    Cleaning applied
    ────────────────
    - Numeric columns coerced; impossible values clipped
      (attendance 0-120 min, practice >= 0).
    - Rows outside the study window (day 1 to CURRENT_DAY) are dropped.
    - last_quiz_score parsed when the column is present in the CSV.
    """
    df = _load_csv(config.STUDENT_METRICS_PATH, "student_daily_metrics")
    df.columns        = df.columns.str.strip().str.lower()
    df["student_id"]  = df["student_id"].str.strip().str.upper()
    df["date"]        = _parse_dates(df["date"])

    df["session_attended_min"] = _coerce_numeric(
        df["session_attended_min"], "session_attended_min"
    )
    df["practice_questions"] = _coerce_numeric(
        df["practice_questions"], "practice_questions"
    )
    if "last_quiz_score" in df.columns:
        df["last_quiz_score"] = _coerce_numeric(
            df["last_quiz_score"], "last_quiz_score"
        )

    df["session_attended_min"] = df["session_attended_min"].clip(lower=0, upper=120)
    df["practice_questions"]   = df["practice_questions"].clip(lower=0)

    if df["date"].notna().any():
        min_date = df["date"].min()
        cutoff   = min_date + pd.Timedelta(days=config.CURRENT_DAY - 1)
        before   = len(df)
        df       = df[df["date"] <= cutoff]
        logger.info(
            "Filtered metrics to study window: kept %d / %d rows.", len(df), before
        )
    return df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — METRIC AGGREGATION
# ════════════════════════════════════════════════════════════════════════════

def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily metric rows into one summary record per student.

    Produces
    ────────
    - Overall attendance and practice statistics.
    - Pre-quiz and post-quiz window averages (for risk engine drop detection).
    - Max Quiz-1 score per student (non-null rows only).
    - Consecutive missed sessions count over the most recent 5 days.
    """
    agg = metrics.groupby("student_id").agg(
        total_days_in_window     = ("date",                 "count"),
        days_attended            = ("session_attended_min", lambda x: (x > 0).sum()),
        total_minutes_attended   = ("session_attended_min", "sum"),
        avg_minutes_per_session  = ("session_attended_min", "mean"),
        total_practice_questions = ("practice_questions",   "sum"),
        avg_practice_per_day     = ("practice_questions",   "mean"),
        days_with_practice       = ("practice_questions",   lambda x: (x > 0).sum()),
    ).reset_index()

    agg["attendance_rate"] = (
        agg["days_attended"] / agg["total_days_in_window"].replace(0, pd.NA)
    )

    # Quiz score: max non-null per student
    if "last_quiz_score" in metrics.columns:
        quiz_agg = (
            metrics.dropna(subset=["last_quiz_score"])
                   .groupby("student_id")["last_quiz_score"]
                   .max()
                   .reset_index()
        )
        agg = agg.merge(quiz_agg, on="student_id", how="left")
    else:
        agg["last_quiz_score"] = pd.NA

    # Pre / post-Quiz-1 window averages
    if metrics["date"].notna().any():
        quiz1_date = metrics["date"].min() + pd.Timedelta(days=config.QUIZ1_DAY - 1)
        pre_quiz   = metrics[metrics["date"] <= quiz1_date]
        post_quiz  = metrics[metrics["date"] >  quiz1_date]

        def _window_avg(window: pd.DataFrame, col: str, suffix: str) -> pd.DataFrame:
            return (
                window.groupby("student_id")[col]
                      .mean()
                      .reset_index()
                      .rename(columns={col: f"{col}_{suffix}"})
            )

        for col in ("session_attended_min", "practice_questions"):
            agg = agg.merge(
                _window_avg(pre_quiz,  col, "pre_quiz"),  on="student_id", how="left"
            )
            agg = agg.merge(
                _window_avg(post_quiz, col, "post_quiz"), on="student_id", how="left"
            )

    # Consecutive missed sessions (last 5 days)
    def _max_consec_miss(group: pd.DataFrame, n: int = 5) -> int:
        recent      = group.sort_values("date").tail(n)
        miss_streak = max_streak = 0
        for val in recent["session_attended_min"]:
            if pd.isna(val) or val == 0:
                miss_streak += 1
                max_streak   = max(max_streak, miss_streak)
            else:
                miss_streak  = 0
        return max_streak

    streak_map = (
        metrics.groupby("student_id")
               .apply(_max_consec_miss)
               .reset_index()
               .rename(columns={0: "recent_consec_misses"})
    )
    agg = agg.merge(streak_map, on="student_id", how="left")
    agg["recent_consec_misses"] = agg["recent_consec_misses"].fillna(0).astype(int)

    return agg


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FACILITATOR NOTES (metadata-driven healing via note_healing.py)
# ════════════════════════════════════════════════════════════════════════════

from note_healing import (
    IdentityRegistry,
    gender_from_student_id,
    heal_combined_notes,
    heal_notes_dataframe,
)


def load_facilitator_notes(metadata_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load facilitator_notes.csv, repair encoding, heal note text, aggregate.

    Healing is two-phase (see note_healing.py):
      1. Row-level — swap wrong cohort names using trusted student_id (fast).
      2. Student-level — optional Gemini pass for Arabic gender agreement.

    Configure with NOON_NOTE_HEALING_MODE (auto | names | llm | off).
    """
    try:
        df = _load_csv(config.FACILITATOR_NOTES_PATH, "facilitator_notes", encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = _load_csv(config.FACILITATOR_NOTES_PATH, "facilitator_notes", encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning(
                "facilitator_notes.csv is not valid UTF-8 — "
                "re-reading as latin-1 and applying Mojibake recovery."
            )
            df = _load_csv(config.FACILITATOR_NOTES_PATH, "facilitator_notes", encoding="latin-1")

    df.columns       = df.columns.str.strip().str.lower()
    df["student_id"] = df["student_id"].str.strip().str.upper()
    df["date"]       = _parse_dates(df["date"])

    if "note_text" in df.columns:
        df["note_text"] = df["note_text"].astype(str).apply(_repair_text_encoding)
        mojibake_mask = df["note_text"].str.contains(r"Ø|Ù|Ã", na=False, regex=True)
        if mojibake_mask.sum():
            logger.warning(
                "Detected %d note(s) with Arabic Mojibake — applying repair.",
                mojibake_mask.sum(),
            )
            df.loc[mojibake_mask, "note_text"] = (
                df.loc[mojibake_mask, "note_text"].apply(_fix_arabic_mojibake)
            )

    registry = IdentityRegistry.from_metadata(metadata_df)
    n_corrected = 0

    if "note_text" in df.columns and config.NOTE_HEALING_MODE != "off":
        df = heal_notes_dataframe(df, registry)
        n_corrected = int((df["name_correction_status"] != "original").sum())

    if n_corrected:
        logger.info(
            "Name correction: fixed wrong Arabic name tokens in %d note row(s).",
            n_corrected,
        )

    notes_agg = (
        df.dropna(subset=["note_text"])
          .sort_values("date")
          .groupby("student_id")["note_text"]
          .apply(lambda texts: " | ".join(texts.astype(str).str.strip()))
          .reset_index()
          .rename(columns={"note_text": "facilitator_notes_combined"})
    )

    if config.NOTE_HEALING_MODE not in ("off", "names") and config.GEMINI_API_KEY:
        llm_statuses: list[str] = []
        for idx, row in notes_agg.iterrows():
            sid = str(row["student_id"])
            combined = str(row["facilitator_notes_combined"] or "")
            healed, status = heal_combined_notes(combined, sid, registry)
            notes_agg.at[idx, "facilitator_notes_combined"] = healed
            llm_statuses.append(status)
        logger.info(
            "LLM note grammar pass: %d student(s) processed.",
            len(llm_statuses),
        )

    if "name_correction_status" in df.columns:
        status_agg = (
            df.groupby("student_id")
              .agg(
                  notes_resolution_summary=(
                      "name_correction_status",
                      lambda s: ",".join(sorted(set(s))),
                  ),
              )
              .reset_index()
        )
    else:
        status_agg = df[["student_id"]].drop_duplicates().copy()
        status_agg["notes_resolution_summary"] = "original"

    status_agg["data_anomaly_detected"] = False

    return notes_agg.merge(status_agg, on="student_id", how="left")


def _export_tiered_outputs(tiered_df: pd.DataFrame) -> None:
    """Write tiered roster CSV artifacts — all three files from the same healed frame."""
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)

    from risk_engine import TIER_RED, TIER_YELLOW  # noqa: PLC0415

    red_df    = tiered_df[tiered_df["risk_tier"] == TIER_RED].copy()
    yellow_df = tiered_df[tiered_df["risk_tier"] == TIER_YELLOW].copy()

    tiered_df.to_csv(config.ALL_STUDENTS_TIERED_PATH, index=False, encoding=_CSV_ENCODING)
    red_df.to_csv(config.RED_TIER_STUDENTS_PATH, index=False, encoding=_CSV_ENCODING)
    yellow_df.to_csv(config.YELLOW_TIER_STUDENTS_PATH, index=False, encoding=_CSV_ENCODING)

    logger.info(
        "Exported tiered outputs: %s (%d rows), %s (%d rows), %s (%d rows)",
        config.ALL_STUDENTS_TIERED_PATH, len(tiered_df),
        config.RED_TIER_STUDENTS_PATH, len(red_df),
        config.YELLOW_TIER_STUDENTS_PATH, len(yellow_df),
    )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PUBLIC ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def build_master_dataframe() -> pd.DataFrame:
    """
    Orchestrate the full ingestion pipeline, assign risk tiers, export CSV
    artifacts, and return one tiered DataFrame per student.

    Merge spine (all left joins, metadata as the left spine):
        student_metadata  <-  metrics_aggregated  <-  healed_notes

    Disk artifacts (written automatically to config.OUTPUTS_DIR):
        config.ALL_STUDENTS_TIERED_PATH
        config.RED_TIER_STUDENTS_PATH
        config.YELLOW_TIER_STUDENTS_PATH

    Guarantees
    ──────────
    - Every student in metadata appears exactly once.
    - Facilitator note text is name-corrected at ingestion (student_id trusted).
    - data_anomaly_detected is always False after text correction.
    - All numeric columns zero-filled for students with no metric rows.
    - All string columns default to empty string or 'no_notes'.
    - student_gender ('أنثى' | 'ذكر') derived from registry Arabic first name.
    """
    # Stages 1 & 2: independent; could be parallelised in a future async layer
    metadata      = load_student_metadata()
    metrics_daily = load_daily_metrics()

    # Stage 3: load notes, correct wrong Arabic name tokens, aggregate
    notes = load_facilitator_notes(metadata)

    # Stage 4: metric aggregation
    metrics_agg = _aggregate_metrics(metrics_daily)

    # Stage 5: master merge
    master = (
        metadata
        .merge(metrics_agg, on="student_id", how="left")
        .merge(notes,       on="student_id", how="left")
    )

    # Zero-fill numeric count columns
    for col in [
        "total_days_in_window", "days_attended", "total_minutes_attended",
        "total_practice_questions", "days_with_practice", "recent_consec_misses",
    ]:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    for col in ("avg_minutes_per_session", "avg_practice_per_day", "attendance_rate"):
        master[col] = master[col].fillna(0)

    for suffix in ("pre_quiz", "post_quiz"):
        for base in ("session_attended_min", "practice_questions"):
            col = f"{base}_{suffix}"
            if col in master.columns:
                master[col] = master[col].fillna(0)

    # String / bool defaults
    master["facilitator_notes_combined"] = (
        master["facilitator_notes_combined"].fillna("")
    )
    master["notes_resolution_summary"] = (
        master["notes_resolution_summary"].fillna("no_notes")
    )
    master["data_anomaly_detected"] = (
        master["data_anomaly_detected"].fillna(False)
    )

    registry = IdentityRegistry.from_metadata(metadata)
    master["student_gender"] = master["student_id"].map(
        lambda sid: gender_from_student_id(str(sid).strip().upper(), registry)
    )

    # Diagnostic logging
    no_metrics = master["total_days_in_window"].eq(0).sum()
    if no_metrics:
        logger.warning("%d student(s) have no metric rows.", no_metrics)

    anomaly_count = int(master["data_anomaly_detected"].sum())
    if anomaly_count:
        logger.warning(
            "%d student(s) have unresolved note anomalies — "
            "check 'data_anomaly_detected' in the master DataFrame.",
            anomaly_count,
        )

    logger.info(
        "Master DataFrame built: %d students, %d columns.",
        len(master), master.shape[1],
    )

    from risk_engine import assign_risk_tiers  # noqa: PLC0415

    tiered = assign_risk_tiers(master)
    _export_tiered_outputs(tiered)

    logger.info(
        "Pipeline complete — tiered roster exported to %s (%d students).",
        config.OUTPUTS_DIR,
        len(tiered),
    )
    return tiered
