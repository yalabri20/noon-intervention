"""
risk_engine.py
Assigns each student a risk tier (Red / Yellow / Green) using
vectorised Pandas operations on the aggregated master DataFrame.

Changes in this revision
─────────────────────────
• Quiz-score integration: students who failed Quiz 1 (score < PASSING_GRADE)
  AND show engagement problems are always Red, regardless of attendance alone.
• Post-quiz momentum drop: a >40 % fall in attendance or practice after
  Quiz 1 (vs. pre-quiz baseline) escalates a passing student to Yellow.
• risk_score now includes a quiz-failure bonus (+3) so the most
  vulnerable students always surface at the top of the facilitator queue.
• Expanded Arabic + English keyword scan for facilitator notes, now applied
  to properly decoded UTF-8 text from the updated pipeline.

Design principle: deterministic rules first, LLM only for message generation.
"""

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

TIER_RED    = "Red"
TIER_YELLOW = "Yellow"
TIER_GREEN  = "Green"


# ── Sub-scorers (each returns a numeric pd.Series, higher = worse) ──────────

def _score_attendance(df: pd.DataFrame) -> pd.Series:
    """0–3 points from attendance metrics."""
    score = pd.Series(0, index=df.index, dtype=int)
    score += (df["avg_minutes_per_session"] < config.ATTENDANCE_THRESHOLD_YELLOW_MIN).astype(int)
    score += (df["avg_minutes_per_session"] < config.ATTENDANCE_THRESHOLD_RED_MIN).astype(int)
    score += (df["recent_consec_misses"] >= config.CONSECUTIVE_MISS_YELLOW).astype(int)
    return score


def _score_practice(df: pd.DataFrame) -> pd.Series:
    """0–2 points from practice engagement."""
    score = pd.Series(0, index=df.index, dtype=int)
    score += (df["avg_practice_per_day"] < config.PRACTICE_YELLOW_THRESHOLD).astype(int)
    score += (df["avg_practice_per_day"] < config.PRACTICE_RED_THRESHOLD).astype(int)
    return score


def _score_quiz(df: pd.DataFrame) -> pd.Series:
    """
    0–3 points from Quiz 1 performance.
    +3 for outright failure (score < PASSING_GRADE).
    +1 for borderline pass (score < PASSING_GRADE + 10) combined with
       other engagement problems — caught downstream via total score.
    Students with no quiz score recorded receive 0 (benefit of the doubt).
    """
    score = pd.Series(0, index=df.index, dtype=int)
    if "last_quiz_score" not in df.columns:
        return score

    quiz_known = df["last_quiz_score"].notna()
    failed     = quiz_known & (df["last_quiz_score"] < config.PASSING_GRADE)
    borderline = quiz_known & (
        (df["last_quiz_score"] >= config.PASSING_GRADE) &
        (df["last_quiz_score"] <  config.PASSING_GRADE + 10)
    )

    score += (failed * 3).astype(int)
    score += (borderline * 1).astype(int)
    return score


def _score_post_quiz_drop(df: pd.DataFrame) -> pd.Series:
    """
    +1 point per metric (attendance, practice) that dropped >40 % after Quiz 1.
    Only meaningful when both pre- and post-quiz window data exist.
    """
    score = pd.Series(0, index=df.index, dtype=int)
    threshold = config.POST_QUIZ_DROP_THRESHOLD

    for base_col in ("session_attended_min", "practice_questions"):
        pre_col  = f"{base_col}_pre_quiz"
        post_col = f"{base_col}_post_quiz"
        if pre_col not in df.columns or post_col not in df.columns:
            continue

        pre  = df[pre_col].fillna(0)
        post = df[post_col].fillna(0)

        # Avoid division by zero; if pre-quiz baseline was already 0, no drop to measure
        meaningful = pre > 0
        dropped    = meaningful & ((pre - post) / pre > threshold)
        score     += dropped.astype(int)

    return score


def _has_notes_concern(df: pd.DataFrame) -> pd.Series:
    """
    Binary flag (0/1): facilitator notes contain a concern keyword.
    Covers both English and Arabic terms (post Mojibake fix, Arabic is readable).
    """
    concern_keywords = [
        # English
        "miss", "absent", "distract", "overwhelm", "anxious",
        "struggle", "behind", "unresponsive", "family issue", "tired",
        "no show", "not attending", "worried", "falling behind",
        # Arabic transliterations / common terms
        "غياب", "تأخر", "قلق", "صعوبة", "لا يحضر", "متأخر",
    ]
    pattern = "|".join(concern_keywords)
    return (
        df["facilitator_notes_combined"]
          .str.lower()
          .str.contains(pattern, na=False, regex=True)
          .astype(int)
    )


# ── Main tiering function ───────────────────────────────────────────────────

def assign_risk_tiers(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `risk_tier`, `risk_score`, and `risk_reasons` columns to master_df.

    Tier logic
    ──────────
    RED   — any of:
              • avg attendance < RED threshold
              • ≥ CONSECUTIVE_MISS_RED sessions missed recently
              • avg practice < RED threshold
              • Quiz 1 failed (score < PASSING_GRADE)   ← NEW
              • Quiz failed AND post-quiz engagement dropped >40 %  ← NEW

    YELLOW — not Red, but any of:
              • avg attendance < YELLOW threshold
              • ≥ CONSECUTIVE_MISS_YELLOW sessions missed recently
              • avg practice < YELLOW threshold
              • Post-quiz attendance OR practice dropped >40 %      ← NEW
              • Facilitator notes flag a concern

    GREEN  — everything else.

    risk_score drives the sort order within each tier (higher = more urgent).
    """
    df = master_df.copy()

    att_score       = _score_attendance(df)
    prac_score      = _score_practice(df)
    quiz_score      = _score_quiz(df)
    drop_score      = _score_post_quiz_drop(df)
    note_concern    = _has_notes_concern(df)

    df["risk_score"] = att_score + prac_score + quiz_score + drop_score + note_concern

    # ── Boolean building blocks ────────────────────────────────────────────
    quiz_failed = (
        df["last_quiz_score"].notna() & (df["last_quiz_score"] < config.PASSING_GRADE)
        if "last_quiz_score" in df.columns
        else pd.Series(False, index=df.index)
    )

    post_quiz_dropped = drop_score > 0   # at least one metric fell >40 %

    red_mask = (
        (df["avg_minutes_per_session"] < config.ATTENDANCE_THRESHOLD_RED_MIN)
        | (df["recent_consec_misses"]  >= config.CONSECUTIVE_MISS_RED)
        | (df["avg_practice_per_day"]  < config.PRACTICE_RED_THRESHOLD)
        | quiz_failed                                           # NEW: quiz failure → always Red
        | (quiz_failed & post_quiz_dropped)                    # NEW: belt-and-braces
    )

    yellow_mask = (
        ~red_mask
        & (
            (df["avg_minutes_per_session"] < config.ATTENDANCE_THRESHOLD_YELLOW_MIN)
            | (df["recent_consec_misses"]  >= config.CONSECUTIVE_MISS_YELLOW)
            | (df["avg_practice_per_day"]  < config.PRACTICE_YELLOW_THRESHOLD)
            | post_quiz_dropped                                 # NEW: post-quiz drop
            | note_concern.astype(bool)                        # note concern escalates to Yellow
        )
    )

    df["risk_tier"]                  = TIER_GREEN
    df.loc[yellow_mask, "risk_tier"] = TIER_YELLOW
    df.loc[red_mask,    "risk_tier"] = TIER_RED

    # ── Human-readable reason strings ──────────────────────────────────────
    def _build_reasons(row: pd.Series) -> str:
        reasons: list[str] = []

        # Attendance
        att = row["avg_minutes_per_session"]
        if att < config.ATTENDANCE_THRESHOLD_RED_MIN:
            reasons.append(f"Low attendance (avg {att:.0f} min/session)")
        elif att < config.ATTENDANCE_THRESHOLD_YELLOW_MIN:
            reasons.append(f"Borderline attendance (avg {att:.0f} min/session)")

        # Consecutive misses
        misses = row["recent_consec_misses"]
        if misses >= config.CONSECUTIVE_MISS_RED:
            reasons.append(f"Missed {misses} consecutive sessions")
        elif misses >= config.CONSECUTIVE_MISS_YELLOW:
            reasons.append(f"{misses} consecutive session misses")

        # Practice
        pq = row["avg_practice_per_day"]
        if pq < config.PRACTICE_RED_THRESHOLD:
            reasons.append(f"Very low practice (avg {pq:.1f} Q/day)")
        elif pq < config.PRACTICE_YELLOW_THRESHOLD:
            reasons.append(f"Below-target practice (avg {pq:.1f} Q/day)")

        # Quiz score
        if "last_quiz_score" in row.index and pd.notna(row["last_quiz_score"]):
            score = row["last_quiz_score"]
            if score < config.PASSING_GRADE:
                reasons.append(f"Failed Quiz 1 (score {score:.0f} / {config.PASSING_GRADE})")
            elif score < config.PASSING_GRADE + 10:
                reasons.append(f"Borderline Quiz 1 pass (score {score:.0f})")

        # Post-quiz drop
        threshold = config.POST_QUIZ_DROP_THRESHOLD
        for base_col, label in (
            ("session_attended_min", "attendance"),
            ("practice_questions",   "practice"),
        ):
            pre_col  = f"{base_col}_pre_quiz"
            post_col = f"{base_col}_post_quiz"
            if pre_col in row.index and post_col in row.index:
                pre  = row[pre_col]
                post = row[post_col]
                if pre > 0 and (pre - post) / pre > threshold:
                    drop_pct = 100 * (pre - post) / pre
                    reasons.append(
                        f"Post-quiz {label} dropped {drop_pct:.0f} % "
                        f"({pre:.0f} → {post:.0f})"
                    )

        if not reasons:
            reasons.append("On track")
        return "; ".join(reasons)

    df["risk_reasons"] = df.apply(_build_reasons, axis=1)

    # ── Summary log ────────────────────────────────────────────────────────
    counts = df["risk_tier"].value_counts()
    logger.info(
        "Risk tiering complete — Red: %d | Yellow: %d | Green: %d",
        counts.get(TIER_RED,    0),
        counts.get(TIER_YELLOW, 0),
        counts.get(TIER_GREEN,  0),
    )

    return df


def get_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """Return the subset of students in `tier`, sorted by risk_score descending."""
    return (
        df[df["risk_tier"] == tier]
          .sort_values("risk_score", ascending=False)
          .reset_index(drop=True)
    )
