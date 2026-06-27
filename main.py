"""
main.py
End-to-end pipeline entry point.
Runs the full pipeline and writes all outputs to outputs/.

Usage:
    python main.py                  # Full run with LLM
    python main.py --no-llm         # Skip LLM (for fast testing)
    python main.py --campus C01     # Filter to one campus
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

import config
from data_pipeline import build_master_dataframe
from risk_engine import get_tier, TIER_RED, TIER_YELLOW, TIER_GREEN

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Deterministic intervention templates ────────────────────────────────────

YELLOW_TEMPLATE = (
    "Hi {name}! 👋 Just a friendly reminder — Quiz 2 is on Day 20 (6 days away). "
    "You've attended {att:.0f} min/session on average and completed {pq:.0f} practice "
    "questions per day. Keep it up, and aim for 90 min + 10 questions daily. "
    "You've got this! 💪"
)

GREEN_TEMPLATE = (
    "Great work, {name}! 🌟 You're on track with {att:.0f} min/session and "
    "{pq:.0f} practice questions/day. Stay consistent — Quiz 2 is in 6 days!"
)


def _render_deterministic_messages(df: pd.DataFrame, template: str) -> pd.DataFrame:
    out = df.copy()
    out["message"] = out.apply(
        lambda r: template.format(
            name=r.get("student_name", r["student_id"]),
            att=r.get("avg_minutes_per_session", 0),
            pq=r.get("avg_practice_per_day", 0),
        ),
        axis=1,
    )
    out["guardrail_passed"] = True
    out["guardrail_reason"] = "deterministic"
    return out


def _save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = config.OUTPUTS_DIR / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved %s (%d rows)", filename, len(df))
    return path


def _save_json(data, filename: str) -> Path:
    path = config.OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s", filename)
    return path


# ── Analysis helpers ────────────────────────────────────────────────────────

def _compute_summary(df_with_tiers: pd.DataFrame) -> dict:
    tiers = df_with_tiers["risk_tier"].value_counts().to_dict()
    total = len(df_with_tiers)
    return {
        "total_students": total,
        "red_count": tiers.get("Red", 0),
        "yellow_count": tiers.get("Yellow", 0),
        "green_count": tiers.get("Green", 0),
        "red_pct": round(100 * tiers.get("Red", 0) / max(total, 1), 1),
        "avg_attendance_red": round(
            df_with_tiers[df_with_tiers["risk_tier"] == "Red"]["avg_minutes_per_session"].mean(), 1
        ),
        "avg_practice_red": round(
            df_with_tiers[df_with_tiers["risk_tier"] == "Red"]["avg_practice_per_day"].mean(), 1
        ),
        "students_with_notes": int((df_with_tiers["facilitator_notes_combined"] != "").sum()),
        "current_day": config.CURRENT_DAY,
        "days_to_quiz2": config.QUIZ2_DAY - config.CURRENT_DAY,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def run(use_llm: bool = True, campus_filter: str | None = None, llm_batch_size: int = 50):
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load, clean, tier, and export tiered roster ───────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Loading data, tiering, and exporting CSVs")
    tiered = build_master_dataframe()

    if campus_filter:
        tiered = tiered[tiered["campus_id"] == campus_filter.upper()]
        logger.info("Filtered to campus '%s': %d students", campus_filter, len(tiered))

    # ── Step 2: Split by risk tier ───────────────────────────────────────
    logger.info("STEP 2: Splitting risk tiers")

    red = get_tier(tiered, TIER_RED)
    yellow = get_tier(tiered, TIER_YELLOW)
    green = get_tier(tiered, TIER_GREEN)

    # ── Step 3: Re-export if campus-filtered (full roster already on disk) ─
    logger.info("STEP 3: Saving tiered roster")
    if campus_filter:
        _save_csv(tiered, config.ALL_STUDENTS_TIERED_PATH.name)
        _save_csv(red, config.RED_TIER_STUDENTS_PATH.name)
        _save_csv(yellow, config.YELLOW_TIER_STUDENTS_PATH.name)
    else:
        logger.info("Tiered CSVs already written by build_master_dataframe().")

    # ── Step 4: Deterministic messages (Yellow + Green) ────────────────────
    logger.info("STEP 4: Generating deterministic reminder messages")
    yellow_with_msgs = _render_deterministic_messages(yellow, YELLOW_TEMPLATE)
    green_with_msgs = _render_deterministic_messages(green, GREEN_TEMPLATE)
    _save_csv(yellow_with_msgs[["student_id", "student_name", "campus_id", "facilitator_email",
                                 "risk_tier", "risk_reasons", "message"]], "yellow_messages.csv")

    # ── Step 5: LLM messages (Red tier) ───────────────────────────────────
    logger.info("STEP 5: Generating LLM parent outreach messages (Red tier)")
    llm_results = []
    if use_llm and len(red) > 0:
        from llm_service import batch_generate
        llm_results = batch_generate(red, max_students=llm_batch_size)
        _save_json(llm_results, "red_llm_messages.json")

        # Flat CSV version for the UI
        llm_df = pd.DataFrame(llm_results)
        red_enriched = red.merge(llm_df[["student_id", "message", "guardrail_passed", "guardrail_reason"]],
                                  on="student_id", how="left")
        _save_csv(red_enriched, "red_messages.csv")
    elif len(red) == 0:
        logger.info("No Red-tier students — skipping LLM step.")
        _save_csv(red.assign(message="", guardrail_passed=True, guardrail_reason="no red students"),
                  "red_messages.csv")
    else:
        logger.info("LLM disabled (--no-llm flag). Saving Red roster only.")
        _save_csv(red, "red_messages.csv")

    # ── Step 6: Summary stats ──────────────────────────────────────────────
    logger.info("STEP 6: Computing summary")
    summary = _compute_summary(tiered)
    _save_json(summary, "run_summary.json")

    logger.info("=" * 60)
    logger.info("✅  Pipeline complete.")
    logger.info(
        "   Red: %d | Yellow: %d | Green: %d (of %d total)",
        summary["red_count"], summary["yellow_count"],
        summary["green_count"], summary["total_students"],
    )
    logger.info("   Outputs written to: %s", config.OUTPUTS_DIR)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noon Academy Intervention Pipeline")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM message generation")
    parser.add_argument("--campus", type=str, default=None, help="Filter to a specific campus ID")
    parser.add_argument("--batch-size", type=int, default=50, help="Max Red students to generate LLM messages for")
    args = parser.parse_args()

    run(use_llm=not args.no_llm, campus_filter=args.campus, llm_batch_size=args.batch_size)
