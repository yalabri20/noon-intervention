"""
llm_service.py  —  Google Gemini backend for Red-tier parent outreach.

Root-cause fixes in this revision
───────────────────────────────────
1. GENERATION ERROR FIX: gemini-2.5-flash runs with internal "thinking"
   by default. When thinking tokens are present, iterating response parts
   naively can miss the actual text part. We now:
     a) Disable thinking via ThinkingConfig(thinking_budget=0) — faster,
        cheaper, and avoids the empty-text edge case entirely.
     b) Add a safe _extract_text() that walks response.candidates[0].content
        .parts directly, skipping thought=True parts, so we NEVER lose text.
     c) Log the full exception type + message on every API failure so the
        terminal always shows the real root cause.

2. GUARDRAIL LOOSENED: after all retries the last generated text is always
   returned (flagged for human review) instead of discarded. The UI shows
   a warning badge; the facilitator can still read and edit it.

3. QUOTA DETECTION: both RESOURCE_EXHAUSTED and per-minute 429 are handled.
   Quota → raises QuotaExhaustedError immediately (batch stops cleanly).
   Rate-limit → exponential backoff, then continues.
"""

import logging
import math
import re
import time
from typing import Optional

from google import genai
from google.genai import types as genai_types

import config

logger = logging.getLogger(__name__)

_client: Optional[genai.Client] = None


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise EnvironmentError(
                "GEMINI_API_KEY is not set. "
                "Export it before running: export GEMINI_API_KEY=<your-key>"
            )
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


# ── Safe text extraction ──────────────────────────────────────────────────────

def _extract_text(response) -> str:
    """
    Safely extract the text content from a Gemini response.

    gemini-2.5-flash may return multiple parts including 'thought' parts
    (internal chain-of-thought). We skip those and concatenate only the
    real output text parts.  Falls back to response.text if the manual
    walk fails, and returns "" rather than raising.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            # Safety block or empty response — check prompt_feedback
            pf = getattr(response, "prompt_feedback", None)
            if pf:
                block_reason = getattr(pf, "block_reason", None)
                if block_reason:
                    logger.warning("Gemini blocked response: %s", block_reason)
            return ""

        cand    = candidates[0]
        content = getattr(cand, "content", None)
        if not content:
            finish = getattr(cand, "finish_reason", None)
            logger.warning("Empty candidate content; finish_reason=%s", finish)
            return ""

        parts = getattr(content, "parts", None) or []
        text_parts = []
        for part in parts:
            # Skip internal thought parts (present in thinking models)
            if getattr(part, "thought", False):
                continue
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                text_parts.append(t)

        if text_parts:
            return "".join(text_parts).strip()

        # Fallback: use the SDK's .text property (returns None if only thoughts)
        sdk_text = getattr(response, "text", None)
        return (sdk_text or "").strip()

    except Exception as exc:
        logger.error("_extract_text failed: %s — %s", type(exc).__name__, exc)
        return ""


# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_prompt(student: dict) -> str:
    name   = student.get("student_name") or student.get("student_id", "الطالب")
    gender = (student.get("student_gender") or "").strip() or "غير محدد"
    notes  = (student.get("facilitator_notes_combined") or "").strip()
    notes_section = (
        f"ملاحظات المشرف / Facilitator notes: {notes}"
        if notes
        else "لا توجد ملاحظات / No facilitator notes recorded."
    )

    quiz_section = ""
    try:
        qs = float(student.get("last_quiz_score") or "nan")
        if not math.isnan(qs):
            status = (
                f"below passing ({config.PASSING_GRADE})"
                if qs < config.PASSING_GRADE else "passed"
            )
            quiz_section = f"- Quiz 1 result: {qs:.0f} / 100 ({status})\n"
    except (TypeError, ValueError):
        pass

    return f"""أنت متخصص تدخل تعليمي في أكاديمية نون، المملكة العربية السعودية.
You are an empathetic EdTech intervention specialist at Noon Academy, Saudi Arabia.

A student is HIGH RISK before Quiz 2 (in 6 days). Write a WhatsApp message for the PARENT.

STUDENT DATA — use ONLY these facts, never invent numbers:
- Student name: {name}
- Student gender: {gender}
- Campus: {student.get('campus_id', 'N/A')}
- Learning track: {student.get('learning_track', 'N/A')}
- Avg session attendance: {float(student.get('avg_minutes_per_session') or 0):.0f} min (target: 90 min)
- Avg daily practice questions: {float(student.get('avg_practice_per_day') or 0):.1f} (target: 10+)
- Recent consecutive misses: {student.get('recent_consec_misses', 0)}
{quiz_section}- Risk summary: {student.get('risk_reasons', 'See notes')}
- {notes_section}

⚠️ IMPORTANT GRAMMAR CONSTRAINT:
The student's name inside the text logs was programmatically corrected to their true identity from the registry database, including verb/pronoun gender agreement where the facilitator had typed a wrong name of the opposite gender. You are provided with the student's correct gender: [{gender}].
When drafting the outreach message to the family, you MUST carefully review the context of the facilitator note. Ensure perfect Arabic grammatical agreement across all verbs, pronouns, and family adjectives (e.g., if the gender is 'ذكر' (Male), ensure verbs use male forms like 'يحضر' or 'يتمرن' instead of female forms, and refer to 'أبو الطالب' appropriately, and vice versa). If any residual mismatch remains in the notes, correct it in your message — do not echo wrong gender forms.

STRICT RULES:
1. Open with: السلام عليكم ورحمة الله وبركاته
2. Mention the student's name ({name}) once — use Arabic spelling if the name has one.
3. State ONE specific concern (pick the worst metric above).
4. Suggest ONE concrete action, e.g. يرجى التواصل مع المشرف / please contact the facilitator.
5. Close with a warm encouraging sentence in Arabic and English.
6. Maximum 3 sentences, under 100 words total.
7. Do NOT invent any number, score, or fact not listed above.

Output ONLY the WhatsApp message — no headers, no markdown, no labels."""


# ── Guardrail ─────────────────────────────────────────────────────────────────

def _is_placeholder_name(name: str) -> bool:
    low = name.lower()
    return (
        low.startswith("student_")
        or low.startswith("stu0")
        or low.startswith("stu ")
    )


def _guardrail_check(message: str, student: dict) -> tuple[bool, str]:
    """
    Soft validation. Returns (passed, reason).
    Designed to catch obvious hallucinations without over-rejecting valid Arabic.
    """
    name = (student.get("student_name") or "").strip()

    # 1. Name / personalisation check (language-agnostic)
    if name and not _is_placeholder_name(name):
        first      = name.replace("_", " ").split()[0]
        has_arabic = any("\u0600" <= ch <= "\u06ff" for ch in message)
        # Pass if Arabic chars present (model used Arabic name) OR English name found
        if not has_arabic and first.lower() not in message.lower():
            return False, f"Generic message — '{first}' not found and no Arabic script"

    # 2. No false praise for low-attendance students
    avg_att = float(student.get("avg_minutes_per_session") or 0)
    if avg_att < config.ATTENDANCE_THRESHOLD_RED_MIN:
        for phrase in ["excellent attendance", "perfect attendance",
                       "always present", "حضور ممتاز", "منتظم في الحضور"]:
            if phrase in message.lower():
                return False, "Hallucinated false attendance praise"

    # 3. No false pass claim for failed quiz
    try:
        qs = float(student.get("last_quiz_score") or "nan")
        if not math.isnan(qs) and qs < config.PASSING_GRADE:
            for phrase in ["passed the quiz", "great quiz",
                           "نجح في الاختبار", "اجتاز الاختبار"]:
                if phrase in message.lower():
                    return False, "Hallucinated false quiz-pass claim"
    except (TypeError, ValueError):
        pass

    # 4. Length sanity
    word_count = len(message.split())
    if word_count < 12:
        return False, f"Too short ({word_count} words) — likely refusal or truncation"
    if word_count > 220:
        return False, f"Too long ({word_count} words) — exceeds usability limit"

    # 5. Action indicator (Arabic or English)
    action_indicators = [
        "call", "contact", "schedule", "please", "reach out", "speak",
        "اتصال", "تواصل", "موعد", "يرجى", "اتصل", "تفضل", "نرجو",
        "تواصلوا", "تواصلي", "ارجو", "أرجو", "التواصل", "المشرف",
    ]
    if not any(ind in message.lower() for ind in action_indicators):
        return False, "No action/next-step indicator found"

    return True, "OK"


# ── Core generation ───────────────────────────────────────────────────────────

class QuotaExhaustedError(Exception):
    """Raised when the Gemini daily free-tier quota is confirmed exhausted."""


def generate_outreach_message(student: dict, retries: int = 1) -> dict:
    """
    Generate a WhatsApp outreach message for ONE Red-tier student.

    Always returns a dict — never raises (except QuotaExhaustedError).
    On failure the 'message' field contains the actual exception text
    so facilitators see a meaningful error instead of a generic string.

    Raises:
        QuotaExhaustedError — caller must stop the batch immediately.
    """
    student_id   = student.get("student_id", "UNKNOWN")
    student_name = student.get("student_name") or student_id
    result_base  = {"student_id": student_id, "student_name": student_name}
    prompt       = _build_prompt(student)

    last_message: str = ""
    last_reason:  str = "No attempts completed"

    for attempt in range(1, retries + 2):
        try:
            client   = _get_client()
            response = client.models.generate_content(
                model=config.LLM_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=config.LLM_TEMPERATURE,
                    max_output_tokens=config.LLM_MAX_TOKENS,
                    # Disable thinking mode: faster, cheaper, avoids empty-text bug
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_budget=0,
                    ),
                ),
            )

            # Use safe extractor instead of bare response.text
            message_text = _extract_text(response)
            last_message = message_text

            if not message_text:
                # Log finish_reason for diagnostics
                try:
                    fr = response.candidates[0].finish_reason if response.candidates else "NO_CANDIDATES"
                    last_reason = f"Empty response (finish_reason={fr})"
                except Exception:
                    last_reason = "Empty response (finish_reason unknown)"
                logger.warning("%s attempt %d: %s", student_id, attempt, last_reason)
                time.sleep(1)
                continue

            passed, reason = _guardrail_check(message_text, student)
            last_reason    = reason

            if passed:
                logger.info("✅ %s (%s): message OK.", student_id, student_name)
                return {**result_base, "message": message_text,
                        "guardrail_passed": True, "guardrail_reason": "OK"}

            logger.warning("⚠️ %s attempt %d guardrail: %s", student_id, attempt, reason)
            if attempt <= retries:
                time.sleep(1)
                continue

            # All retries done — return LAST text flagged for human review.
            # Do NOT discard: text is usually fine; guardrail may be overly strict.
            logger.info("%s: returning guardrail-flagged text for human review.", student_id)
            return {**result_base, "message": last_message,
                    "guardrail_passed": False, "guardrail_reason": last_reason}

        except Exception as exc:
            exc_type = type(exc).__name__
            err_str  = str(exc)

            # Always log the FULL root cause so terminal shows real error
            logger.error(
                "%s attempt %d — %s: %s", student_id, attempt, exc_type, err_str
            )

            is_quota = (
                "RESOURCE_EXHAUSTED" in err_str
                or ("429" in err_str and "quota" in err_str.lower())
            )
            is_rate = "429" in err_str and not is_quota

            if is_quota:
                raise QuotaExhaustedError(err_str) from exc

            if is_rate:
                wait = min(30, 10 * attempt)
                logger.warning("%s: rate-limit — sleeping %ds", student_id, wait)
                time.sleep(wait)
                continue

            # Unknown error — store diagnostic text, retry if attempts remain
            last_message = f"[{exc_type}] {err_str[:200]}"
            last_reason  = f"{exc_type}: {err_str[:120]}"
            time.sleep(2)

    # Exhausted all attempts — return best-effort with full error visible to facilitator
    return {
        **result_base,
        "message": last_message or f"[Generation failed after {retries + 1} attempt(s). Check terminal logs.]",
        "guardrail_passed": False,
        "guardrail_reason": last_reason,
    }


def batch_generate(
    red_df,
    max_students: int = 15,
    progress_callback=None,
) -> tuple[list[dict], bool]:
    """
    Generate messages for up to `max_students` Red-tier students.
    Returns (results_list, quota_hit_bool).
    Partial results are always returned even if quota is hit mid-batch.
    """
    students: list[dict] = red_df.head(max_students).to_dict(orient="records")
    results:  list[dict] = []
    total = len(students)

    for i, student in enumerate(students, 1):
        name = student.get("student_name", student.get("student_id"))
        logger.info("Generating %d/%d — %s …", i, total, name)

        if progress_callback:
            progress_callback(i, total, name)

        try:
            results.append(generate_outreach_message(student))
        except QuotaExhaustedError:
            logger.error("Quota exhausted after %d/%d messages.", len(results), total)
            return results, True

    passed = sum(1 for r in results if r["guardrail_passed"])
    logger.info("Batch done: %d/%d passed guardrail.", passed, len(results))
    return results, False


# ── Facilitator note grammar healing ────────────────────────────────────────

def _build_note_healing_prompt(
    note_text: str,
    student_id: str,
    arabic_name: str,
    gender: str,
) -> str:
    gender_hint = "مذكر (he/him)" if gender == "ذكر" else "أنثى (she/her)"
    return f"""أنت محرر نصوص عربية لملاحظات المعلمين في برنامج تعليمي.

student_id الموثوق: {student_id}
اسم الطالب الصحيح: {arabic_name}
جنس الطالب: {gender_hint}

المشكلة: المعلم كتب الملاحظة أحياناً باسم طالب آخر أو بصيغة جنس خاطئة (أفعال/ضمائر).

المطلوب:
1. صحّح أي اسم طالب خاطئ إلى «{arabic_name}» فقط عندما يُقصد الطالب.
2. صحّح أفعال وضمائر وصفات الطالب لتطابق جنسه ({gender_hint}).
3. لا تغيّر كلام الأم/الأب (مثل: ام {arabic_name} قالت …).
4. لا تغيّر كلام المعلم عن نفسه (اتصلت، سألت، …).
5. لا تضف معلومات جديدة ولا تحذف معنى.
6. احتفظ بفاصل « | » بين الملاحظات إن وُجد.
7. أرجع النص العربي المصحح فقط — بدون شرح أو markdown.

النص:
{note_text}
"""


def heal_facilitator_note(
    note_text: str,
    student_id: str,
    arabic_name: str,
    gender: str,
) -> str:
    """
    Use Gemini to fix gender agreement and residual wrong names in one note blob.
    Returns original text on failure (never raises).
    """
    if not note_text.strip() or not config.GEMINI_API_KEY:
        return note_text

    prompt = _build_note_healing_prompt(note_text, student_id, arabic_name, gender)
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=config.NOTE_HEALING_LLM_MAX_TOKENS,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )
        healed = _extract_text(response).strip()
        if not healed or len(healed) < max(8, len(note_text) // 4):
            logger.warning("%s note healing: empty/short LLM output — keeping original.", student_id)
            return note_text
        # Strip accidental markdown fences
        healed = re.sub(r"^```(?:arabic|text)?\s*", "", healed)
        healed = re.sub(r"\s*```$", "", healed).strip()
        return healed
    except Exception as exc:
        logger.warning("%s note healing failed (%s): %s", student_id, type(exc).__name__, exc)
        return note_text