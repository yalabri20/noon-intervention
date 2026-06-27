"""
note_healing.py — Scalable facilitator-note correction.

Architecture (no hardcoded regex morphology lists)
──────────────────────────────────────────────────
1. IdentityRegistry  — built from student_metadata.csv + data/student_name_aliases.json
2. Name correction   — deterministic token swap (student_id is authoritative)
3. Grammar healing     — optional Gemini pass for gender agreement & colloquial fixes

Configure via config.NOTE_HEALING_MODE:
  auto  — names always; LLM grammar when GEMINI_API_KEY is set (default)
  names — deterministic names only (no API)
  llm   — names + LLM (raises if no API key)
  off   — skip healing
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

import config

logger = logging.getLogger(__name__)

_ARABIC_TOKEN = re.compile(r"[\u0600-\u06ff]{2,}")
_ALIASES_PATH = Path(__file__).parent / "data" / "student_name_aliases.json"


def _normalise_arabic(text: str) -> str:
    text = "".join(
        ch for ch in unicodedata.normalize("NFD", text)
        if unicodedata.category(ch) != "Mn"
    )
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"[ىئ]", "ي", text)
    return text.lower().strip()


def _load_aliases(path: Path | None = None) -> dict[str, Any]:
    path = path or _ALIASES_PATH
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class StudentIdentity:
    student_id: str
    english_root: str
    arabic_name: str
    gender: str  # أنثى | ذكر
    arabic_variants: frozenset[str] = field(default_factory=frozenset)


class IdentityRegistry:
    """Cohort identity lookup — rebuilt whenever metadata changes."""

    def __init__(self, aliases: dict[str, Any], identities: dict[str, StudentIdentity]) -> None:
        self._aliases = aliases
        self._by_id = identities
        self._non_names: frozenset[str] = frozenset(aliases.get("non_name_tokens", []))
        self._legacy_imposters: frozenset[str] = frozenset(
            aliases.get("legacy_imposter_names", [])
        )
        self._variant_map: dict[str, str] = {}
        for canonical, variants in aliases.get("arabic_surface_variants", {}).items():
            for v in variants:
                self._variant_map[_normalise_arabic(v)] = canonical
        self._all_cohort_surfaces: frozenset[str] = frozenset()
        for ident in identities.values():
            surfaces = {ident.arabic_name}
            for v in ident.arabic_variants:
                surfaces.add(v)
            canon = aliases.get("arabic_surface_variants", {}).get(ident.arabic_name, [])
            surfaces.update(canon)
            self._all_cohort_surfaces |= surfaces

    @classmethod
    def from_metadata(
        cls,
        metadata_df: pd.DataFrame,
        aliases_path: Path | None = None,
    ) -> IdentityRegistry:
        aliases = _load_aliases(aliases_path)
        en_to_ar = aliases.get("english_to_arabic", {})
        gender_map = aliases.get("gender_by_english_root", {})
        variant_cfg = aliases.get("arabic_surface_variants", {})

        identities: dict[str, StudentIdentity] = {}
        for _, row in metadata_df.iterrows():
            sid = str(row.get("student_id", "") or "").strip().upper()
            if not sid:
                continue
            raw = str(row.get("student_name", "") or "").strip()
            root = re.split(r"[\s_]+", raw)[0].lower() if raw else ""
            arabic = en_to_ar.get(root, "")
            gender = gender_map.get(root, "")
            variants = frozenset(variant_cfg.get(arabic, [arabic]) if arabic else [])
            identities[sid] = StudentIdentity(
                student_id=sid,
                english_root=root,
                arabic_name=arabic,
                gender=gender,
                arabic_variants=variants,
            )
        return cls(aliases, identities)

    def get(self, student_id: str) -> StudentIdentity | None:
        return self._by_id.get(str(student_id or "").strip().upper())

    def gender_label(self, student_id: str) -> str:
        ident = self.get(student_id)
        return ident.gender if ident else ""

    def arabic_name(self, student_id: str) -> str:
        ident = self.get(student_id)
        return ident.arabic_name if ident else ""

    def _is_replaceable_name(self, token: str, true_norm: str) -> bool:
        if token in self._non_names:
            return False
        token_norm = _normalise_arabic(token)
        if token_norm == true_norm:
            return False
        if token in self._all_cohort_surfaces:
            return True
        if token in self._legacy_imposters:
            return True
        if token_norm in self._variant_map:
            mapped = _normalise_arabic(self._variant_map[token_norm])
            return mapped != true_norm
        for surface in self._all_cohort_surfaces:
            if _normalise_arabic(surface) == token_norm:
                return True
        return False

    def correct_names(self, text: str, student_id: str) -> tuple[str, bool]:
        """Swap wrong student-name tokens using trusted student_id."""
        ident = self.get(student_id)
        if not ident or not ident.arabic_name or not text:
            return text, False

        true_name = ident.arabic_name
        true_norm = _normalise_arabic(true_name)
        changed = False

        def _replacer(match: re.Match[str]) -> str:
            nonlocal changed
            token = match.group(0)
            if not self._is_replaceable_name(token, true_norm):
                if _normalise_arabic(token) == true_norm and token != true_name:
                    changed = True
                    return true_name
                return token
            changed = True
            return true_name

        fixed = _ARABIC_TOKEN.sub(_replacer, text)
        return fixed, changed


def _normalize_typos(text: str) -> str:
    text = re.sub(r"معاهاا+", "معاها", text)
    text = re.sub(r"وضعهاا+", "وضعها", text)
    text = re.sub(r"حضورهاا+", "حضورها", text)
    return text.replace("استراتتجية", "استراتيجية")


def _healing_mode() -> str:
    return (getattr(config, "NOTE_HEALING_MODE", "auto") or "auto").strip().lower()


def _use_llm_grammar() -> bool:
    mode = _healing_mode()
    if mode == "off" or mode == "names":
        return False
    if mode == "llm":
        if not config.GEMINI_API_KEY:
            raise EnvironmentError(
                "NOTE_HEALING_MODE=llm requires GEMINI_API_KEY."
            )
        return True
    # auto
    return bool(config.GEMINI_API_KEY)


def heal_note_text(
    text: str,
    student_id: str,
    registry: IdentityRegistry,
    *,
    use_llm: bool | None = None,
    names_only: bool = False,
    grammar_only: bool = False,
) -> tuple[str, str]:
    """
    Full note healing for one text blob.

    Returns (healed_text, status) where status is one of:
      original | names_corrected | llm_healed | names_and_llm
    """
    if _healing_mode() == "off" or not text.strip():
        return text, "original"

    ident = registry.get(student_id)
    if not ident or not ident.arabic_name:
        return text, "original"

    fixed = text
    names_changed = False
    if not grammar_only:
        fixed, names_changed = registry.correct_names(text, student_id)
        fixed = _normalize_typos(fixed)

    status = "names_corrected" if names_changed else "original"

    llm_on = _use_llm_grammar() if use_llm is None else use_llm
    if llm_on and not names_only:
        from llm_service import heal_facilitator_note  # noqa: PLC0415

        llm_out = heal_facilitator_note(
            note_text=fixed,
            student_id=student_id,
            arabic_name=ident.arabic_name,
            gender=ident.gender,
        )
        if llm_out and llm_out.strip() and llm_out.strip() != fixed.strip():
            fixed = llm_out.strip()
            status = "names_and_llm" if names_changed else "llm_healed"
        elif names_changed:
            status = "names_corrected"

    return fixed, status


def heal_notes_dataframe(
    notes_df: pd.DataFrame,
    registry: IdentityRegistry,
    text_col: str = "note_text",
    id_col: str = "student_id",
) -> pd.DataFrame:
    """Row-level deterministic name healing before aggregation (fast, no API)."""
    df = notes_df.copy()
    statuses: list[str] = []

    for idx, row in df.iterrows():
        sid = str(row.get(id_col, "") or "").strip().upper()
        raw = str(row.get(text_col, "") or "")
        healed, status = heal_note_text(raw, sid, registry, names_only=True)
        df.at[idx, text_col] = healed
        statuses.append(status)

    df["name_correction_status"] = statuses
    return df


def heal_combined_notes(
    combined_text: str,
    student_id: str,
    registry: IdentityRegistry,
) -> tuple[str, str]:
    """Post-aggregation grammar pass — one optional LLM call per student."""
    combined_text = _normalize_typos(combined_text)
    if not _use_llm_grammar():
        return combined_text, "names_corrected" if combined_text else "original"
    return heal_note_text(
        combined_text, student_id, registry, grammar_only=True
    )


def build_identity_lookup(metadata_df: pd.DataFrame) -> dict[str, str]:
    """student_id → canonical Arabic first name (for downstream columns)."""
    reg = IdentityRegistry.from_metadata(metadata_df)
    return {sid: ident.arabic_name for sid, ident in reg._by_id.items()}


def gender_from_student_id(student_id: str, registry: IdentityRegistry) -> str:
    return registry.gender_label(student_id)
