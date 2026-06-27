# Noon Academy — Intervention Dashboard

[Walkthrough Video Link](https://www.loom.com/share/dc218e45e957471d8240b7f14f324db1)

## Quick start (fresh clone)

```bash
git clone https://github.com/yalabri20/noon-intervention && cd noon_intervention && python run.py
```

`run.py` installs dependencies and launches Streamlit. Bundled CSVs live in `data/`; outputs go to `outputs/`.

## Configuration

Copy `.env.example` → `.env` and set `GEMINI_API_KEY` for Red-tier outreach and note grammar healing.

**Note healing** (`NOON_NOTE_HEALING_MODE`): `auto` (default — local name fix + Gemini grammar when keyed), `names` (no API), `llm` (grammar required), `off`. Cohort name maps: `data/student_name_aliases.json`.

## Demo flow

Click **Run Analysis & Clean Data** in the app. CLI alternative: `python main.py --no-llm`.
