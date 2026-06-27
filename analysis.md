# Noon Academy Intervention — Engineering Memo

## 1. Diagnosis

The intervention rate is stuck at 30% because facilitators face alert fatigue across a flat, undifferentiated student list with no reliable priority signal. Without a deterministic risk framework, high-urgency cases are buried alongside on-track students, so outreach effort spreads thin and meaningful contact never scales. Compounding this, unstructured Arabic facilitator notes contained systematic name typos that misaligned text context with trusted `student_id` records, degrading both facilitator trust and downstream message quality.

## 2. What you found in the data

- The master pipeline ingests and merges **200 students** across **5 campuses** (`C01`–`C05`) from three source CSVs into a single tiered roster exported to `outputs/all_students_tiered.csv`.
- Risk classification assigns **73 Red**, **33 Yellow**, and **94 Green** students using fixed thresholds: attendance **<45 min** (Red) / **<67 min** (Yellow), practice **<5** / **<10** questions per day, Quiz 1 failure **<60**, and post-quiz engagement drops **>40%**.
- Facilitator note healing corrected wrong Arabic name tokens—and cross-gender verb/pronoun morphology where needed—for **58 of 200** students (`text_corrected` in `notes_resolution_summary`), drawn from **180** raw note rows.

## 3. What you built and why

- **`data_pipeline.py` ingestion layer** loads metadata, daily metrics, and facilitator notes via `config.py` paths, producing one clean in-memory master dataframe without external infrastructure overhead.
- **Deterministic Arabic note-healing** replaces imposter registry names and flips gender agreement when facilitators typed the wrong student, preserving trusted `student_id` keys while making note text LLM-ready.
- **Integrated risk tiering and CSV export** runs `assign_risk_tiers()` inside `build_master_dataframe()` and automatically writes all three tier artifacts to `outputs/` on every pipeline run.
- **`app.py` Streamlit dashboard** lets facilitators upload CSVs or use bundled `data/`, trigger cleaning with one action, and review campus-filtered Red/Yellow/Green cohorts in a single UI.
- **Human-in-the-loop outreach workflow** pairs deterministic Yellow reminders with on-demand Gemini drafts for Red-tier parents, keeping facilitators as the approval gate before any message leaves the system.

## 4. What you cut and why

- **PostgreSQL / Docker persistence** was cut in favor of pandas in-memory processing and flat CSV outputs, which delivered a working end-to-end demo within a 2-day MVP window without deployment complexity.
- **Direct automated SMS/WhatsApp API dispatch** was cut so facilitators must review, edit, and approve every outreach message—preventing erroneous parent contact from noisy data or immature LLM output.

## 5. What you'd build next

- Engineer a direct WhatsApp Business API integration that establishes an automated messaging pipeline where approved message drafts are dispatched straight to parents with a single click from the facilitator dashboard.