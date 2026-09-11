# V30: Evidence references in analyst explanations

The LLM can select references such as {{fact:f0}}. Python inserts the original field path and value, retaining segment/metric identity. Naked numeric text, malformed references, missing references and guaranteed-outcome wording still trigger the Python fallback. Customer ID fields are not included in the reference map.

This validates referenced values and their provenance; it does not prove every surrounding sentence is semantically correct. Numeric claims spelled in words are discouraged by the prompt but are not fully validated. No live Ollama generation or cloud deployment was tested for this update. The full evidence remains visible separately. At most 120 scalar references are supplied to bound context size.

## Install on Windows
Extract this ZIP into your existing project folder (where README.md is located), replacing matching code files. The ZIP does not contain your secrets, virtual environment, Git history or private dataset. Keep the running tunnel, Ollama and gateway open.

From the project directory:

```powershell
git add -- src/analyst.py src/narrative_evidence.py tests/test_core.py tests/test_narrative_evidence.py docs/V30_UPDATE.md
git diff --cached --stat
git commit -m "Ground analyst narrative values in evidence references"
git push
```

On Streamlit Cloud, wait for the update and submit a fresh analyst question using Ollama. Success appears as `Interpretation: Ollama validated interpretation`. If the model ignores references, fallback remains expected; inspect its warning. Existing results do not automatically become new answers.

For the local Docker app only, rebuild with `docker compose -f compose.team.yml up --build -d`. No database re-import is needed.
