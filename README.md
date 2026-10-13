# Regression Models Explorer

An interactive walk-through of the M10 lecture *Regression Models: From the Line to State of the Art, and How to Verify What an AI Assistant Builds*.
It has 15 sections, each with toggles and sliders, and is designed to be recorded for a video.

**Live app:** https://snsie.github.io/regression-app/

Every model is trained ahead of time in Python. The React app only reads the resulting JSON, so it deploys as a static site with no server.

```
pipeline/   Python: download public data → train models → run verification analyses → JSON
web/        React + TypeScript (Vite); reads web/public/data/*.json
.github/workflows/deploy.yml   builds web/ and publishes it to GitHub Pages
```

## Data (all public)

| Dataset | Source | Role in the story |
|---|---|---|
| NHANES 2017–2018: HbA1c → fasting glucose (n≈2,500) | CDC/NCHS public-use files | Nearly linear |
| NHANES 2017–2018: systolic BP (n≈4,400) | CDC/NCHS public-use files | Weak signal; the opening "AI assistant" example |
| Diabetes progression (n=442) | scikit-learn (Efron et al. 2004) | Small n: boosting overfits |
| Parkinson's telemonitoring (5,875 recordings, 42 patients) | UCI ML Repository (Tsanas et al. 2010) | Repeated patients: leakage and the choice of split |

## Models

Baseline (mean), linear regression, random forest, LightGBM, and the **TabICL** tabular foundation model (Apache-2.0, open weights).
TabICL takes the place of TabPFN, whose current weights need a licence token.

## Regenerate the results

You need a CUDA GPU for TabICL. A full run took about 14 minutes on one NVIDIA L4.

```bash
cd pipeline
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r pyproject.toml
.venv/bin/python run_all.py          # downloads raw data, writes ../web/public/data/*.json
.venv/bin/python test_story.py       # checks the claims written in the app's text still hold
```

`run_all.py --quick` uses 2 repeats instead of 10, and `--only datasets|playground|lab` reruns one part.
If `test_story.py` fails after a rerun, update the wording in `web/src/sections/` to match the new numbers.

## Run the app locally

```bash
cd web
npm install
npm run dev          # http://localhost:5173
npm run build        # static site in web/dist
```

## Deploy to GitHub Pages

1. Push this folder to a GitHub repository, with the generated JSON committed.
2. Go to **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.
3. Push to `main`. The workflow builds `web/` and publishes it to https://snsie.github.io/regression-app/. Vite uses a relative base (`./`), so no further configuration is needed.

## Recording tips

- Use **← / →** (or PageUp/PageDown) to move between sections, and **p** to toggle presenter mode (larger text).
- Each section has its own URL (`#/leakage`, `#/compare`, …), so you can open a section directly.
- The theme button cycles through auto, dark and light.
- "Pause and think" boxes stay closed until you click **Reveal**.
- Every interactive starts in the same default state, so retakes match.
