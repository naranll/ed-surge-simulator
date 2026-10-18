# Deploying the planner dashboard (Streamlit Community Cloud)

1. Push the repo (with `models/` and `results/linuxone_run.json`, but never the raw or processed data) to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, choose **New app**, pick the repo and branch.
3. Set **Main file path** to `src/dashboard/app_v2.py` (Streamlit reads `src/dashboard/requirements.txt` next to it).
4. Click **Deploy**. The first start takes a few minutes; the page simulates about 20 days in under 5 seconds.
5. Open the "LinuxONE run record" panel at the bottom: it must say "No LinuxONE run recorded yet" unless `results/linuxone_run.json` was really produced on LinuxONE. Do not call the page "running on LinuxONE": it is an ordinary Streamlit page.

Run locally: `streamlit run src/dashboard/app_v2.py`

## Choosing the results folder

The page reads saved results (plan searches, scenario summary, `linuxone_run.json`, `stage_chart.png`, `demo_results.json`)
from one folder. Default: `results/`, which holds the final LinuxONE run. The sidebar box "Results folder to read" also
lists every `results/results_*` folder and every `results_*` folder at the repo root. Model files always come from `models/`.

Run from the repo root, so Streamlit picks up the light theme in `.streamlit/config.toml`
(Streamlit Community Cloud reads the same file from the repo root):

    python -m streamlit run src/dashboard/app_v2.py

Then open http://localhost:8501. Open a day type directly with `?scenario=heatwave`
(also `normal`, `hurricane`, `hurricane_burst`, `compound`), e.g. http://localhost:8501/?scenario=heatwave.
The green "Ran on IBM LinuxONE (s390x)" pill appears only when the chosen folder holds a `linuxone_run.json` with
`uname_m` = `s390x`. The page itself is an ordinary Streamlit page; it is not claimed to run on LinuxONE.
