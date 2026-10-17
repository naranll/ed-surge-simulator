# Deploying the planner dashboard (Streamlit Community Cloud)

1. Push the repo (with `models/` and `results/linuxone_run.json`, but never the raw or processed data) to GitHub.
2. Go to share.streamlit.io, sign in with GitHub, choose **New app**, pick the repo and branch.
3. Set **Main file path** to `src/dashboard/app_v2.py` (Streamlit reads `src/dashboard/requirements.txt` next to it).
4. Click **Deploy**. The first start takes a few minutes; the page simulates about 20 days in under 5 seconds.
5. Check the last panel: it must say "No LinuxONE run recorded yet" unless `results/linuxone_run.json` was really produced on LinuxONE. Do not call the page "running on LinuxONE": it is an ordinary Streamlit page.

Run locally: `streamlit run src/dashboard/app_v2.py`
