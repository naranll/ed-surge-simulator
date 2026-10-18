# PulseOps - ED Surge Simulator

Live demo (Streamlit, deployed from a separate repo that is not synced with this one): https://ed-surge-simulator.streamlit.app/

A simulator and early-warning tool for a hospital emergency department (ED). It runs a normal day, a heatwave, a hurricane
or a combined surge, shows where the ED strains first (urgent waits, beds, or staff), warns about the next 4 hours, and tests
fixes (an extra doctor, fast track, faster ward transfers). It recommends a plan only when the benefit is clear and non-urgent
patients do not wait unfairly longer. Built for the IBM Z Datathon 2026.

**This is a simulation.** It is not a real hospital and not medical advice. The ED is a model; patient behaviour comes from
national survey data (NHAMCS 2022), and many settings are stated assumptions (listed below).

## Data used

| Dataset | Used by code? | For what |
| --- | --- | --- |
| **NHAMCS 2022 emergency department file** (CDC public-use file) | **Yes. The only patient dataset read by code.** | Arrivals by hour, weekday shares, triage mix, admission by triage level and age, care time, calibration targets (urgent median wait, share leaving unseen). Weighted by the survey weight PATWT. Not in the repo; anyone can download it. |
| Published studies | Yes, as cited numbers | Surge sizes only: heatwave +6% to +13% of visits; hurricane (receiving hospital) +9% to +13%. As cited by the team, not independently checked. |
| NOAA weather, FEMA declarations, NOAA hurricane tracks | Context only | Background for the story. They never set a surge size and are never joined to patient records. |
| CMS Timely and Effective Care, MIMIC-IV-ED | **No** | Not used. Future work: a check against a real hospital's timings. |

## What is inside

| Part | What it does | Where |
| --- | --- | --- |
| Simulator | Event-by-event model: arrival, triage, bed, doctor, then discharge or admission. Plain Python, no extra libraries. | `src/simulation/ed_simulator.py` |
| Scenario registry | ONE definition of all five scenarios, used by every entry point. | `src/simulation/registry.py` |
| Capacity file | ONE source of the hospital size (beds, doctors, triage nurses), fitted by calibration. | `configs/capacity.json` |
| Patient-flow model (model 1) | Learns from real NHAMCS visits how likely a patient is to be admitted and how long care takes. The simulator uses a small lookup table made from it. | `scripts/train_patient_flow.py`, `models/patient_flow/` |
| Early-warning model (model 2) | Learns from thousands of simulated days the chance, in the next 4 hours, of late urgent care, full beds, or patients leaving unseen. | `scripts/train_early_warning.py`, `models/predict_risk.py` |
| Plan search (whole day) | Tests an extra doctor for 6 hours, fast track and faster ward transfers at different start times on the same random days. | `src/optimization/optimizer.py` |
| Live demo (next 4 hours) | Same rules as the plan search, applied to the next 4 hours of a live day. | `run.py`, `src/demo/live_demo.py` |
| Dashboard | A what-if page for a hospital planner (laptop only). | `src/dashboard/app_v2.py` |
| One-command run | Tests, scenarios and plan search, both models, timing. | `run_all.py` |

## Other parts in this repo

| Part | What it is | Where | How to run |
| --- | --- | --- | --- |
| Classic dashboard | The first Streamlit digital-twin page: 20-bed floor plan, waiting room, intervention toggles, 24-hour charts, shift briefing. Uses the original placeholder simulator settings, not the calibrated capacity or the plan-search rulebook. | `src/dashboard/app.py` | `python run.py dashboard` or `streamlit run src/dashboard/app.py` |
| Web front end / Digital Twin | Interactive React + TypeScript digital twin with 4-hour ML early warning surrogate, hospital scale presets, and real-time bed simulation. | `frontend/` | `python run.py twin` (portable static twin, works on s390x & local) or `cd frontend && npm run dev` |
| NHAMCS 2018-22 export with LinuxONE evidence | A standalone, standard-library exporter of arrival-hour and triage shares from the third-party NHAMCS 2018-22 file, with an execution record from an s390x machine (2026-10-17). The simulator does not read it; it is a cross-check only. | `event_nhamcs/` | See `event_nhamcs/README.md` |

## Scenarios

| Key | What it is | Source |
| --- | --- | --- |
| `normal` | Ordinary Monday | Hourly shape and triage mix: NHAMCS 2022. Volume 200/day: UNSOURCED |
| `heatwave` | +6% to +13% visits, extra visits 12:00-20:00, known the day before | Size: Nature Health study of summer-2025 US heatwaves (team-cited). Window: UNSOURCED |
| `hurricane` | +9% to +13% all day at a receiving hospital, known the day before | Hurricane Harvey, Dallas-Fort Worth receiving hospital (team-cited) |
| `hurricane_burst` | Hurricane plus 10-20 very urgent patients in one unknown hour | Burst: UNSOURCED stress test |
| `compound` | Heatwave and hurricane on the same day | Both cited ranges. **ASSUMPTION: effects multiply (independence).** Not a historical event |

The planner only sees what is known in advance (the scenario and its cited size range), never the true day, its sampled size or the burst hour.

## How a plan is chosen (same rules everywhere)

- **Score** = % of urgent patients (triage levels 1-3) seen within their time target, minus 7 points per $1000 of cost, minus 0.5 points per patient who leaves unseen.
- **Fairness**: the median wait of non-urgent patients may rise by at most 15% (+1 minute).
- **Bar**: a plan is recommended only if the low end of the 95% range of its gain over "do nothing" is above 2 points.
- **Rule (c)**: when urgent patients are seen late more often than on a normal day, "do nothing" is not allowed while some tested plan's 95% range stays above 0.
  If none does, the output says "No tested plan helps; constraint is <boarding / beds / staff>".
- `run.py demo` applies these rules to the **next 4 hours** and always also prints the **whole-day plan** for the scenario, with one sentence if they differ (a busy afternoon can justify an extra doctor even when the whole day does not).
- Every plan shows its cost in dollars and in staff-hours. **Costs are UNSOURCED and illustrative.**

## Run it

You need Python 3.11 or newer.

```bash
pip install numpy pandas scikit-learn xgboost matplotlib scipy pytest   # models, tests
pip install -r src/dashboard/requirements.txt                          # dashboard only (laptop; needs pyarrow, not available on s390x)
```

Put the cleaned NHAMCS file at `data/processed/nhamcs_2022_cleaned.csv`. Without it the run still works, but model 1 and the calibration step are skipped with a clear message.

| Goal | Command | Time (8-core PC) |
| --- | --- | --- |
| Live demo, one scenario | `python run.py demo --scenario heatwave --seed 42 --hour 14` | about 1-3 s |
| Live demo through the day | `python run.py rolling --scenario compound --seed 42` | about 1 minute |
| Quick check (2 scenarios, 30 plan-search runs) | `python run_all.py --quick` | about 2-4 minutes |
| Full run (5 scenarios x 50 days, both models) | `python run_all.py` | about 10 minutes |
| Dashboard | `streamlit run src/dashboard/app_v2.py` | opens in the browser |
| Tests | `python -m pytest -q tests` | about 1.5 minutes |

Useful flags: `--scenario normal|heatwave|hurricane|hurricane_burst|compound`, `--replications N` (forecast runs per plan, default 30), `--workers N` (worker processes; helps the plan search, barely helps the demo).
`ED_USE_LEGACY_PARAMS=1` restores the original placeholder numbers and the original `run.py` behaviour exactly.

**Warning:** `run_all.py` rewrites `models/` and `results/`. The quick run replaces full-size results with small ones, so run the full command again afterwards.
A run writes `results/run.log` and a run record: `results/linuxone_run.json` if the machine is s390x, otherwise `results/local_run.json`.

## Architecture

```
NHAMCS 2022 (real visits) --> patient-flow model --> admission / care-time lookup --+
NHAMCS 2022 targets --> calibration --> configs/capacity.json (beds, doctors, nurses)-+
Published studies (cited ranges) --> scenario registry (5 scenarios) ---------------+
                                                                                     v
        registry --> true day (hidden from planner) + advance info -----------> Simulator
                                  |                                               |
                                  v                                               v
                   Forecast (advance info only)                     metrics + snapshots every 30 min
                                  |                                               |
                                  v                                               v
       Plan search (whole day) and live demo (next 4 h):            Early-warning model --> 3 risks + "breaks first"
       same score, fairness, 2-point bar, rule (c)
                                  \_____________________________  ______________________________/
                                                                \/
                                             run.py output / dashboard / results files
```

## Assumptions and UNSOURCED values

**From data (NHAMCS 2022, computed in code):** arrivals by hour, weekday shares, triage mix, admission chance by triage level and age band,
care time by triage level (admitted / discharged), urgent median wait 15 min and 1.9% leaving unseen (calibration targets).

**Cited, not independently checked:** heatwave +6% to +13%, hurricane +9% to +13%; time targets from the Australasian Triage Scale
(immediate, 10, 30, 60, 120 min), applied to NHAMCS triage levels 1-5 as an assumption.

**Fitted, not measured:** the reference hospital (60 beds, 4 doctors, 3 triage nurses) is chosen so the simulated normal day matches the NHAMCS targets. NHAMCS has no bed or staff counts.

**UNSOURCED (our choices):**
- volume 200 patients per day; Small / Medium presets at 30% / 60% of it (not calibrated); calibration search grid; "not saturated" = peak beds under 95%;
- doctor consult time (means 45 / 30 / 20 / 12 / 8 min by level), triage time 2-4 min, fixed patience of 180 min before leaving unseen;
- fast-track effect (-40% doctor time for levels 4-5) and faster ward transfers (25% / 50%);
- minimum bed hold 5 min after the consult; care time capped at 24 h; small-cell shrinkage k = 10 in the admission table;
- heatwave window 12:00-20:00; burst size 10-20, hour 12:00-21:00, 25% level 1; compound effects multiply;
- all costs ($150 per clinician-hour, $150 per fast-track window, $300 per 30-point transfer cut), cost penalty 7 points per $1000, 0.5 points per patient leaving unseen,
  fairness limit 15% + 1 min, 2-point bar, 6-hour plan windows starting every 3 hours, 4-hour demo window;
- rule (c) helpers: boarding is "the constraint" when admitted patients waiting for a ward hold 25% or more of beds, a resource busy 90% or more is "saturated", up to 3 plans re-checked;
- early-warning breach definitions (at least 2 late urgent patients and 25% late; level-1 grace 10 min; beds above 95%; at least 1 leaving unseen) and the 5% false-alarm line.

## Limits (read these)

- The early-warning model learns the **simulator**, which rests on the assumptions above; it is not validated on a real hospital.
- Only the reference (Large) hospital is calibrated. On a normal day the simulated median wait for urgent patients is about **5 minutes**; the survey reports about **15 minutes**. We report this gap and have not closed it.
- Triage level is missing in **25.9%** of 2022 visits, and there are only **139** level-1 visits. National sample (response rate about 40%), not one hospital.
- Care time is "leave time minus wait time" from the survey: a proxy that includes tests and, for admitted patients, boarding. It is not pure doctor time.
- The "leaving unseen" warning looks very accurate partly because everyone has the same assumed patience of 180 minutes.
- On normal days the early-warning model does **not** beat a simple rule ("alarm if the urgent queue is long") for late urgent care. It does better on surge days.
- The live demo's 4-hour view and the whole-day plan can differ on purpose; both are printed.
- LinuxONE results are claimed only if `results/linuxone_run.json` exists, and every display checks that it says `s390x`.

## Tech stack

Python 3.11+. The simulator, scenarios, registry, forecast, plan search and live demo use only the standard library.
Models and plots: numpy, pandas, scikit-learn, XGBoost, scipy, matplotlib. Dashboard: Streamlit (laptop only). Tests: pytest.
Models are saved as JSON so they load on any machine, including big-endian s390x. Never copy pickle files between machines.

## Run on IBM LinuxONE

Only one person may use the team's LinuxONE account at a time. Paths below are examples; change them to match yours.
Target container: Python 3.12, numpy 2.0.2, pandas 3.0.6, scikit-learn 1.9.0, xgboost 3.4.1, scipy 1.13.1, matplotlib 3.11.2. No pyarrow, no Streamlit: the pipeline needs neither.

1. Log in and check the machine. `uname -m` must print `s390x`. Also run `nproc` and `lscpu`.
2. Get the code (first time: clone; later: pull):
   ```bash
   git clone <repo-url> ed-surge-simulator      # first time only
   cd ed-surge-simulator
   git checkout feature/prediction-model
   git pull
   ```
3. Copy the cleaned data from your PC (it is not in git). On LinuxONE: `mkdir -p data/processed`. On your PC:
   ```powershell
   scp nhamcs_2022_cleaned.csv linuxone:~/ed-surge-simulator/data/processed/
   ```
   Check both ends match: `sha256sum data/processed/nhamcs_2022_cleaned.csv` (PowerShell: `Get-FileHash -Algorithm SHA256 <file>`).
4. Choose the Python and check the libraries:
   ```bash
   PY=/home/linux1/jupyter-env/bin/python        # or python3
   $PY -c "import numpy, pandas, sklearn, xgboost, scipy, matplotlib; print(numpy.__version__, pandas.__version__, sklearn.__version__, xgboost.__version__, scipy.__version__)"
   $PY -m pytest --version || $PY -m pip install --user pytest      # pytest may be missing; without it only the test step fails
   ```
5. Quick run inside tmux, so it survives a dropped connection:
   ```bash
   tmux new -s edrun
   $PY run_all.py --quick --workers 3 2>&1 | tee quick_run.out
   ```
   Expected: one line per step (`tests`, `scenarios_and_plan_search`, `early_warning_data`, a 1/2/4/8-worker timing table, `calibration`,
   `patient_flow_model`, `early_warning_model`, `demo_results`, `plots`), each ending `ok`, then `Wrote results/linuxone_run.json`.
   If it says `local_run.json`, the machine was not s390x.
6. Full run, then the live demo for every scenario:
   ```bash
   $PY run_all.py --workers 3 2>&1 | tee full_run.out
   for s in normal heatwave hurricane hurricane_burst compound; do $PY run.py demo --scenario $s --seed 42; done | tee demo.out
   ```
   Detach with `Ctrl-b` then `d`; come back with `tmux attach -t edrun`.
7. Look at the result: `cat results/linuxone_run.json` (machine, cores, seconds per step, worker timings) and `ls results results/for_slides`.
8. Bring results home (PowerShell): `scp -r linuxone:~/ed-surge-simulator/results .\results_from_linuxone`.

If a step fails, the run records it and continues; read `results/run.log`. A step that needs the data file is skipped with a message if the file is missing.
On s390x one test (exact text hashes of generated arrivals) is skipped on purpose: the last digit of some maths results can differ on that CPU; a separate test still checks the arrivals exactly.
Say "ran on LinuxONE" only about what `results/linuxone_run.json` records.

## Team

- **Abhigyan Tiwari**: simulator, first optimiser, dashboards, web front end ([GitHub](https://github.com/Abh-igyan))
- Team KnightAI, IBM Z Datathon 2026

## License

MIT. See [LICENSE](LICENSE).
