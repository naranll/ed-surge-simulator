# Key numbers (auto-written by scripts/make_plots.py)

Every number below comes from a file in this repo. Sources: NHAMCS 2022 (CDC public-use file) for patient behaviour; published studies for surge sizes (heatwave +6% to +13%, hurricane +9% to +13%, as cited by the team, not independently checked). Weather, FEMA and hurricane records are context only.

## Model 1 - patient flow (real data)
- Visits used: 10,125 of 16,025 (valid triage 1-5), 147 hospitals; held-out hospitals for testing.
- Admission: hgb; AUC 0.845 (95% 0.790-0.878) vs triage-only baseline 0.732.
- Care time (log, proxy LOV-WAITTIME): R-squared 0.19; error 0.676 vs baseline 0.694.

## Model 2 - early warning (simulated days; learns the simulator)
- 750 day numbers x 4 scenario types, 150,000 snapshots; split by whole day (450/150/150).
- beds_full: AUC 0.940; catches 87% of risky moments at 5% false alarms on normal days (simple rule: 61%).
- lwbs: AUC 0.974; catches 92% of risky moments at 5% false alarms on normal days (simple rule: 79%).
- urgent_late: AUC 0.873; catches 82% of risky moments at 5% false alarms on normal days (simple rule: 46%).

## Calibration (real vs simulated normal day)
```
quantity,real_NHAMCS_2022,simulated_normal_day,note
"urgent (levels 1-3) median wait, min",15.0,5.1,NHAMCS WAITTIME vs simulated door-to-doctor
"left without being seen, %",1.89,1.59,NHAMCS LWBS flag vs simulated (patience is a fixed 180 min ASSUMPTION)
"admitted to hospital, % of triaged visits",12.24,11.9,ADMITHOS vs simulated admission draw
"peak bed occupancy, % (design target, not data)",,86.44,ASSUMPTION: keep under 95% on a normal day
"peak arrival hour share, % (10:00)",6.43,6.21,NHAMCS ARRTIME vs simulated arrivals
"quietest arrival hour share, % (05:00)",1.71,1.47,NHAMCS ARRTIME vs simulated arrivals
"median care time level 1, min",332.0,338.55,LOV-WAITTIME (proxy) vs simulated consult + bed stay
"median care time level 2, min",257.0,250.42,LOV-WAITTIME (proxy) vs simulated consult + bed stay
"median care time level 3, min",186.0,173.66,LOV-WAITTIME (proxy) vs simulated consult + bed stay
"median care time level 4, min",91.0,85.7,LOV-WAITTIME (proxy) vs simulated consult + bed stay
"median care time level 5, min",70.0,76.42,LOV-WAITTIME (proxy) vs simulated consult + bed stay

chosen capacity (ASSUMPTION set by calibration),beds=60,doctors=4,triage nurses=3; 30 simulated days
```

## Scenarios (reference hospital)
```
scenario,hospital,days,patients_per_day,urgent_late_pct,lwbs_pct,peak_bed_occupancy_pct,longest_queue,ms_per_simulated_day,recommended_plan,plan_search_seconds,plan_search_replications
normal,Large,50,200.9,13.75,1.89,87.1,13.5,7.08,Do nothing,8.0,50
heatwave,Large,50,219.5,26.73,6.78,96.0,21.2,7.65,"12:00-18:00 +1 clinician, fast track",8.8,50
hurricane,Large,50,222.7,22.31,5.25,94.9,19.2,7.81,12:00-18:00 +1 clinician,8.9,50
hurricane_burst,Large,50,237.6,45.04,11.09,99.1,31.4,8.1,12:00-18:00 +1 clinician,8.9,50
compound,Large,50,243.2,38.92,11.76,99.1,28,8.35,"12:00-18:00 +1 clinician, fast track",9.7,50
```

## Limits and honesty
- Triage level is missing in 25.9% of 2022 visits; wait time in 17.2%. Only 139 level-1 visits. National sample (about 40% response rate), not one hospital.
- Care time LOV-WAITTIME is a proxy (includes tests and boarding), not pure service time. ATS time targets are applied to NHAMCS triage levels as an assumption.
- Hospital sizes, doctor consult times, patience (180 min), fast-track effect, breach thresholds and plan costs are ASSUMPTIONS (see the parameter table in the dashboard).
- The early-warning model learns the simulator, which rests on stated assumptions; it is not validated on a real hospital.
- The 'urgent median wait' of the simulator is lower than the survey's (see calibration table); we report this gap rather than hide it.

## LinuxONE
- Recorded run: s390x, 3 cores, 226.4 s total.