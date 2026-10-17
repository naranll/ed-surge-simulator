<!-- patient_flow:start -->
## Model 1 - patient flow (real data, NHAMCS 2022)

**What it does, in plain words.** It learns, from 10,000+ real emergency visits, two things about a patient *when they walk in*:
how likely they are to be admitted to the hospital, and how long they will need care. The simulator uses the result
(the admission lookup) so that simulated patients behave like real ones.

**Data.** NHAMCS 2022 official public-use file (CDC). 10,125 visits with a valid triage level 1-5 were used
(of 16,025); 8,547 have a valid care time. Everything is weighted by PATWT
(the survey weight). The sample is national, from 147 hospitals, not one Houston hospital. Triage is missing for about 26% of visits.

**Only facts known at arrival were used:** age_years, triage_level, arrival_hour, weekday, payment_type, sex, ambulance_arrival, injury_flag.
**Never used:** WAITTIME, LOV (they define the care-time target); ADISP / ADMITHOS / LOS / BOARDED / disposition columns (outcomes); diagnoses, tests, imaging, procedures, medicines (recorded after arrival); vital signs, pain scale and reason-for-visit (recorded at triage, not at the door); LWBS; hospital-level survey columns. HOSPCODE is used only to GROUP the split, never as a feature.

**Split.** Whole hospitals are held out (80% train / 20% test, fixed seed 20261017); no hospital is in both. Inside the training part,
5-fold hospital-grouped, stratified cross-validation chose the model. Rule: take the simplest model whose CV error is within one
standard error of the best.

**Admission model.** Chosen: **hgb** {'l2_regularization': 0.0, 'learning_rate': 0.1, 'max_depth': 2, 'max_iter': 120, 'min_samples_leaf': 20} (searched 20 settings).
On the held-out hospitals: AUC 0.845 (95% 0.790 to 0.878); "triage level only" baseline AUC 0.732 (95% 0.651 to 0.788).
Brier 0.081 (95% 0.054 to 0.117) vs 0.090 (95% 0.057 to 0.130). Calibration error (ECE) 0.025 (95% 0.020 to 0.059).
Lookup table (triage x age band) AUC 0.821 (95% 0.735 to 0.866).
Plain reading: age adds some information on top of triage level, but a lot is unknown at the door, so this is a modest model, not a crystal ball.

**Care-time model (log scale).** Chosen: **hgb** {'l2_regularization': 5.0, 'learning_rate': 0.03, 'max_depth': 2, 'max_iter': 120, 'min_samples_leaf': 50}. Median absolute error (log) 0.532 (95% 0.479 to 0.601),
R-squared 0.193 (95% 0.095 to 0.249); baseline (median by triage) MAE 0.694 (95% 0.631 to 0.811) vs model 0.676 (95% 0.628 to 0.763).
**Care time = LOV - WAITTIME is a proxy**: it is the whole stay after first provider contact, including tests and, for admitted patients, boarding. It is not pure service time.

**Limits.** Survey data (about 40% response rate, possible non-response bias). No variance estimates. Only 139 level-1 visits.
Model accuracy is on held-out hospitals of the same survey, not on a real Houston hospital.
Saved files: admission: admission_xgboost.json; admission_note: CV winner is HistGradientBoosting (cannot be saved portably); the saved file is the best portable candidate instead: xgb {'n_estimators': 120, 'learning_rate': 0.06, 'max_depth': 2, 'min_child_weight': 1.0, 'subsample': 0.9, 'reg_lambda': 5.0}; care_time: care_time_xgboost.json; care_time_note: CV winner is HistGradientBoosting (cannot be saved portably); the saved file is the best portable candidate instead: xgb {'n_estimators': 120, 'learning_rate': 0.06, 'max_depth': 2, 'min_child_weight': 1.0, 'subsample': 0.9, 'reg_lambda': 5.0}.
The admission lookup that the simulator reads is a descriptive table computed on ALL valid rows; its accuracy on held-out hospitals (shown above) was measured with a copy built from the training rows only.
<!-- patient_flow:end -->

<!-- early_warning:start -->
## Model 2 - early warning (simulated days)

**What it does, in plain words.** Every 30 simulated minutes it looks at what a charge nurse could see (queues, longest waits, bed use, who is on shift, the hour,
and how many patients the forecast expects) and says how likely it is that in the next 4 hours (1) urgent patients will be seen late, (2) beds will be over 95% full,
(3) patients will walk out unseen. "Breaks first" is the risk that is highest compared with its own alarm line.

**This model learns the SIMULATOR, which rests on stated assumptions. It is a fast early-warning surrogate, not validated on a real hospital.**

**Data.** 750 day numbers x 4 scenario types (normal, heatwave, hurricane, hurricane with a burst) = 3000 simulated days,
120,000 snapshots; hospital size cycles Small / Medium / Large. Labels come from what happens next in the same run under "do nothing".
Thresholds that define a breach are ASSUMPTIONS (urgent late: at least 2 late urgent patients and at least 25% of urgent patients in the window; beds: occupancy over 95%; leaving: at least 1).

**Honest splits.** By whole day number: 450 train / 150 validation / 150 test days. A day number is never in two splits (in any scenario),
and the seed range for the optimiser's look-ahead runs (200000+) never overlaps the truth seeds (100000+). Features never use the true day or the future (a test checks the feature list).
Settings were picked once on the stage-3 training days with day-grouped cross-validation and the "simplest within one standard error" rule, among models that can be saved in a portable file (logistic or XGBoost; HistGradientBoosting was tried too).

**Results (95% ranges are in models/stage_results.csv).**
- **urgent late**: stage 1 AUC 0.762 (normal days) -> stage 2 on heatwave 0.846, hurricane 0.819, hurricane+burst 0.861 -> stage 3 (all scenarios) 0.856. Pooled over all four scenario types (mostly surge days), the alarm line set at 5% false alarms on NORMAL days catches 76% of risky moments (the simple rule: 43%). Leave-one-scenario-out AUC: hurricane 0.819, heatwave 0.845.
- **beds full**: stage 1 AUC 0.907 (normal days) -> stage 2 on heatwave 0.933, hurricane 0.926, hurricane+burst 0.947 -> stage 3 (all scenarios) 0.935. Pooled over all four scenario types (mostly surge days), the alarm line set at 5% false alarms on NORMAL days catches 81% of risky moments (the simple rule: 61%). Leave-one-scenario-out AUC: hurricane 0.923, heatwave 0.934.
- **lwbs**: stage 1 AUC 0.962 (normal days) -> stage 2 on heatwave 0.970, hurricane 0.968, hurricane+burst 0.964 -> stage 3 (all scenarios) 0.968. Pooled over all four scenario types (mostly surge days), the alarm line set at 5% false alarms on NORMAL days catches 89% of risky moments (the simple rule: 78%). Leave-one-scenario-out AUC: hurricane 0.968, heatwave 0.970.

**What the stages really showed (computed from the table, not hand-written).**
- urgent late: ranking quality (AUC) 0.762 on normal days vs 0.842 on unseen surge days (NOT worse). Catch rate at the alarm line: 28% (normal days) -> 61% (surge days, stage-1 model) -> 81% (surge days, retrained). On normal days the model catches 29% vs 32% for the simple rule.
- beds full: ranking quality (AUC) 0.907 on normal days vs 0.935 on unseen surge days (NOT worse). Catch rate at the alarm line: 53% (normal days) -> 63% (surge days, stage-1 model) -> 85% (surge days, retrained). On normal days the model catches 54% vs 52% for the simple rule.
- lwbs: ranking quality (AUC) 0.962 on normal days vs 0.967 on unseen surge days (NOT worse). Catch rate at the alarm line: 78% (normal days) -> 88% (surge days, stage-1 model) -> 91% (surge days, retrained). On normal days the model catches 78% vs 72% for the simple rule.
Note: "patients leaving unseen" is very predictable because patience is a fixed 180-minute ASSUMPTION in the simulator (anyone already waiting about 150 minutes will leave), so a high score there says little about a real hospital.
Alarm lines come from normal-day validation snapshots only; surge days contain many more risky moments, so a higher catch rate there is expected.

**How to read the stages.** Stage 1 is the easy case (normal days only). Stage 2 shows what happens when surge days were never seen in training: the model is worse.
Stage 3 retrains with surge days. The leave-one-scenario-out rows test a surge type the model never saw in training.
<!-- early_warning:end -->
