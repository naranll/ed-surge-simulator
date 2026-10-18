# Fresh NHAMCS parameter handoff

This standalone implementation was newly written in the empty sibling workspace
authorized by the user on 2026-10-17. It does not import or copy prior project
preprocessing modules or tests. It reads the existing external dataset. This is
not a claim of organizer eligibility: the exact event coding window, scaffold
policy, and any disclosure requirements remain subject to organizer confirmation.
Existing repositories and history remain intact. The current implementation has
not been integrated into the Simulation Team's project.

The exporter and tests require Python 3.10+ standard library only. The notebook
also uses pandas for schema inspection, plus Jupyter to execute notebook cells.
No package installation or remote environment changes are required by the code.

## Local reproduction (PowerShell, from this directory)

```powershell
$py = '..\..\ed-surge-simulator\.venv\Scripts\python.exe'
& $py -B -m unittest test_export_parameters -v
& $py -B export_parameters.py --input '..\..\ed-surge-simulator\data\raw\nhamcs\nhamcs_data_2018_22.csv' --output outputs/params.json --verify-reference --evidence
```

The existing interpreter supplies Python only; none of the existing repository's
implementation code is imported. Output JSON is deterministic; execution evidence
contains actual run timestamps and therefore deliberately changes between runs.

## LinuxONE reproduction (after transfer approval)

Upload only the five source/documentation files into a new `event_nhamcs` folder
under `/home/linux1/ed-surge-simulator`. Preserve the existing notebook, dataset,
and Simulation Team files. Do not transfer an old handoff ZIP or old scripts.

```bash
cd /home/linux1/ed-surge-simulator/event_nhamcs
uname -m
/home/linux1/jupyter-env/bin/python -B export_parameters.py \
  --input ../nhamcs_data_2018_22.csv --output outputs/params.json \
  --verify-reference --evidence
/home/linux1/jupyter-env/bin/python -m jupyter nbconvert --to notebook --execute \
  nhamcs_preprocessing.ipynb --output nhamcs_preprocessing.executed.ipynb \
  --output-dir outputs --ExecutePreprocessor.timeout=120
sha256sum ../nhamcs_data_2018_22.csv outputs/params.json
```

The notebook accepts `NHAMCS_INPUT` to override its input; its default resolves
the existing raw dataset on LinuxONE or the separate local repository. Launch it
from this directory. The kernel must use `/home/linux1/jupyter-env/bin/python`;
the notebook reports its interpreter and architecture. If the default kernel is
different, select the existing jupyter-env kernel in JupyterLab before running.
Do not assume notebook execution succeeded merely because a CLI run succeeded.

`--evidence` runs the tests, repeats the export, checks the source hash, and writes
`outputs/execution_evidence.json`. It records timestamps, architecture, `uname -m`
on Linux, Python, code/source/output hashes, test summary and reference checks.
It omits hostnames, SSH credentials, tokens, user names and environment dumps.
Notebook outputs supply additional pandas/version and schema evidence. Remote
execution is not claimed until the commands actually pass on s390x.

## Dataset and validation

Input: third-party processed NHAMCS 2018-2022 CSV. Download URL and transformation
details remain unknown. Reference SHA-256:

`439a87b766cba5e8c7b838372f45b680ba29295b148078b50e8ab7056d20349b`

Reference checks require 58,124 rows, 49 columns, 57,113 valid arrival times,
1,011 missing arrivals, peak hour 11 with 3,488 records, and triage counts
`[846, 8597, 29568, 16715, 2398]`. The independent reference checker uses raw
integer tokens and Counter rather than the production time parser. Hash mismatch
fails reference validation instead of applying these expectations to other data.

HHMM must be finite, integral, and have hour 0-23 and minute 0-59. Zero and 0000
are midnight. Blank/whitespace cells are missing; other malformed values, negative
codes and non-finite tokens are invalid. Codes have no invented sentinel meanings.
Triage accepts only integers 1-5. Each distribution uses its own denominator;
duplicate records are retained. Zero denominators and malformed CSV fail before
writing a parameter file. No source values are modified or imputed.

## JSON schema and consumption

| Key | Contract |
|---|---|
| `schema_version` | `1.0` |
| `source` | Dataset/file, SHA-256, records, column count/list, provenance status, nullable download URL, `weighting: unweighted_sample`, pooling policy |
| `hourly_arrivals` | `hours` 0-23, 24 `counts` and `proportions`, valid `denominator`, `excluded_records`, `excluded_by_reason` with missing/invalid counts |
| `triage` | `codes` 1-5, five `counts` and `proportions`, its own denominator/exclusions |
| `hourly_shape`, `triage_mix` | Compatibility objects: `status`, `tag`, `value`, weighting, denominator, source, note |
| `unavailable_parameters` | Provider service times, admission probabilities, boarding durations, hospital capacity; each has null value, unavailable status, reason, owner |
| `limitations` | Explicit source and integration caveats |

`status: verified` and `tag: DATA` mean the sample calculation passed structural
and arithmetic validation, not verified third-party clinical semantics or
population representativeness. `--verify-reference` additionally checks the known
snapshot. Arrival array index equals hour; triage index 0 is code 1. Values are
unrounded binary float ratios; sums are checked within 1e-12. Arrays in the
compatibility entries equal the respective canonical arrays.

The Simulation Team must approve replacing its PATWT-weighted 2022 baseline with
this pooled unweighted sample. It must provide baseline arrival volume, care
times, admission/boarding and capacity assumptions. Waiting time is not service
time. Weekday labels and clinical triage label equivalence are unverified. There
are no weekday factors, multipliers, scenarios, or patient/weather joins here.

After team approval, point the existing simulator at this separate file:

```bash
export ED_SCENARIO_PARAMS=/home/linux1/ed-surge-simulator/event_nhamcs/outputs/params.json
# Then start the team's existing simulation entry point in this shell.
```

The inspected consumer accepts `hourly_shape` (24 values) and `triage_mix` (5),
with status verified and tag DATA. Acceptance was inspected at commit
`82667692f89c88597f3724298e288020ddc99c16`; this is not a claim of a live integrated
test. Ensure its legacy-parameter override is off and confirm both distributions
loaded instead of accepting fallback warnings. Non-Monday weekday scaling remains
unsupported by this dataset. Do not overwrite `scenario_parameters.json`.

To retrieve generated parameters from Windows after remote execution:

```powershell
scp linuxone:/home/linux1/ed-surge-simulator/event_nhamcs/outputs/params.json .\params.json
Get-FileHash -Algorithm SHA256 .\params.json
```

Compare the hash with remote execution evidence. Generated files stay outside
source control. No Git repository was initialized or history rewritten here.
