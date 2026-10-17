# Phase 2C-1: OpenFEMA acquisition and independent source audit

## Scope

Official source: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries

Dataset documentation: https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2

API documentation: https://www.fema.gov/about/openfema/api

Pagination documentation: https://www.fema.gov/about/openfema/working-with-large-data-sets

The source filter is exactly `state eq 'TX' and fipsCountyCode eq '201'`.
For explicit Harris records the audit additionally checks source state FIPS `48`,
county FIPS `201`, designation `Harris (County)`, and `tribalRequest=false`.
Combined FIPS is 48201. Geographic source fields remain strings, including leading
zeros where present. No scalar coercion or source record transformation occurs.

The old v1 link is replaced with v2. Phase 2C-2 adds a declaration-level handoff
table. HURDAT2, physical-event consolidation, simulation scenarios, ED multipliers,
and NHAMCS joins remain outside this implementation.

## Commands

From the repository root on `feature/data-preprocessing`:

```powershell
# Creates a NEW timestamped directory under data/raw/fema/.
.venv/Scripts/python.exe -B src/data_processing/disaster_data.py

# Explicit new destination and bounded request settings.
.venv/Scripts/python.exe -B src/data_processing/disaster_data.py --output data/raw/fema/my_new_snapshot --page-size 20 --timeout 15 --retries 2

# Offline: no network calls; choose a NEW report directory each time.
.venv/Scripts/python.exe -B src/data_processing/fema_audit.py --manifest data/raw/fema/harris_20261016/manifest.json --report-dir reports/fema_review_new

# All tests, including synthetic FEMA fixtures (no live API dependencies).
.venv/Scripts/python.exe -B -m unittest discover -s tests -v

# Offline declaration-level handoff. The destination must not already exist.
.venv/Scripts/python.exe -B src/data_processing/fema_preprocessing.py --manifest data/raw/fema/harris_20261016/manifest.json --output data/processed/fema_harris_declarations.csv
```

Linux: replace the interpreter path with `.venv/bin/python` or `python`.
No dependencies were added: acquisition uses the standard library and auditing
uses the existing Pandas dependency. Cross-platform execution has not been tested.

## Immutable acquisition

Each page is retained byte-for-byte as a complete JSON API response in
`page_0001.json`, etc. The final verification response is `count_probe.json` and
is NOT appended as another record. `manifest.json` records source URL, all query
parameters, start/end/retrieval UTC timestamps, page sizes, SHA-256 hashes, response
lengths, all request attempts/errors, advertised totals, and reconciliation status.
Source `hash` fields are retained but not substituted for our SHA-256 file hashes.

Queries use `$orderby=id asc`, `$top`, `$skip`, and `$inlinecount=allpages`.
This count parameter was verified against the live v2 endpoint for this run.
Offsets advance by the actual number returned, not a presumed page size.
Completion requires consistent advertised counts on every page, exact total row
count, populated unique string IDs, expected filter/offset metadata, expected
county/state values, string geographic codes, and the final total-count probe.
An empty page before the advertised total, duplicate IDs, count drift, malformed
JSON, or unexpected geography produces an **incomplete** manifest and nonzero exit.
No failed capture is silently treated as a successful dataset. Previously saved
pages remain available for investigation. Malformed successful response bodies
are preserved even when validation fails. Transport failures retain error text;
an HTTP error response body is not guaranteed to be captured.

Default requests use a 20-second socket timeout and a 22-second total deadline;
the completed snapshot used 15 and 17 seconds respectively. A Python subprocess
enforces the total deadline, covering DNS, connection and body reads. It writes
no files and is killed on timeout. At most two retries follow the initial request,
with bounded backoff (1 and 2 seconds by default). CLI limits bound page size,
timeout and retry count. A 1,000-page safety limit prevents runaway pagination.

Existing snapshot directories are refused; retry a failed acquisition into a new
directory. Raw data is never overwritten. API contents may change between runs;
reproducibility means re-auditing the saved snapshot, not expecting future network
responses or retrieval timestamps to be identical. Stable counts and IDs cannot
prove that no concurrent source edits occurred during paging.

## Independent offline audit

`fema_audit.py` does not import acquisition helpers. It independently re-reads raw
response bytes, verifies SHA-256 hashes and offsets/filter scope, recomputes record
and unique ID counts, and reconciles every advertised total plus the count probe.
Incomplete captures that remain parseable receive failed checks. A corrupt JSON
body cannot be audited as records and raises an error; no successful report is
fabricated. Raw inputs are never written by the audit.

The JSON report contains:

- Total rows, source IDs, duplicate IDs, repeated declaration numbers, exact duplicates.
- All observed field types and missingness; date minima/maxima, missing/invalid counts.
- Begin/end/declaration comparisons, including missing ends and end-before-begin.
- Geographic values and mutually exclusive source-ID partitions: explicit Harris,
  statewide review, tribal review, unresolved/other area.
- Incident/declaration type counts and incidentId completeness.
- Shared incidentId groups, conflicting group dates/types, same-title/year candidates,
  and all overlapping valid closed-period pairs.
- Source hashes, retrieval provenance, request failures, code hash and runtime versions.

The accompanying Markdown report provides findings and recommendations. Empty and
null identifiers are counted as missing. Dates are interpreted without converting
midnight UTC labels to the previous Texas calendar date. Original date strings
are never replaced. Invalid dates remain visible in diagnostics. Missing ends
are not filled and are excluded from closed-period overlap comparisons. A late
declaration is not automatically an error. No rows are discarded or merged.

## Completed snapshot: 2026-10-16

Acquisition succeeded at approximately 12:27 UTC: three pages (20 + 20 + 1),
41 advertised records, 41 retrieved records, 41 unique IDs, and final count 41.
All four requests succeeded on their first attempts. There were no incomplete
or failed API requests in this acquisition. Synthetic timeout messages in tests
are not actual request failures.

The independent audit found 41 distinct declaration numbers and no duplicate
records. All rows are explicit Harris County designations. Declaration dates span
1973-07-11 through 2024-07-09. Incident begin/end and declaration dates are complete
and parseable; 12 disaster closeout dates are null. All 41 incidentId values are
present, comprising 37 distinct IDs. Four shared-ID pairs require review:

| Incident | Declarations | Date issue |
|---|---|---|
| Winter storms 2021 | DR-4586-TX / EM-3554-TX | Same incident period |
| Rita 2005 | DR-1606-TX / EM-3261-TX | Different begin dates |
| Ike 2008 | DR-1791-TX / EM-3294-TX | Different end dates |
| COVID-19 | EM-3458-TX / DR-4485-TX | Same multi-year incident period |

There are 17 overlapping closed-period pairs. Generic title matches also include
two 2016 severe-storm/flood declarations and do not establish event identity.
35 declaration dates follow incident begins; 5 follow incident ends. None precedes
the incident begin, and no end precedes begin. A multi-year biological declaration
must not become a continuous hospital surge assumption.

These findings describe this API-filtered snapshot only. The lack of recent or
older rows is not proof that no relevant Houston event occurred. Statewide records
were not queried. Broad, tribal or unresolved labels would be partitioned for
review if present; their zero counts here do not establish regional completeness.

## Proposed later catalogue rules (not implemented)

1. Keep declaration-area records and stable source IDs as the source layer.
2. Include only verified explicit Harris designations in the core county layer;
   assess statewide and other geography separately.
3. Keep declaration date distinct from the declared incident period and unknown ends.
4. Treat shared incident IDs as candidates requiring source review; retain every
   declaration ID even if a future reviewed crosswalk groups physical events.
5. Do not merge by title or overlapping dates alone. Preserve conflicting periods.
6. Choose hazard scope explicitly before excluding biological or other types.
7. Keep event occurrence, local exposure evidence and ED-demand assumptions separate.

## Phase 2C-2 declaration handoff

`data/processed/fema_harris_declarations.csv` contains one row for each of the 41
source declarations. It is ordered deterministically by `source_id`; `output_row`
is one-based. The preprocessing first reruns the independent snapshot-integrity
checks offline. It refuses an incomplete, altered, duplicate-ID, non-Harris, or
count-mismatched snapshot, and refuses to overwrite an existing output. It does
not contact the network, remove incident types, or consolidate declarations.

### Short data dictionary

| Fields | Meaning |
|---|---|
| `output_row`, `source_id` | Stable output order and OpenFEMA source row ID |
| `fema_declaration_string`, `disaster_number` | FEMA declaration identifiers; each source declaration remains a row |
| `incident_id` | Source grouping field; not a validated unique physical-event ID |
| `incident_type`, `declaration_type`, `declaration_title` | Source categories/title; biological and every other category retained |
| `state`, `fips_state_code`, `fips_county_code`, `county_fips`, `place_code`, `designated_area`, `tribal_request` | Source geography plus combined `48201`; geographic identifiers are strings |
| `*_date_raw` | Original FEMA timestamp text for declaration, incident begin, or incident end |
| `declaration_date`, `incident_begin_date`, `incident_end_date` | ISO `YYYY-MM-DD` derived independently from each source timestamp |
| `*_date_status` | `valid_source_date`, `missing_source`, or `invalid_source_date`; missing/invalid dates are not imputed |
| `ih_program_declared`, `ia_program_declared`, `pa_program_declared`, `hm_program_declared` | Source assistance-program boolean flags |
| `source_last_refresh`, `source_record_hash` | FEMA source update/hash fields |
| `source_record_sha256`, `source_page_*`, `source_snapshot_*` | Record, raw page, and snapshot provenance and integrity fields |
| `source_api_url`, `source_api_version`, `source_filter` | Exact source endpoint and acquisition scope |
| `geographic_scope`, `record_semantics` | Explicit Harris designation and declaration—not physical-event—semantics |

For handoff, load geographic and identifier fields as strings. Treat the three
date fields independently. Use incident categories for downstream filtering rather
than deleting records. Repeated `incident_id`, overlapping periods, or matching
titles must remain separate unless a future reviewed crosswalk explicitly groups
them. The declared incident interval is not measured hospital exposure, and no
field represents an ED-arrival multiplier.

Statewide, tribal, and unresolved geographic declarations are outside this initial
filtered table. Existing `.gitignore` excludes raw snapshots, processed outputs,
and reports; do not force-add them. No secrets or API credentials are used.
