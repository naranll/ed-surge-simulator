# HURDAT2 handoff

Run offline from the repository root:

```powershell
.venv/Scripts/python.exe -B src/data_processing/hurdat2_preprocessing.py
```

The command verifies the existing NOAA/NHC snapshot against its manifest, parses
all storm headers and track records, verifies 1851–2025 coverage, and refuses to
overwrite existing outputs. It then exports:

- `data/processed/hurdat2_atlantic_tracks.csv`: 55,524 observed track rows.
- `data/processed/fema_hurdat2_candidate_crosswalk.csv`: 19 reviewed candidate
  relationships across 18 FEMA declarations.

The track schema preserves `storm_id`, `storm_name`, source date/time text, UTC
timestamp, HURDAT2 record identifier, status, raw and signed numeric coordinates,
raw and typed maximum sustained wind and central pressure, explicit missing-source
status, source line number/hash, and file/manifest provenance. Native wind is knots;
pressure is millibars. `-99`/`-999` wind and `-999` pressure remain in raw columns
and become empty typed values with `missing_source_sentinel` status. No positions,
winds, or pressures are interpolated.

The crosswalk preserves FEMA and storm identifiers, both lifetimes, temporal-overlap
result, nearest recorded distance to Houston reference point 29.7604, -95.3698,
geographic screening category, reason, and confidence exactly as `High`, `Medium`,
or `Unresolved`. Every row is marked `candidate_not_confirmed_physical_event`.
Marco/Laura remains one FEMA declaration with two storm candidates. Rita and Ike
retain separate EM/DR declarations linked to the same candidate storm.

Load identifier, raw-value, status, confidence, and provenance columns as strings.
Typed coordinates and distance are decimal degrees/kilometres; typed wind/pressure
are nullable integers. An empty typed value is explained by its adjacent status
field. Timestamps are UTC ISO strings.

Nearest recorded center distance is contextual screening. It does not measure
local wind, precipitation, flooding, surge, damage, hospital exposure, or ED demand.
The best-track data is generally six-hourly and this pipeline performs no temporal
interpolation. FEMA records remain administrative declarations, not unique physical
events. The pipeline implements no GIS polygons, ED multiplier, simulation logic,
visualization, patient linkage, or automatic declaration merge.

Generated data remains covered by `.gitignore`. Do not force-add raw or processed
datasets. No network request is made by preprocessing.
