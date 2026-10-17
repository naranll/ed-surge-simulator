# Data sources

This document lists the official sources intended for the ED Surge Simulator's initial data preprocessing pipeline. The pipeline will preserve original downloads locally and keep derived data separate. Dataset files are not included in this repository.

## NHAMCS 2022 Emergency Department patient data

- **Official source:** [CDC/NCHS NHAMCS questionnaires, datasets, and documentation](https://www.cdc.gov/nchs/nhamcs/documentation/index.html)
- **2022 ED technical documentation:** [2022 NHAMCS ED Public Use Data File Documentation (PDF)](https://ftp.cdc.gov/pub/health_statistics/nchs/dataset_documentation/NHAMCS/doc22-ed-508.pdf)
- **Purpose:** Visit-level national emergency department data for describing patient and visit characteristics used by the simulator.
- **Preprocessing note:** `src/data_processing/nhamcs_preprocessing.py` implements Phase 2A for `IMMEDR`, `WAITTIME`, `LOV`, `ARRTIME`, `AGE`, and `ADISP`. Rules follow the 2022 ED codebook (items 3–6, 34, and 251; pages 32–33, 36, and 53). Variable-specific sentinel meanings are retained in missing-reason columns; Stata extended missing markers remain distinct. `AGE=0` means under one year and `AGE=94` means 94 or older (top-coded). `LOV` is ED visit duration; `LOS` remains hospital length of stay. Triage estimates use `PATWT`; design fields `CSTRATM` and `CPSUM` are required for weighted output, while variance estimates are outside this MVP. The public-use records must not be treated as Houston-specific patient data or joined directly to local weather or disaster events without valid geographic and temporal linkage.

## NOAA GHCN-Daily weather

- **Official data archive:** [NOAA NCEI GHCN-Daily](https://www.ncei.noaa.gov/pub/data/ghcn/daily/)
- **Dataset documentation:** [GHCN-Daily documentation (PDF)](https://www.ncei.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf)
- **Purpose:** Daily station observations for a documented set of Houston-area weather stations; supports heatwave analysis and scenario construction.
- **Preprocessing note:** Station metadata, observation flags, missing-value sentinels, element units, and any station selection/aggregation rules must be documented. Heatwave threshold and duration choices are scenario parameters, not implicit data-cleaning rules.

## FEMA disaster declarations

- **Official dataset page:** [OpenFEMA Disaster Declarations Summaries v2](https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2)
- **API endpoint:** [DisasterDeclarationsSummaries v2 API](https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries)
- **Purpose:** Declared disaster event timelines and locations for event-level scenario context.
- **Preprocessing note:** Preserve the API response and request metadata. FEMA documents limitations in historical geography and manually entered records; declarations are not a complete inventory of all local hazard exposure.
- **Implemented scope:** Phase 2C-1 acquisition/audit and Phase 2C-2 declaration-level handoff preprocessing. The query explicitly filters `state eq 'TX' and fipsCountyCode eq '201'`; FIPS identifiers remain strings. Statewide relevance is not implied. No physical-event consolidation, HURDAT2 processing, or disaster demand model is implemented. See [OpenFEMA acquisition, audit, and handoff](openfema_acquisition_audit.md).

## NOAA HURDAT2 hurricane tracks

- **Official archive:** [NOAA National Hurricane Center data archive](https://www.nhc.noaa.gov/data/)
- **Format documentation:** [Atlantic HURDAT2 format documentation (PDF)](https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-atl-1851-2021.pdf)
- **Purpose:** Atlantic tropical cyclone best-track positions and storm intensity for event-level hurricane scenario analysis near Houston.
- **Implemented snapshot:** Atlantic HURDAT2 1851–2025, file `hurdat2-1851-2025-092326.txt`, retrieved 2026-10-16 from NHC. The raw-file SHA-256 is `75202780ce56c8458f16fbef5fb8a026b99b00878dbb95c458211b3e6ef125b4`.
- **Preprocessing note:** The handoff retains raw and typed track fields and reviewed FEMA candidate relationships. Nearest recorded distance to the documented Houston reference point is contextual screening only. It does not establish local wind, rain, flooding, damage, hospital exposure, or ED demand. See [HURDAT2 handoff instructions](hurdat2_handoff.md).

## Simulation parameters and evidence

Parameters will be maintained separately from observations. Each value should identify whether it is observed, literature-based, or a synthetic assumption, with units, context, uncertainty, and a citation or rationale. This prevents scenario assumptions from being mistaken for source data.
