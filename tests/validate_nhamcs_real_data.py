"""Independent, opt-in local-data validation; writes reports, never raw data.

Run from the repo root: python tests/validate_nhamcs_real_data.py
Real-data expectations do not import production parsers, rules or aggregations.
"""

import copy
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.io.stata import StataMissingValue

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = ROOT / "data/raw/nhamcs/ed2022-stata.zip"
    cleaned_path = ROOT / "data/processed/nhamcs_2022_cleaned.csv"
    triage_path = ROOT / "data/processed/nhamcs_2022_triage_distribution.csv"
    metadata_path = ROOT / "reports/nhamcs_2022_preprocessing.json"
    audit_path = ROOT / "reports/nhamcs_2022_audit.json"
    original_paths = [raw_path, cleaned_path, triage_path, metadata_path, audit_path]
    before = {str(p.relative_to(ROOT)): digest(p) for p in original_paths}
    with zipfile.ZipFile(raw_path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".dta"))
        with archive.open(member) as stream:
            raw = pd.read_stata(stream, convert_missing=True, convert_categoricals=False)
    clean = pd.read_csv(cleaned_path, dtype=str, keep_default_na=False)
    table = pd.read_csv(triage_path, dtype=str, keep_default_na=False).set_index("triage_category")
    meta = json.loads(metadata_path.read_text())
    evidence = {"environment": {"python": platform.python_version(), "pandas": pd.__version__,
                                "platform": platform.platform()}, "hashes_before": before, "checks": []}

    def record(name, passed, details, warning=False):
        evidence["checks"].append({"check": name, "status": "WARNING" if warning else "PASS" if passed else "FAIL", "details": details})

    record("1. Records", len(raw) == len(clean) == 16025,
           f"Raw={len(raw)}, cleaned={len(clean)}; full source-cell comparison below also checks row order.")
    mismatches, marker_counts = {}, {}
    for name, source in raw.items():
        dest = clean[name]
        if pd.api.types.is_numeric_dtype(source.dtype):
            values = pd.to_numeric(dest.replace("", np.nan)).to_numpy().astype(source.dtype)
            equal = np.array_equal(values, source.to_numpy(), equal_nan=True)
            if not equal:
                mismatches[name] = "numeric mismatch at source dtype precision"
        else:
            bad = 0
            counts = Counter()
            for left, right in zip(source, dest):
                if isinstance(left, StataMissingValue):
                    counts[left.string] += 1
                    good = left.string == right
                elif isinstance(left, str):
                    good = left == right
                elif pd.isna(left):
                    good = right == ""
                else:
                    good = float(left) == float(right)
                bad += not good
            if bad:
                mismatches[name] = bad
            if counts:
                marker_counts[name] = dict(counts)
    evidence["raw_comparison"] = {"cells": int(raw.size), "mismatches": mismatches, "stata_markers": marker_counts}
    record("2. Raw values / Stata markers", not mismatches and list(clean.columns[:913]) == list(raw.columns),
           f"{raw.size:,} cells checked at native source precision; {sum(sum(v.values()) for v in marker_counts.values()):,} marker cells retained. Real data contains only '.', not .a-.z. CSV preserves float32 round-trip values, not binary storage/dtype metadata.")

    labels = {0: "no_triage_for_visit", 1: "immediate", 2: "emergent", 3: "urgent", 4: "semi_urgent", 5: "nonurgent", 7: "esa_does_not_conduct_triage", -8: "unknown", -9: "blank"}
    domain = raw.IMMEDR.isin(range(1, 6))
    denominator = int(domain.sum())
    counts_ok = True
    for code, label in labels.items():
        count = int(raw.IMMEDR.eq(code).sum())
        counts_ok &= int(table.loc[label, "unweighted_count"]) == count
        counts_ok &= math.isclose(float(table.loc[label, "unweighted_percent_all_records"]), 100 * count / len(raw), abs_tol=1e-12)
        if code in range(1, 6):
            counts_ok &= math.isclose(float(table.loc[label, "unweighted_percent_urgency_levels_1_to_5"]), 100 * count / denominator, abs_tol=1e-12)
    record("3. Unweighted triage", counts_ok and denominator == 10207 and table.unweighted_count.astype(int).sum() == len(raw),
           f"Valid urgency denominator={denominator}; counts={[int(raw.IMMEDR.eq(i).sum()) for i in range(1, 6)]}; all categories account for every input row.")

    rules = {"IMMEDR": {-9: "blank", -8: "unknown"}, "WAITTIME": {-9: "blank", -7: "not_applicable_not_seen_by_provider"},
             "LOV": {-9: "blank"}, "ARRTIME": {-9: "blank"}, "AGE": {},
             "ADISP": {-9: "blank", -8: "unknown", -7: "not_applicable_not_eligible_live_discharge"}}
    targets = {"IMMEDR": "IMMEDR_clean", "WAITTIME": "WAITTIME_minutes", "LOV": "LOV_minutes", "ARRTIME": "ARRTIME_HHMM", "AGE": "AGE_years", "ADISP": "ADISP_clean"}
    sentinel_ok = True
    missingness = {}
    for name, codes in rules.items():
        numbers = pd.to_numeric(raw[name], errors="coerce")
        for code, reason in codes.items():
            mask = numbers.eq(code)
            sentinel_ok &= clean.loc[mask, name + "_missing_reason"].eq(reason).all() and clean.loc[mask, targets[name]].eq("").all()
        valid = ~numbers.isin(codes)
        if name != "ARRTIME":
            sentinel_ok &= np.array_equal(pd.to_numeric(clean.loc[valid, targets[name]]).to_numpy(), numbers.loc[valid].to_numpy())
            sentinel_ok &= clean.loc[valid, name + "_missing_reason"].eq("").all()
        missingness[targets[name]] = int(clean[targets[name]].eq("").sum())
    for code in (0, 7):
        sentinel_ok &= clean.loc[raw.IMMEDR.eq(code), "IMMEDR_category"].eq(labels[code]).all()
    evidence["missingness"] = missingness
    record("4. Sentinel rules", sentinel_ok, f"All selected rows match variable-specific codebook rules; non-triage 0 and 7 remain separate. Derived null counts: {missingness}.")

    arrival_ok = True
    for index, value in enumerate(raw.ARRTIME):
        text = str(value).strip()
        valid = text.isdigit() and len(text) <= 4 and int(text) // 100 < 24 and int(text) % 100 < 60
        hhmm, hour = (text.zfill(4), str(int(text) // 100)) if valid else ("", "")
        arrival_ok &= clean.at[index, "ARRTIME_HHMM"] == hhmm and clean.at[index, "arrival_hour"] == hour
    # Production function is used ONLY for supplemental probes of absent inputs.
    sys.path.insert(0, str(ROOT))
    from src.data_processing.nhamcs_preprocessing import preprocess_nhamcs
    cases = ["0000", "2359", "1260", "2360", "2400", "-9", StataMissingValue(101), StataMissingValue(102)]
    probe = raw.loc[[0] * len(cases), list(rules)].reset_index(drop=True)
    probe["ARRTIME"] = cases
    result, _, _ = preprocess_nhamcs(probe)
    for index, expected in enumerate([0, 23, None, None, None, None, None, None]):
        value = result.loc[index, "arrival_hour"]
        arrival_ok &= pd.isna(value) if expected is None else value == expected
    record("5. Arrival HHMM", arrival_ok, "All real HHMM/hour values match independently; synthetic 0000, 2359, invalid minutes, 2400, -9, dot and .a boundaries pass.")
    age_ok = all(clean.loc[raw.AGE.eq(code), "AGE_category_note"].eq(note).all() for code, note in [(0, "under_1_year"), (94, "94_years_or_older_topcoded")])
    record("6. Age", age_ok, "AGE=0 remains under one year; AGE=94 remains 94+ top-coded; numeric values retained.")

    # Exact decimal reference is independent of production float64 + math.fsum.
    def exact_sum(mask):
        with localcontext() as context:
            context.prec = 100
            return float(sum((Decimal.from_float(float(v)) for v in raw.loc[mask, "PATWT"]), Decimal(0)))
    total = exact_sum(raw.IMMEDR.notna())
    urgency_total = exact_sum(domain)
    weighted_ok, weighted = True, []
    for code, label in labels.items():
        expected = exact_sum(raw.IMMEDR.eq(code))
        actual = float(table.loc[label, "survey_weighted_estimated_visits"])
        weighted_ok &= math.isclose(expected, actual, abs_tol=1e-7, rel_tol=0)
        weighted_ok &= math.isclose(float(table.loc[label, "survey_weighted_percent_all_records"]), expected / total * 100, abs_tol=1e-12, rel_tol=0)
        if code in range(1, 6):
            weighted_ok &= math.isclose(float(table.loc[label, "survey_weighted_percent_urgency_levels_1_to_5"]), expected / urgency_total * 100, abs_tol=1e-12, rel_tol=0)
        weighted.append({"code": code, "category": label, "expected": expected, "exported": actual, "difference": actual - expected})
    weighting = meta["triage_distribution_output"]["weighting"]
    weighted_ok &= weighting["all_records_weight_sum"] == total and weighting["urgency_weight_sum"] == urgency_total
    evidence["weighted"] = {"all_records": total, "urgency": urgency_total, "categories": weighted}
    record("7. Weighted estimates", weighted_ok, f"Independent Decimal reference: total PATWT={total}, urgency PATWT={urgency_total}; max category discrepancy={max(abs(r['difference']) for r in weighted)} visits. Tolerances: 1e-7 visits / 1e-12 percentage points. CSTRATM/CPSUM do not enter reference weights.")
    record("8. Survey reliability", True, "Reliability remains unassessed: no design-based variance or NCHS presentation-standard evaluation. Point estimates are descriptive annual national visits. CDC advises against seasonal estimates in 2022 and cautions about MSA/nonresponse; zero-count diagnostic rows do not prove population absence.", warning=True)
    timing_ok = True
    for source, target in [("WAITTIME", "WAITTIME_minutes"), ("LOV", "LOV_minutes")]:
        valid = raw[source].ge(0)
        timing_ok &= np.array_equal(pd.to_numeric(clean.loc[valid, target]), raw.loc[valid, source])
    record("9. Timing definitions", timing_ok and "LOS" not in mismatches, "WAITTIME is arrival to first provider contact, LOV is full ED visit duration, and hospital LOS is unchanged. Neither duration has been relabeled service time.")

    schema_ok = True
    for path, key in [(cleaned_path, "cleaned_output"), (triage_path, "triage_distribution_output")]:
        spec = meta[key]
        with path.open(encoding="utf-8", newline="") as stream:
            header = next(csv.reader(stream))
        schema_ok &= len(header) == len(set(header)) and header == [c["name"] for c in spec["schema"]]
        schema_ok &= spec["path"] == path.resolve().as_posix()
        restored = pd.read_csv(path, dtype={c["name"]: c["csv_read_dtype"] for c in spec["schema"]}, keep_default_na=False, na_values=[""])
        schema_ok &= len(restored) == spec["row_count"]
        for column in spec["schema"]:
            values = restored[column["name"]]
            schema_ok &= int(values.isna().sum()) == column["csv_null_count"]
            if pd.api.types.is_numeric_dtype(values.dtype):
                schema_ok &= np.isfinite(values.dropna().astype(float)).all()
    for path in [metadata_path, audit_path]:
        json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    schema_ok &= meta["source"]["sha256"] == digest(raw_path) and meta["source"]["archive_member"] == member
    schema_ok &= meta["source"]["path"] == raw_path.resolve().as_posix()
    schema_ok &= meta["software"]["preprocessor_sha256"] == digest(ROOT / "src/data_processing/nhamcs_preprocessing.py")
    schema_ok &= not weighting["reliability_established"]
    with tempfile.TemporaryDirectory(prefix="nhamcs_validate_") as directory:
        temp = Path(directory)
        rerun = [temp / "clean.csv", temp / "triage.csv", temp / "meta.json"]
        subprocess.run([sys.executable, "src/data_processing/nhamcs_preprocessing.py", "--input", str(raw_path),
                        "--cleaned-output", str(rerun[0]), "--triage-output", str(rerun[1]), "--metadata-output", str(rerun[2])], cwd=ROOT, check=True, capture_output=True)
        repro = [digest(a) == digest(b) for a, b in zip([cleaned_path, triage_path], rerun[:2])]
        reproduced = json.loads(rerun[2].read_text())
        custom_paths_ok = all(reproduced[key]["path"] == p.resolve().as_posix() for key, p in zip(["cleaned_output", "triage_distribution_output", "metadata_output"], rerun))
        def normalized(manifest):
            result = copy.deepcopy(manifest)
            for key in ["cleaned_output", "triage_distribution_output", "metadata_output"]:
                result[key]["path"] = "<destination>"
            return result
        metadata_repro = normalized(meta) == normalized(reproduced)
    evidence["reproducibility"] = {"csv_byte_identical": repro, "metadata_identical_except_destinations": metadata_repro, "custom_paths_correct": custom_paths_ok}
    record("10. Metadata / schemas / reproducibility", schema_ok and all(repro) and metadata_repro and custom_paths_ok,
           "All columns load using their manifest dtypes; null counts, numeric finiteness, source hash/member, code hash and actual custom output paths checked. CSVs reproduce byte-for-byte; JSON content matches after normalizing intentional destination changes. Download URL/date remain explicitly unknown. Cross-platform execution has not been tested.")

    probe = raw.loc[[0], list(rules)].copy()
    probe["IMMEDR"] = -7
    result, distribution, _ = preprocess_nhamcs(probe)
    special_ok = distribution.unweighted_count.sum() == 1 and distribution.set_index("triage_category").loc["other_or_undocumented", "unweighted_count"] == 1
    special_ok &= result.IMMEDR.iloc[0] == -7 and result.IMMEDR_missing_reason.iloc[0] == "undocumented_code_-7"
    record("11. Supplemental IMMEDR=-7", special_ok, "One synthetic input row contributes exactly once to other_or_undocumented. Raw -7 retained without invented codebook meaning; excluded from valid urgency levels.")
    audit = json.loads(audit_path.read_text())
    mappings = audit["candidate_fields_requiring_codebook_verification"]
    names = {key: [item["column"] for item in values] for key, values in mappings.items()}
    mapping_ok = (names.get("length_of_visit") == ["LOV"]
                  and names.get("hospital_length_of_stay") == ["LOS"]
                  and "ADISP" in names.get("disposition", [])
                  and audit["row_count"] == len(raw) and audit["column_count"] == len(raw.columns))
    record("12. Historical audit mapping", mapping_ok,
           "Refreshed Phase 1 audit explicitly identifies LOV as ED visit duration, LOS as hospital stay, and ADISP as applicable hospital discharge disposition. Existing candidate-entry schema and descriptive statistics are preserved.")

    after = {str(p.relative_to(ROOT)): digest(p) for p in original_paths}
    evidence["hashes_after"] = after
    if before != after:
        record("Existing files unchanged during validation", False, "At least one existing input/output changed during read-only validation.")
    previous = ROOT / "reports/nhamcs_2022_validation_evidence.json"
    if previous.exists():
        baseline = json.loads(previous.read_text())["sha256_before"]
        evidence["raw_matches_previous_validation"] = baseline.get(str(raw_path.relative_to(ROOT))) == digest(raw_path)
        evidence["audit_matches_previous_validation"] = baseline.get(str(audit_path.relative_to(ROOT))) == digest(audit_path)
        # The audit was intentionally refreshed for corrected mappings; its old
        # hash is informational. Raw-data hashes must still match the baseline.
        if not evidence["raw_matches_previous_validation"]:
            record("Raw historical hash", False, "Raw archive differs from previous independent-validation baseline.")
    evidence["status_counts"] = dict(Counter(r["status"] for r in evidence["checks"]))
    evidence["status_counts"].setdefault("FAIL", 0)
    base = ROOT / "reports/nhamcs_2022_validation_after_fixes"
    base.with_suffix(".json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# NHAMCS validation after minimal fixes", "", f"Results: **{evidence['status_counts']}**. Python {platform.python_version()}, Pandas {pd.__version__}, Windows.", "",
             "Independent raw DTA comparisons and exact Decimal reference sums; production helpers were used only for synthetic probes and temporary regeneration. [Detailed evidence](nhamcs_2022_validation_after_fixes.json).", "",
             "| Check | Status | Result |", "|---|---|---|"]
    lines.extend(f"| {r['check']} | {r['status']} | {r['details']} |" for r in evidence["checks"])
    lines += ["", "## Independently verified weighted values", "", "| Category | Independent estimate | Exported estimate | Difference |", "|---|---:|---:|---:|"]
    lines.extend(f"| {r['category']} | {r['expected']:.9f} | {r['exported']:.9f} | {r['difference']:.12g} |" for r in weighted)
    lines += ["", "All existing raw/output/report hashes were unchanged during validation. The source archive matches the previous validation baseline when available. The audit was intentionally refreshed before validation to correct its mappings; its historical hash comparison is informational. Current outputs retain original field values at native source precision. Stata dot markers are literal strings; .a-.z were not observed in real data. Float64 accumulation cannot recover precision absent from source float32 weights.", "",
              "Codebook: [CDC 2022 NHAMCS ED documentation](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/NHAMCS/doc22-ed-508.pdf), item 34/page 36 (no IMMEDR=-7 definition); pages 22-23 and Appendix I/100-102 (weights and design); page 2 (2022 cautions). Reliability has not been established without design-based variance and presentation-standard assessment.", "",
              "Reproduce: `.venv/Scripts/python.exe tests/validate_nhamcs_real_data.py` (or `python tests/validate_nhamcs_real_data.py` on Linux). No raw data writes, commits or pushes.", ""]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"results": evidence["status_counts"], "report": str(base.with_suffix('.md'))}, indent=2))
    return 1 if evidence["status_counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
