"""Extract the baseline scenario numbers from the real NHAMCS 2022 data.

What this script does (in plain words)
--------------------------------------
It reads the cleaned NHAMCS 2022 file and works out four things:

  1. hourly_shape    - share of ED arrivals in each clock hour (0..23)
  2. weekday_shares  - share of arrivals on each day of the week (VDAYR 1..7)
  3. monday_factor   - Monday share divided by an even share (1/7)
  4. triage_mix      - share of patients at triage levels 1..5

It writes them to a small JSON file (totals only, no patient rows) that the
scenario code reads. It also checks the numbers against the official Stata
file and against the numbers quoted in the project notes, and prints a table.

Honesty rules
-------------
  * "tag" is DATA only for numbers computed from the real file.
  * "status" is "verified" only if the cross-checks below passed.
    Otherwise it is "unverified".
  * Nothing here is a surge size. NHAMCS has no dates or places, so surge
    sizes come only from published studies (set elsewhere, tagged EVIDENCE).
  * The NOAA heatwave events are CONTEXT only. They are never a multiplier.

Run it from the repo root (the defaults are repo-relative):

    python scripts/extract_scenario_parameters.py

or with explicit paths:

    python scripts/extract_scenario_parameters.py --cleaned X.csv \
        --dta-zip ed2022-stata.zip --events heatwave_events.csv --out out.json

This script may use pandas. The simulator and optimiser must not; they only
read the JSON file this script writes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED = REPO_ROOT / "data/processed/nhamcs_2022_cleaned.csv"
DEFAULT_DTA_ZIP = REPO_ROOT / "data/raw/nhamcs/ed2022-stata.zip"
DEFAULT_EVENTS = REPO_ROOT / "data/processed/heatwave_events.csv"
DEFAULT_OUT = REPO_ROOT / "src/simulation/params/scenario_parameters.json"

SCHEMA_VERSION = "1.0"
SUM_TOLERANCE = 1e-9       # shares must add up to 1 within this
DECIMALS = 12              # shares are rounded to this many decimals in the file
PASS_LIMIT = 0.10          # "pass" means within 10 percent (relative)
WEIGHT_RTOL = 1e-6         # PATWT in the CSV vs the .dta (the .dta stores float32)

HOURS = list(range(24))
WEEKDAY_CODES = [1, 2, 3, 4, 5, 6, 7]
TRIAGE_LEVELS = [1, 2, 3, 4, 5]

# Numbers quoted in the project notes that we compare against (we never copy them over).
NOTES_VALUES = {
    "peak_hour_share": 0.064,
    "hour_10_share": 0.064,
    "quietest_hour_share": 0.017,
    "monday_share": 0.169,
    "saturday_share": 0.125,
    "sunday_share": 0.125,
    "triage_counts": [139, 1595, 5340, 2826, 307],
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_list(values) -> list:
    return [round(float(v), DECIMALS) for v in values]


def check_sums_to_one(name: str, values: list) -> None:
    """Fail loudly if a list of shares does not add up to 1."""
    total = sum(values)
    if abs(total - 1.0) > SUM_TOLERANCE:
        raise ValueError(f"SELF-CHECK FAILED: {name} sums to {total!r}, not 1")


def weighted_and_plain_shares(codes: pd.Series, weights: pd.Series, categories: list):
    """Return (weighted shares, unweighted shares, unweighted counts) per category.

    codes and weights must already be limited to the valid rows.
    """
    codes = codes.astype(int)
    weight_sums = weights.groupby(codes).sum().reindex(categories, fill_value=0.0)
    counts = codes.value_counts().reindex(categories, fill_value=0).astype(int)
    weighted = weight_sums / weight_sums.sum()
    plain = counts / counts.sum()
    return weighted, plain, counts


# --------------------------------------------------------------------------
# Reading the data (read only)
# --------------------------------------------------------------------------

def load_cleaned(path: Path) -> pd.DataFrame:
    """Read only the columns we need from the cleaned CSV.

    Everything is read as text first, so blanks stay blanks (no hidden NaN
    rules), then the number columns are converted on purpose.
    """
    wanted = ["PATWT", "ARRTIME", "arrival_hour", "VDAYR", "IMMEDR", "IMMEDR_clean"]
    raw = pd.read_csv(path, usecols=wanted, dtype=str, keep_default_na=False)
    frame = pd.DataFrame({
        "PATWT": pd.to_numeric(raw["PATWT"], errors="coerce"),
        "ARRTIME": raw["ARRTIME"].str.strip(),
        # empty text becomes NaN here (arrival time was invalid or blank)
        "arrival_hour": pd.to_numeric(raw["arrival_hour"], errors="coerce"),
        "VDAYR": pd.to_numeric(raw["VDAYR"], errors="coerce"),
        "IMMEDR": pd.to_numeric(raw["IMMEDR"], errors="coerce"),
        "IMMEDR_clean": pd.to_numeric(raw["IMMEDR_clean"], errors="coerce"),
    })
    if frame["PATWT"].isna().any() or (frame["PATWT"] <= 0).any():
        raise ValueError("PATWT has missing or non-positive values; cannot weight.")
    if frame["VDAYR"].isna().any():
        raise ValueError("VDAYR has missing values in the cleaned file.")
    return frame


def load_official_dta(zip_path: Path) -> dict:
    """Read the official Stata file straight from the zip (nothing is extracted).

    Returns the few columns we need, the value labels, and the weekday labels
    as Stata itself decodes them.
    """
    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".dta")]
        if len(members) != 1:
            raise ValueError(f"Expected exactly one .dta in {zip_path.name}; found {len(members)}.")
        member = members[0]
        raw_bytes = archive.read(member)

    # Data: raw codes only (no label conversion).
    frame = pd.read_stata(
        io.BytesIO(raw_bytes),
        columns=["VDAYR", "ARRTIME", "IMMEDR", "PATWT"],
        convert_categoricals=False,
    )
    frame["ARRTIME"] = frame["ARRTIME"].astype(str).str.strip()
    frame["PATWT"] = frame["PATWT"].astype("float64")

    # Value labels, as the file stores them (keyed by label-set name).
    with pd.io.stata.StataReader(io.BytesIO(raw_bytes)) as reader:
        label_sets = reader.value_labels()

    # Which label set belongs to which variable: let pandas decode the column
    # and pair each code with its text label. This is the check that the
    # "VDAYRF" / "IMMEDRF" label sets really belong to VDAYR / IMMEDR.
    decoded = {}
    for variable in ("VDAYR", "IMMEDR"):
        coded = frame[variable]
        text = pd.read_stata(
            io.BytesIO(raw_bytes), columns=[variable], convert_categoricals=True
        )[variable]
        pairs = pd.DataFrame({"code": coded, "label": text.astype(str)}).drop_duplicates()
        decoded[variable] = {int(c): str(l) for c, l in zip(pairs["code"], pairs["label"])}

    return {
        "member": member,
        "frame": frame,
        "label_sets": label_sets,
        "decoded_labels": decoded,
    }


# --------------------------------------------------------------------------
# Cross-checks against the official file
# --------------------------------------------------------------------------

def hour_from_raw_arrtime(text: str):
    """Hour (0..23) from a raw ARRTIME string like '0604', or None if invalid.

    Codebook rule: military time HHMM, 0000-2359; -9 means blank.
    """
    text = text.strip()
    if re.fullmatch(r"[0-9]{4}", text) and int(text[:2]) <= 23 and int(text[2:]) <= 59:
        return int(text[:2])
    return None


def rows_line_up(cleaned: pd.DataFrame, dta: pd.DataFrame) -> dict:
    """Check the cleaned CSV is the same rows, in the same order, as the .dta."""
    result = {"same_row_count": len(cleaned) == len(dta)}
    if not result["same_row_count"]:
        result["rows_line_up"] = False
        return result
    result["same_VDAYR"] = bool((cleaned["VDAYR"].astype("int64") == dta["VDAYR"].astype("int64")).all())
    result["same_IMMEDR"] = bool((cleaned["IMMEDR"].astype("int64") == dta["IMMEDR"].astype("int64")).all())
    result["same_ARRTIME_text"] = bool((cleaned["ARRTIME"] == dta["ARRTIME"]).all())
    rel = ((cleaned["PATWT"] - dta["PATWT"]).abs() / dta["PATWT"]).max()
    result["max_relative_PATWT_difference"] = float(rel)
    result["same_PATWT"] = bool(rel <= WEIGHT_RTOL)
    result["rows_line_up"] = all(
        result[k] for k in ("same_VDAYR", "same_IMMEDR", "same_ARRTIME_text", "same_PATWT")
    )
    return result


# --------------------------------------------------------------------------
# The four parameter blocks
# --------------------------------------------------------------------------

def build_hourly(cleaned: pd.DataFrame, dta: pd.DataFrame, lined_up: dict) -> dict:
    valid = cleaned["arrival_hour"].notna()
    hours = cleaned.loc[valid, "arrival_hour"]
    if not hours.between(0, 23).all() or not (hours == hours.round()).all():
        raise ValueError("arrival_hour has values that are not whole hours 0..23.")
    weights = cleaned.loc[valid, "PATWT"]
    weighted, plain, counts = weighted_and_plain_shares(hours, weights, HOURS)

    # Independent recount straight from the official file's raw ARRTIME text.
    dta_hours = dta["ARRTIME"].map(hour_from_raw_arrtime)
    dta_valid = dta_hours.notna()
    dta_counts = dta_hours[dta_valid].astype(int).value_counts().reindex(HOURS, fill_value=0).astype(int)
    counts_identical = [int(c) for c in counts] == [int(c) for c in dta_counts]
    invalid_identical = int((~valid).sum()) == int((~dta_valid).sum())
    row_by_row_identical = bool(
        len(cleaned) == len(dta)
        and (cleaned["arrival_hour"].fillna(-1).astype(int).to_numpy()
             == dta_hours.fillna(-1).astype(int).to_numpy()).all()
    )
    dta_weighted, _, _ = weighted_and_plain_shares(
        dta_hours[dta_valid].astype(int), dta.loc[dta_valid, "PATWT"], HOURS
    )
    max_share_diff = float((weighted - dta_weighted).abs().max())
    verified = counts_identical and invalid_identical and row_by_row_identical and lined_up["rows_line_up"]

    value = round_list(weighted)
    check_sums_to_one("hourly_shape.value", value)
    check_sums_to_one("hourly_shape.alt_unweighted", round_list(plain))

    invalid_values = cleaned.loc[~valid, "ARRTIME"].value_counts()
    return {
        "value": value,
        "tag": "DATA",
        "status": "verified" if verified else "unverified",
        "weighting": "PATWT",
        "denominator": int(valid.sum()),
        "source": "ARRTIME (hour of the HHMM arrival time; cleaned column arrival_hour)",
        "note": (
            "Share of ED visits arriving in each clock hour (index 0 = 00:00-00:59). "
            "Weighted by PATWT. National sample, not one hospital. "
            "Visits with a blank or invalid arrival time are left out."
        ),
        "alt_unweighted": round_list(plain),
        "counts_unweighted": [int(c) for c in counts],
        "excluded_invalid": {
            "count": int((~valid).sum()),
            "by_raw_ARRTIME": {str(k): int(v) for k, v in invalid_values.items()},
            "note": "-9 is 'blank' in the codebook.",
        },
        "cross_check": {
            "method": "hour recomputed from raw ARRTIME text in the official .dta (HHMM, 0000-2359)",
            "hourly_counts_identical": counts_identical,
            "invalid_count_identical": invalid_identical,
            "row_by_row_hours_identical": row_by_row_identical,
            "cleaned_csv_lines_up_with_dta": lined_up["rows_line_up"],
            "dta_valid_rows": int(dta_valid.sum()),
            "dta_invalid_rows": int((~dta_valid).sum()),
            "max_abs_diff_weighted_share_vs_dta_weights": max_share_diff,
        },
    }


def build_weekday(cleaned: pd.DataFrame, dta_info: dict, lined_up: dict) -> tuple[dict, dict]:
    """Return (weekday_shares block, monday_factor block)."""
    dta = dta_info["frame"]
    codes = cleaned["VDAYR"]
    if not codes.isin(WEEKDAY_CODES).all():
        raise ValueError("VDAYR has codes outside 1..7.")
    weighted, plain, counts = weighted_and_plain_shares(codes, cleaned["PATWT"], WEEKDAY_CODES)

    dta_counts = dta["VDAYR"].astype(int).value_counts().reindex(WEEKDAY_CODES, fill_value=0).astype(int)
    counts_identical = [int(c) for c in counts] == [int(c) for c in dta_counts]

    # What do the official labels say?
    decoded = dta_info["decoded_labels"]["VDAYR"]
    stored = dta_info["label_sets"].get("VDAYRF")
    stored = {int(k): str(v) for k, v in stored.items()} if stored else None
    labels_agree = stored is None or stored == decoded
    monday_codes = [c for c, text in decoded.items() if text.strip().lower() == "monday"]
    labels_ok = bool(decoded) and labels_agree and len(monday_codes) == 1
    code_labels = {str(c): decoded[c] for c in sorted(decoded)} if decoded else None

    value = round_list(weighted)
    check_sums_to_one("weekday_shares.value", value)
    check_sums_to_one("weekday_shares.alt_unweighted", round_list(plain))
    share_by_code = {c: float(weighted.loc[c]) for c in WEEKDAY_CODES}
    plain_by_code = {c: float(plain.loc[c]) for c in WEEKDAY_CODES}

    verified = labels_ok and counts_identical and lined_up["rows_line_up"]
    weekday_block = {
        "value": {str(c): round(share_by_code[c], DECIMALS) for c in WEEKDAY_CODES},
        "tag": "DATA",
        "status": "verified" if verified else "unverified",
        "weighting": "PATWT",
        "denominator": int(counts.sum()),
        "source": "VDAYR (day of week of visit)",
        "note": (
            "Share of ED visits on each day. Keys are the VDAYR codes exactly as in the file; "
            "'code_labels' gives the meaning read from the official .dta value labels. "
            "Weighted by PATWT. National sample."
        ),
        "alt_unweighted": {str(c): round(plain_by_code[c], DECIMALS) for c in WEEKDAY_CODES},
        "counts_unweighted": {str(c): int(counts.loc[c]) for c in WEEKDAY_CODES},
        "code_labels": code_labels,
        "excluded_invalid": {"count": 0, "note": "every row has a VDAYR code 1..7"},
        "cross_check": {
            "method": "counts per VDAYR code compared with the official .dta; labels decoded by pandas "
                      "and compared with the stored label set VDAYRF",
            "weekday_counts_identical": counts_identical,
            "labels_agree_with_stored_label_set": labels_agree,
            "monday_codes_found": monday_codes,
            "cleaned_csv_lines_up_with_dta": lined_up["rows_line_up"],
        },
    }

    if labels_ok:
        monday_code = monday_codes[0]
        factor = round(share_by_code[monday_code] * 7.0, DECIMALS)
        factor_plain = round(plain_by_code[monday_code] * 7.0, DECIMALS)
        monday_note = (
            f"Monday share divided by an even share (1/7). Monday is VDAYR code {monday_code}, "
            "confirmed from the official .dta value labels. Weighted by PATWT."
        )
    else:
        monday_code, factor, factor_plain = None, None, None
        monday_note = (
            "Value is null because the official .dta value labels do not clearly identify "
            "which VDAYR code is Monday."
        )
    monday_block = {
        "value": factor,
        "tag": "DATA",
        "status": "verified" if verified else "unverified",
        "weighting": "PATWT",
        "denominator": int(counts.sum()),
        "source": "VDAYR (Monday share / (1/7))",
        "note": monday_note,
        "alt_unweighted": factor_plain,
        "monday_code": monday_code,
        "monday_share": round(share_by_code[monday_code], DECIMALS) if labels_ok else None,
    }
    return weekday_block, monday_block


def build_triage(cleaned: pd.DataFrame, dta_info: dict, lined_up: dict) -> dict:
    dta = dta_info["frame"]
    valid = cleaned["IMMEDR_clean"].isin(TRIAGE_LEVELS)
    levels = cleaned.loc[valid, "IMMEDR_clean"]
    weighted, plain, counts = weighted_and_plain_shares(levels, cleaned.loc[valid, "PATWT"], TRIAGE_LEVELS)

    dta_valid = dta["IMMEDR"].isin(TRIAGE_LEVELS)
    dta_counts = dta.loc[dta_valid, "IMMEDR"].astype(int).value_counts().reindex(TRIAGE_LEVELS, fill_value=0).astype(int)
    counts_identical = [int(c) for c in counts] == [int(c) for c in dta_counts]
    cleaned_vs_raw_same = bool((valid == cleaned["IMMEDR"].isin(TRIAGE_LEVELS)).all())

    value = round_list(weighted)
    check_sums_to_one("triage_mix.value", value)
    check_sums_to_one("triage_mix.alt_unweighted", round_list(plain))

    # Rows that are not a valid level 1..5, listed by raw code.
    invalid_codes = cleaned.loc[~valid, "IMMEDR"].astype(int).value_counts().sort_index()
    labels = dta_info["decoded_labels"]["IMMEDR"]
    verified = counts_identical and cleaned_vs_raw_same and lined_up["rows_line_up"]
    return {
        "value": value,
        "tag": "DATA",
        "status": "verified" if verified else "unverified",
        "weighting": "PATWT",
        "denominator": int(valid.sum()),
        "source": "IMMEDR (triage level; valid levels 1..5 only, cleaned column IMMEDR_clean)",
        "note": (
            "Index 0 = level 1 (Immediate) ... index 4 = level 5 (Nonurgent). "
            "Weighted by PATWT. Only visits with a recorded level 1..5 are used; "
            "the rest are left out, not assigned a level. Level 1 has few visits, so its share is uncertain."
        ),
        "alt_unweighted": round_list(plain),
        "counts_unweighted": [int(c) for c in counts],
        "code_labels": {str(c): labels[c] for c in sorted(labels)} if labels else None,
        "excluded_invalid": {
            "count": int((~valid).sum()),
            "by_code": {str(k): int(v) for k, v in invalid_codes.items()},
        },
        "cross_check": {
            "method": "valid-level counts compared with IMMEDR 1..5 in the official .dta",
            "level_counts_identical": counts_identical,
            "dta_level_counts": [int(c) for c in dta_counts],
            "cleaned_csv_lines_up_with_dta": lined_up["rows_line_up"],
        },
    }


def build_context(events_path: Path) -> dict:
    """NOAA heatwave events: context only, never a multiplier."""
    base_note = (
        "CONTEXT only. These NOAA events use a percentile rule with season 'all', so some are "
        "winter events (see events_by_start_month). Never used as a surge size or multiplier; "
        "surge sizes come only from published studies."
    )
    if not events_path.is_file():
        return {"tag": "CONTEXT", "status": "skipped", "note": f"{events_path.name} not found. " + base_note}
    events = pd.read_csv(events_path)
    needed = {"start_date", "duration_days"}
    if not needed.issubset(events.columns):
        return {"tag": "CONTEXT", "status": "skipped",
                "note": f"{events_path.name} lacks columns {sorted(needed)}. " + base_note}
    events["duration_days"] = pd.to_numeric(events["duration_days"], errors="coerce")
    events["start_month"] = pd.to_datetime(events["start_date"], errors="coerce").dt.month

    def summary(subset: pd.DataFrame) -> dict:
        days = subset["duration_days"].dropna()
        return {
            "n_events": int(len(subset)),
            "duration_days_median": float(days.median()) if len(days) else None,
            "duration_days_mean": round(float(days.mean()), 4) if len(days) else None,
        }

    by_month = events["start_month"].value_counts().sort_index()
    return {
        "tag": "CONTEXT",
        "status": "computed",
        "source": f"NOAA GHCN-Daily heatwave events ({events_path.name})",
        "stations": sorted(events["station_id"].dropna().unique().tolist()) if "station_id" in events else None,
        "definitions": sorted(events["definition"].dropna().unique().tolist()) if "definition" in events else None,
        "all_events": summary(events),
        "events_starting_jun_to_sep": summary(events[events["start_month"].between(6, 9)]),
        "events_by_start_month": {str(int(m)): int(n) for m, n in by_month.items()},
        "note": base_note,
    }


# --------------------------------------------------------------------------
# Project-notes comparison table
# --------------------------------------------------------------------------

def compare_row(name: str, notes_number, computed, note: str) -> dict:
    """One row of the project-notes check. computed=None means 'not checkable'."""
    if computed is None:
        return {"name": name, "notes_value": notes_number, "computed_value": None,
                "relative_difference": None, "pass_within_10pct": None, "note": note}
    relative = (computed - notes_number) / notes_number
    return {
        "name": name,
        "notes_value": notes_number,
        "computed_value": computed,
        "relative_difference": round(relative, 6),
        "pass_within_10pct": bool(abs(relative) <= PASS_LIMIT),
        "note": note,
    }


def build_notes_check(hourly: dict, weekday: dict, triage: dict) -> list:
    rows = []
    shares = hourly["value"]
    peak_hour = max(HOURS, key=lambda h: shares[h])
    quiet_hour = min(HOURS, key=lambda h: shares[h])
    rows.append(compare_row(
        "peak_hour_share", NOTES_VALUES["peak_hour_share"], round(shares[peak_hour], 6),
        f"Actual peak hour is {peak_hour}:00 (the project notes say 10 AM). PATWT-weighted.",
    ))
    rows.append(compare_row(
        "hour_10_share", NOTES_VALUES["hour_10_share"], round(shares[10], 6),
        "Share at 10 AM, because the project notes name 10 AM as the peak. PATWT-weighted.",
    ))
    rows.append(compare_row(
        "quietest_hour_share", NOTES_VALUES["quietest_hour_share"], round(shares[quiet_hour], 6),
        f"Actual quietest hour is {quiet_hour}:00. PATWT-weighted.",
    ))

    # Weekday rows: only checkable if the codes were confirmed from the labels.
    labels = weekday.get("code_labels") or {}
    code_for = {text.strip().lower(): code for code, text in labels.items()}
    confirmed = weekday["status"] == "verified"
    for name, day in (("monday_share", "monday"), ("saturday_share", "saturday"), ("sunday_share", "sunday")):
        if confirmed and day in code_for:
            code = code_for[day]
            rows.append(compare_row(
                name, NOTES_VALUES[name], round(weekday["value"][code], 6),
                f"{day.title()} is VDAYR code {code} (from the .dta value labels). PATWT-weighted.",
            ))
        else:
            rows.append(compare_row(name, NOTES_VALUES[name], None,
                                    "not checkable: weekday code unverified"))

    for level, notes_count, computed_count in zip(
        TRIAGE_LEVELS, NOTES_VALUES["triage_counts"], triage["counts_unweighted"]
    ):
        rows.append(compare_row(
            f"triage_level_{level}_count", notes_count, int(computed_count),
            "Visits with a valid recorded level (unweighted count).",
        ))
    return rows


def format_value(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.4f}"


def print_check_table(rows: list) -> None:
    header = ["name", "project notes", "computed", "rel_diff", "pass_10pct", "note"]
    table = [header]
    for row in rows:
        rel = row["relative_difference"]
        table.append([
            row["name"],
            format_value(row["notes_value"]),
            format_value(row["computed_value"]),
            "n/a" if rel is None else f"{rel * 100:+.1f}%",
            "n/a" if row["pass_within_10pct"] is None else ("PASS" if row["pass_within_10pct"] else "FAIL"),
            row["note"],
        ])
    widths = [max(len(r[i]) for r in table) for i in range(5)]
    for index, cells in enumerate(table):
        line = "  ".join(cells[i].ljust(widths[i]) for i in range(5)) + "  " + cells[5]
        print(line)
        if index == 0:
            print("-" * min(len(line), 140))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Extract baseline scenario parameters from NHAMCS 2022.")
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED, help="cleaned NHAMCS 2022 CSV (default: %(default)s)")
    parser.add_argument("--dta-zip", type=Path, default=DEFAULT_DTA_ZIP, help="official Stata zip (default: %(default)s)")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS, help="NOAA heatwave events CSV, optional (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON file to write (default: %(default)s)")
    args = parser.parse_args(argv)

    for label, path in (("cleaned CSV", args.cleaned), ("official .dta zip", args.dta_zip)):
        if not path.is_file():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    cleaned = load_cleaned(args.cleaned)
    dta_info = load_official_dta(args.dta_zip)
    dta = dta_info["frame"]
    lined_up = rows_line_up(cleaned, dta)

    hourly = build_hourly(cleaned, dta, lined_up)
    weekday, monday = build_weekday(cleaned, dta_info, lined_up)
    triage = build_triage(cleaned, dta_info, lined_up)
    context = build_context(args.events)
    check = build_notes_check(hourly, weekday, triage)

    result = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": "NHAMCS 2022 Emergency Department public-use file (national sample)",
            "file": args.dta_zip.name,
            "dta_member": dta_info["member"],
            "sha256": sha256_of_file(args.dta_zip),
            "records": int(len(dta)),
            "cleaned_file": args.cleaned.name,
            "cleaned_sha256": sha256_of_file(args.cleaned),
            "computed_on": date.today().isoformat(),
        },
        "hourly_shape": hourly,
        "weekday_shares": weekday,
        "monday_factor": monday,
        "triage_mix": triage,
        "notes_check": check,
        "context": context,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")

    # ---- plain-language summary ----
    print(f"Rows in cleaned CSV: {len(cleaned):,}   rows in official .dta: {len(dta):,}   "
          f"rows line up: {lined_up['rows_line_up']}")
    print(f"Hourly: {hourly['denominator']:,} valid rows, {hourly['excluded_invalid']['count']} invalid "
          f"arrival times left out, status {hourly['status']}")
    print(f"Weekday: labels {weekday['code_labels']}, status {weekday['status']}, "
          f"Monday factor {monday['value']}")
    print(f"Triage: {triage['denominator']:,} valid rows, counts {triage['counts_unweighted']}, "
          f"status {triage['status']}")
    print(f"Context: {context.get('status')}")
    print()
    print_check_table(check)
    failed = [r["name"] for r in check if r["pass_within_10pct"] is False]
    print()
    print(f"Project-notes numbers outside 10 percent: {failed if failed else 'none'}")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
