"""Compute a small NHAMCS parameter contract from a raw third-party CSV.

Standalone standard-library implementation. No prior preprocessing modules used.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

REFERENCE_SHA256 = '439a87b766cba5e8c7b838372f45b680ba29295b148078b50e8ab7056d20349b'
REQUIRED = ('arrival_time', 'target_triage_acuity')


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def integer_token(token: str) -> tuple[int | None, str]:
    if not token.strip():
        return None, 'missing'
    try:
        value = Decimal(token.strip())
    except InvalidOperation:
        return None, 'invalid'
    if not value.is_finite() or value != value.to_integral_value():
        return None, 'invalid'
    # Bound before conversion to avoid pathological numeric tokens.
    if not 0 <= value <= 2359:
        return None, 'invalid'
    return int(value), 'valid'


def arrival_hour(token: str) -> tuple[int | None, str]:
    value, status = integer_token(token)
    if status != 'valid':
        return None, status
    hour, minute = divmod(value, 100)
    return (hour, 'valid') if hour < 24 and minute < 60 else (None, 'invalid')


def distribution(counts: list[int], excluded: dict[str, int], total: int) -> dict:
    denominator = sum(counts)
    if denominator == 0:
        raise ValueError('Cannot export probabilities with zero valid records')
    if denominator + sum(excluded.values()) != total:
        raise ValueError('Record reconciliation failed')
    probabilities = [n / denominator for n in counts]
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError('Probability normalization failed')
    return dict(counts=counts, proportions=probabilities, denominator=denominator,
                excluded_records=sum(excluded.values()), excluded_by_reason=excluded)


def calculate(data: bytes, filename: str) -> dict:
    reader = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    try:
        columns = next(reader)
    except StopIteration:
        raise ValueError('Empty CSV') from None
    if len(columns) != len(set(columns)) or any(not name for name in columns):
        raise ValueError('CSV has empty or duplicate column names')
    if not set(REQUIRED).issubset(columns):
        raise ValueError('Required columns: ' + ', '.join(REQUIRED))
    ai, ti = (columns.index(name) for name in REQUIRED)
    hours, triage = [0] * 24, [0] * 5
    arrival_exclusions = dict(missing=0, invalid=0)
    triage_exclusions = dict(missing=0, invalid=0)
    total = 0
    for row in reader:
        if len(row) != len(columns):
            raise ValueError(f'Malformed CSV record {total + 1}: unexpected field count')
        total += 1
        hour, status = arrival_hour(row[ai])
        if status == 'valid':
            hours[hour] += 1
        else:
            arrival_exclusions[status] += 1
        code, status = integer_token(row[ti])
        if status == 'valid' and code not in range(1, 6):
            status = 'invalid'
        if status == 'valid':
            triage[code - 1] += 1
        else:
            triage_exclusions[status] += 1
    arrivals = dict(hours=list(range(24)), **distribution(hours, arrival_exclusions, total))
    acuity = dict(codes=list(range(1, 6)), **distribution(triage, triage_exclusions, total))
    source = dict(dataset='Third-party processed NHAMCS 2018-2022', file=filename,
                  sha256=digest(data), records=total, column_count=len(columns), columns=columns,
                  provenance_status='third_party_transformation_not_fully_documented',
                  original_download_url=None, weighting='unweighted_sample',
                  pooling='all supplied rows and years; duplicate records retained')
    missing = {
        'provider_service_times': 'No verified service-time field; waiting time is not provider service time.',
        'admission_probabilities': 'No verified admission outcome field.',
        'boarding_durations': 'No verified boarding duration field.',
        'hospital_capacity': 'No hospital capacity field.',
    }
    return {
        'schema_version': '1.0', 'source': source,
        'hourly_arrivals': arrivals, 'triage': acuity,
        'hourly_shape': dict(status='verified', tag='DATA', value=arrivals['proportions'],
                             weighting='unweighted_sample', denominator=arrivals['denominator'],
                             source=filename, note='Verified arithmetic and source reconciliation only; index is hour 0-23.'),
        'triage_mix': dict(status='verified', tag='DATA', value=acuity['proportions'],
                          weighting='unweighted_sample', denominator=acuity['denominator'],
                          source=filename, note='Verified arithmetic and source reconciliation only; index is code minus one, levels 1-5.'),
        'unavailable_parameters': {key: dict(value=None, status='unavailable_from_source',
                                            owner='Simulation Team', reason=reason)
                                   for key, reason in missing.items()},
        'limitations': [
            'Unweighted sample proportions, not observed Houston arrival rates or absolute hourly volumes.',
            'Third-party transformation provenance is incomplete; records appear restricted to valid triage levels.',
            'No survey weights, hospital identifiers or complete patient arrival dates are supplied.',
            'Weekday labels are unverified; no named weekday factors are exported.',
            'Triage codes retain source numbering; clinical label equivalence is not established.',
            'Missing/invalid arrival values affect only the arrival denominator; triage has its own exclusions.',
            'Duplicate records are retained; no patient records are deleted, imputed or joined to weather/disaster data.',
            'Baseline volume, clinical durations, capacity and scenario effects require separate Simulation Team decisions.',
            'Replacing a PATWT-weighted baseline with this sample requires Simulation Team acceptance.',
        ],
    }


def verify_reference(data: bytes, params: dict) -> dict:
    """Independent count check using raw tokens and Counter, for the known snapshot."""
    if digest(data) != REFERENCE_SHA256:
        raise ValueError('Reference validation requires the independently verified snapshot hash')
    rows = list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline='')))
    # Known snapshot: independent integer-string conversion, no production parser.
    times = [int(r['arrival_time']) for r in rows if r['arrival_time'].strip()]
    if any(t < 0 or t // 100 > 23 or t % 100 > 59 for t in times):
        raise ValueError('Unexpected arrival token in reference snapshot')
    hc = Counter(t // 100 for t in times)
    tc = Counter(int(r['target_triage_acuity']) for r in rows)
    h, t = params['hourly_arrivals'], params['triage']
    checks = {
        'shape_58124_by_49': len(rows) == 58124 and len(rows[0]) == 49,
        'arrival_denominator': h['denominator'] == len(times) == 57113,
        'arrival_exclusions': h['excluded_by_reason'] == {'missing': 1011, 'invalid': 0},
        'all_hour_counts': h['counts'] == [hc[i] for i in range(24)],
        'peak_hour': max(hc, key=hc.get) == 11 and hc[11] == 3488,
        'triage_denominator': t['denominator'] == 58124 and t['excluded_records'] == 0,
        'all_triage_counts': t['counts'] == [tc[i] for i in range(1, 6)] == [846, 8597, 29568, 16715, 2398],
        'probabilities': all(d['proportions'] == [n / d['denominator'] for n in d['counts']]
                             and math.isclose(math.fsum(d['proportions']), 1, abs_tol=1e-12)
                             for d in [h, t]),
    }
    if not all(checks.values()):
        raise ValueError(f'Reference reconciliation failed: {checks}')
    return checks


def export(source: Path, destination: Path, reference: bool = False) -> dict:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError('Output cannot overwrite source')
    if destination.name != 'params.json':
        raise ValueError('Use a separate params.json; do not overwrite a simulator baseline')
    if destination.exists():
        previous = json.loads(destination.read_text(encoding='utf-8'))
        if not isinstance(previous, dict) or not {'hourly_arrivals', 'triage', 'unavailable_parameters'}.issubset(previous):
            raise ValueError('Refusing to overwrite an unrelated JSON file')
    data = source.read_bytes()
    params = calculate(data, source.name)
    if reference:
        verify_reference(data, params)
    if digest(source.read_bytes()) != params['source']['sha256']:
        raise ValueError('Source changed during processing')
    encoded = (json.dumps(params, indent=2, allow_nan=False) + '\n').encode('utf-8')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('outputs/params.json'))
    parser.add_argument('--verify-reference', action='store_true')
    parser.add_argument('--evidence', action='store_true', help='Run focused tests and save sanitized execution evidence')
    args = parser.parse_args()
    started = datetime.now(timezone.utc).isoformat()
    params = export(args.input, args.output, args.verify_reference)
    if args.evidence:
        here = Path(__file__).resolve().parent
        run = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'test_export_parameters', '-v'],
                             cwd=here, capture_output=True, text=True)
        if run.returncode:
            raise SystemExit('Tests failed; inspect with python -B -m unittest test_export_parameters -v')
        original = args.output.read_bytes()
        export(args.input, args.output, args.verify_reference)
        if original != args.output.read_bytes():
            raise SystemExit('Repeat export differs')
        evidence = {
            'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
            'system': platform.system(), 'architecture': platform.machine(),
            'uname_m': subprocess.check_output(['uname', '-m'], text=True).strip() if platform.system() == 'Linux' else None,
            'python': platform.python_version(),
            'source_sha256': params['source']['sha256'], 'output_sha256': digest(original),
            'source_unchanged': digest(args.input.read_bytes()) == params['source']['sha256'],
            'repeat_export_identical': True,
            'reference_checks': verify_reference(args.input.read_bytes(), params) if args.verify_reference else None,
            'tests': {'exit_code': run.returncode, 'summary': run.stderr.strip().splitlines()[-3:]},
            'code_sha256': {name: digest((here / name).read_bytes()) for name in ['export_parameters.py', 'test_export_parameters.py', 'nhamcs_preprocessing.ipynb']},
            'eligibility': 'Execution evidence only; does not establish event eligibility.',
        }
        (args.output.parent / 'execution_evidence.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'records': params['source']['records'],
                      'valid_arrivals': params['hourly_arrivals']['denominator'],
                      'valid_triage': params['triage']['denominator'],
                      'output_sha256': digest(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
