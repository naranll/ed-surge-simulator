"""Independent offline source audit. Does not import acquisition/curation helpers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import pandas as pd

ENTITY = 'DisasterDeclarationsSummaries'
DATE_FIELDS = ['declarationDate', 'incidentBeginDate', 'incidentEndDate', 'disasterCloseoutDate']
ROOT = Path(__file__).resolve().parents[2]
LIMITATIONS = [
    'Only explicit TX/county-201 API records were requested. Statewide and other-area events were not acquired.',
    'Count reconciliation is not a transactional API snapshot or a guarantee against concurrent edits.',
    'Incident periods describe declared incidents, not continuous hospital exposure.',
    'Repeated incident IDs, matching titles and overlapping dates are review candidates, not automatic merges.',
    'Historical missing county designations and non-declared disasters limit completeness.',
    'No HURDAT2, ED multipliers, simulation scenarios or patient-level joins are implemented.']


def missing(value):
    return value is None or (isinstance(value, str) and not value.strip())


def geography(record):
    """Conservative mutually exclusive classification; retain original strings."""
    area = str(record.get('designatedArea') or '').strip().casefold()
    if record.get('tribalRequest') is True or any(x in area for x in ['tribe', 'tribal', 'reservation']):
        return 'tribal_review'
    if record.get('fipsCountyCode') == '000' or 'statewide' in area or 'state wide' in area:
        return 'statewide_review'
    if (record.get('state') == 'TX' and record.get('fipsStateCode') == '48'
            and record.get('fipsCountyCode') == '201' and area == 'harris (county)'
            and record.get('tribalRequest') is False):
        return 'explicit_harris_county'
    return 'unresolved_or_other_area'


def audit_records(records):
    """Describe source records without dropping, coercing or merging them."""
    rows = list(records)
    ids = [r.get('id') for r in rows]
    def labels(field):
        return dict(Counter('<missing>' if missing(r.get(field)) else str(r[field]) for r in rows))
    def groups(field):
        grouped = defaultdict(list)
        for i, r in enumerate(rows):
            if not missing(r.get(field)):
                grouped[str(r[field])].append(i)
        return grouped
    def evidence(indices):
        return [{k: rows[i].get(k) for k in ['id','femaDeclarationString','disasterNumber','incidentId',
                'declarationTitle','incidentBeginDate','incidentEndDate','declarationDate']} for i in indices]
    dates, parsed = {}, {}
    for field in DATE_FIELDS:
        values = [r.get(field) for r in rows]
        parsed[field] = [pd.to_datetime(v, utc=True, errors='coerce') if not missing(v) else pd.NaT for v in values]
        valid = [v for v in parsed[field] if not pd.isna(v)]
        dates[field] = {'missing_count': sum(missing(v) for v in values),
                        'invalid_nonmissing_count': sum(not missing(v) and pd.isna(d) for v,d in zip(values,parsed[field])),
                        'minimum': min(valid).isoformat() if valid else None,
                        'maximum': max(valid).isoformat() if valid else None}
    issues = defaultdict(list)
    for i, r in enumerate(rows):
        begin, end, declared = (parsed[f][i] for f in ['incidentBeginDate','incidentEndDate','declarationDate'])
        if pd.isna(end): issues['missing_or_invalid_incident_end'].append(i)
        if not pd.isna(begin) and not pd.isna(end) and end < begin:
            issues['end_before_begin'].append(i)
        if not pd.isna(begin) and not pd.isna(declared):
            if declared < begin: issues['declaration_before_incident_begin'].append(i)
            if declared > begin: issues['declaration_after_incident_begin'].append(i)
        if not pd.isna(end) and not pd.isna(declared) and declared > end:
            issues['declaration_after_incident_end'].append(i)
    incident_groups = groups('incidentId')
    repeated_incidents = {k:evidence(v) for k,v in incident_groups.items() if len(v)>1}
    titles = defaultdict(list)
    for i,r in enumerate(rows):
        if missing(r.get('declarationTitle')): continue
        title = re.sub(r'\s+', ' ', str(r['declarationTitle']).strip().upper())
        begin=parsed['incidentBeginDate'][i]
        if not pd.isna(begin): titles[(title,begin.year)].append(i)
    repeated_titles = {f'{k[1]} | {k[0]}':evidence(v) for k,v in titles.items()
                       if len({rows[i].get('femaDeclarationString') for i in v})>1}
    overlaps = []
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            b1,e1,b2,e2 = parsed['incidentBeginDate'][i],parsed['incidentEndDate'][i],parsed['incidentBeginDate'][j],parsed['incidentEndDate'][j]
            if not any(pd.isna(v) for v in [b1,e1,b2,e2]) and b1<=e1 and b2<=e2 and max(b1,b2)<=min(e1,e2):
                overlaps.append({'source_ids':[ids[i],ids[j]], 'declarations':[rows[i].get('femaDeclarationString'),rows[j].get('femaDeclarationString')],
                                 'same_nonmissing_incident_id': not missing(rows[i].get('incidentId')) and rows[i].get('incidentId')==rows[j].get('incidentId')})
    geo = defaultdict(list)
    for i,r in enumerate(rows): geo[geography(r)].append(i)
    incident_consistency = {}
    for key, indices in incident_groups.items():
        differences={f:sorted({str(rows[i].get(f)) for i in indices}) for f in ['incidentBeginDate','incidentEndDate','incidentType']}
        if any(len(v)>1 for v in differences.values()): incident_consistency[key]=differences
    return {'records':len(rows), 'unique_nonmissing_source_ids':len(groups('id')),
            'missing_source_ids':sum(missing(i) for i in ids),
            'duplicate_source_ids':{k:evidence(v) for k,v in groups('id').items() if len(v)>1},
            'repeated_declaration_numbers':{k:evidence(v) for k,v in groups('disasterNumber').items() if len(v)>1},
            'unique_declaration_numbers':len(groups('disasterNumber')),
            'exact_duplicate_records':len(rows)-len({json.dumps(r,sort_keys=True) for r in rows}),
            'source_field_types':{k:sorted({type(r.get(k)).__name__ for r in rows}) for k in sorted({k for r in rows for k in r})},
            'missingness':{k:sum(missing(r.get(k)) for r in rows) for k in sorted({k for r in rows for k in r})},
            'dates':dates, 'date_relationships':{k:{'count':len(issues[k]),'records':evidence(issues[k])} for k in
                ['missing_or_invalid_incident_end','end_before_begin','declaration_before_incident_begin','declaration_after_incident_begin','declaration_after_incident_end']},
            'geography':{k:{'count':len(geo[k]),'source_ids':[ids[i] for i in geo[k]]} for k in
                ['explicit_harris_county','statewide_review','tribal_review','unresolved_or_other_area']},
            'geographic_values':{k:labels(k) for k in ['state','fipsStateCode','fipsCountyCode','placeCode','designatedArea','tribalRequest']},
            'incident_types':labels('incidentType'), 'declaration_types':labels('declarationType'),
            'incident_id_missing':sum(missing(r.get('incidentId')) for r in rows),
            'unique_incident_ids':len(incident_groups), 'repeated_incident_ids':repeated_incidents,
            'incident_group_inconsistencies':incident_consistency,
            'same_title_year_candidates':repeated_titles,'overlapping_closed_period_pairs':overlaps,
            'limitations':LIMITATIONS}


def audit_snapshot(manifest_path):
    """Re-read bytes and metadata independently; incomplete captures remain explicit."""
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    records, checks, hashes, counts = [], [], {}, []
    offset = 0
    def check(name, condition, evidence):
        checks.append({'check':name, 'status':'PASS' if condition else 'FAIL', 'evidence':evidence})
    for index, page in enumerate(manifest['pages']):
        path=(manifest_path.parent/page['file']).resolve()
        if path.parent != manifest_path.parent:
            raise ValueError('Page path escapes snapshot directory')
        body=path.read_bytes(); digest=hashlib.sha256(body).hexdigest(); hashes[page['file']]=digest
        check(f'Page {index+1} hash',digest==page['sha256'],digest)
        obj=json.loads(body); batch=obj[ENTITY]; meta=obj['metadata']; counts.append(meta.get('count'))
        check(f'Page {index+1} offset/filter',meta.get('skip')==offset and meta.get('filter')=="state eq 'TX' and fipsCountyCode eq '201'", {'offset':offset,'returned':len(batch)})
        records.extend(batch); offset+=len(batch)
    probe=manifest.get('count_probe')
    probe_count=None
    if probe:
        path=(manifest_path.parent/probe['file']).resolve()
        if path.parent != manifest_path.parent: raise ValueError('Probe path escapes snapshot directory')
        body=path.read_bytes(); digest=hashlib.sha256(body).hexdigest(); hashes[probe['file']]=digest
        check('Count probe hash',digest==probe['sha256'],digest)
        obj=json.loads(body); probe_count=obj.get('metadata',{}).get('count')
        check('Count probe scope',obj.get('metadata',{}).get('filter')=="state eq 'TX' and fipsCountyCode eq '201'" and obj.get('metadata',{}).get('skip')==0,'Same county filter and zero offset')
    ids=[r.get('id') for r in records]
    check('Independent count reconciliation', bool(counts) and all(type(n) is int and n==len(records) for n in counts)
          and probe_count==len(records) and manifest.get('retrieved_records')==len(records),
          {'page_totals':counts,'records':len(records),'final_probe_count':probe_count})
    check('Unique populated source IDs', all(isinstance(i,str) and bool(i.strip()) for i in ids) and len(set(ids))==len(ids),len(set(ids)))
    check('Acquisition marked complete',manifest['status']=='complete',manifest['status'])
    check('Source geographic types',all(isinstance(r.get(f),str) for r in records for f in ['fipsStateCode','fipsCountyCode','placeCode']), 'No geographic numeric coercion')
    report=audit_records(records)
    check('Filter returned expected geography',all(r.get('state')=='TX' and r.get('fipsCountyCode')=='201' for r in records),'TX and source county string 201')
    check('Valid incident intervals',report['date_relationships']['end_before_begin']['count']==0,'End-before-begin count')
    report.update(snapshot_manifest=str(manifest_path), snapshot_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  raw_response_hashes=hashes, independent_checks=checks, acquisition_attempts=manifest['attempts'],
                  retrieval_started_utc=manifest['started_utc'], retrieval_finished_utc=manifest['finished_utc'])
    report['audit_implementation']={'python':platform.python_version(),'pandas':pd.__version__,
                                    'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report['warnings']=[
        f"{len(report['repeated_incident_ids'])} repeated incidentId groups need review, not merging.",
        f"{len(report['overlapping_closed_period_pairs'])} overlapping closed-period pairs are not evidence of event identity.",
        'No statewide/tribal/unresolved coverage outside the explicit county query was assessed.']
    if report['incident_id_missing']:
        report['warnings'].append(f"{report['incident_id_missing']} records lack incidentId; absence does not establish separate physical events.")
    if report['date_relationships']['missing_or_invalid_incident_end']['count']:
        report['warnings'].append(f"{report['date_relationships']['missing_or_invalid_incident_end']['count']} incident ends missing/invalid; no duration filled.")
    return report


def markdown(report):
    r=report
    lines=['# OpenFEMA Harris County independent source audit','',
           f"Snapshot: `{r['snapshot_manifest']}`",'',
           f"Records: **{r['records']}**; unique source IDs: **{r['unique_nonmissing_source_ids']}**; unique declaration numbers: **{r['unique_declaration_numbers']}**.",
           f"Retrieval: {r['retrieval_started_utc']} to {r['retrieval_finished_utc']}. Request attempts: {len(r['acquisition_attempts'])}; failed attempts: {sum(a['status']=='failed' for a in r['acquisition_attempts'])}.",
           '', '## Independent integrity checks','', '| Check | Status | Evidence |','|---|---|---|']
    lines += [f"| {x['check']} | {x['status']} | {x['evidence']} |" for x in r['independent_checks']]
    lines += ['', '## Date coverage and missingness','', '| Field | Minimum | Maximum | Missing | Invalid |','|---|---|---|---:|---:|']
    lines += [f"| {k} | {v['minimum']} | {v['maximum']} | {v['missing_count']} | {v['invalid_nonmissing_count']} |" for k,v in r['dates'].items()]
    lines += ['', '## Verified source distributions','',f"- Geographic partitions: { {k:v['count'] for k,v in r['geography'].items()} }",
              f"- Geographic values: {r['geographic_values']}",f"- Incident types: {r['incident_types']}",
              f"- Declaration types: {r['declaration_types']}",f"- Duplicate source IDs: {len(r['duplicate_source_ids'])}; exact duplicate records: {r['exact_duplicate_records']}.",
              f"- Repeated declaration numbers: {list(r['repeated_declaration_numbers'])}.",
              f"- Nonmissing unique incident IDs: {r['unique_incident_ids']}; missing incident IDs: {r['incident_id_missing']}."]
    lines += ['', '## Date relationship diagnostics','']
    lines += [f"- {k}: {v['count']} records." for k,v in r['date_relationships'].items()]
    lines += ['', 'Declaration-after-begin/end is not inherently an error. Early declarations can also be valid; review the source rather than correcting dates.',
              '', '## Unresolved grouping candidates','']
    for key,group in r['repeated_incident_ids'].items():
        lines.append(f"- incidentId {key}: "+'; '.join(f"{x['femaDeclarationString']} ({x['declarationTitle']}; {x['incidentBeginDate']} through {x['incidentEndDate']})" for x in group))
    lines += [f"- Same-title/year candidate groups: {list(r['same_title_year_candidates'])}.",
              f"- Incident-group field differences: {r['incident_group_inconsistencies']}.",
              f"- Overlapping incident-period pairs: {len(r['overlapping_closed_period_pairs'])}; full identifiers are in the JSON report.",
              '', '## Warnings',''] + ['- '+x for x in r['warnings']]
    lines += ['', '## Recommendations for later review','',
              '- Retain declaration-area rows and source IDs. Do not silently deduplicate declaration numbers.',
              '- Admit explicit Harris records only after geography validation; keep other designations in separate review partitions.',
              '- Preserve missing incident ends. Do not substitute declaration/closeout dates or infer continuous exposure.',
              '- Review shared incident IDs with titles, dates and official event evidence. Create a sourced crosswalk only after approval.',
              '- Treat overlap and matching generic titles as candidate signals only; hazards may coexist.',
              '- Audit broader statewide coverage separately before claiming a complete Houston event catalogue.',
              '', '## Limitations',''] + ['- '+x for x in LIMITATIONS]
    lines += ['', 'Official source: https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries',
              'Source documentation: https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2',
              '', 'The audit is offline and independent of acquisition helpers. Raw responses and all input values remain unchanged. No curated catalogue was generated.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True,help='New report directory; no overwrites')
    args=parser.parse_args()
    report=audit_snapshot(args.manifest)
    args.report_dir.mkdir(parents=True,exist_ok=False)
    (args.report_dir/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (args.report_dir/'audit.md').write_text(markdown(report),encoding='utf-8')
    failures=[c for c in report['independent_checks'] if c['status']=='FAIL']
    print(json.dumps({'records':report['records'],'checks':dict(Counter(c['status'] for c in report['independent_checks'])),
                      'warnings':report['warnings'],'report':str(args.report_dir/'audit.md')},indent=2))
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
