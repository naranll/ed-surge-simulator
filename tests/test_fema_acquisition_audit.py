"""Small synthetic fixtures only; no test contacts FEMA."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from src.data_processing.disaster_data import acquire, fetch_bytes, FILTER, ENTITY
from src.data_processing.fema_audit import audit_records, audit_snapshot, geography


def record(i, **changes):
    row={'id':f'id-{i:03}', 'disasterNumber':i, 'femaDeclarationString':f'DR-{i}-TX',
         'state':'TX', 'fipsStateCode':'48', 'fipsCountyCode':'201', 'placeCode':'99201',
         'designatedArea':'Harris (County)', 'tribalRequest':False,
         'incidentId':'incident-a', 'incidentType':'Hurricane', 'declarationType':'DR',
         'declarationTitle':'SYNTHETIC STORM', 'incidentBeginDate':'2020-01-01T00:00:00.000Z',
         'incidentEndDate':'2020-01-03T00:00:00.000Z', 'declarationDate':'2020-01-02T00:00:00.000Z',
         'disasterCloseoutDate':None, 'extra_field':{'preserve':[1,None,'001']}}
    row.update(changes)
    return row


def fake_api(rows, mutate=None):
    def fetch(url, timeout):
        p=parse_qs(urlsplit(url).query); skip=int(p['$skip'][0]); top=int(p['$top'][0])
        obj={'metadata':{'skip':skip,'filter':FILTER,'count':len(rows)}, ENTITY:rows[skip:skip+top]}
        if mutate: mutate(obj,skip,top)
        return json.dumps(obj,indent=1).encode()
    return fetch


class FEMACollectionTests(unittest.TestCase):
    def test_pagination_lossless_records_offline_audit_and_no_overwrite(self):
        rows=[record(i) for i in range(5)]; original=copy.deepcopy(rows)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'snapshot'
            m=acquire(dest,page_size=2,fetch=fake_api(rows))
            self.assertEqual(m['status'],'complete')
            self.assertEqual([p['record_count'] for p in m['pages']],[2,2,1])
            self.assertEqual(m['unique_source_ids'],5)
            restored=[r for p in m['pages'] for r in json.loads((dest/p['file']).read_bytes())[ENTITY]]
            self.assertEqual(restored,original)
            self.assertIsInstance(restored[0]['fipsCountyCode'],str)
            with patch('urllib.request.urlopen',side_effect=AssertionError('network forbidden')):
                report=audit_snapshot(dest/'manifest.json')
            self.assertTrue(all(x['status']=='PASS' for x in report['independent_checks']))
            with self.assertRaises(FileExistsError): acquire(dest,fetch=fake_api(rows))
        self.assertEqual(rows,original)

    def test_retry_success_and_bounded_exhaustion(self):
        calls=[]; sleeps=[]
        def flaky(url,timeout):
            calls.append(url)
            if len(calls)==1: raise TimeoutError('synthetic timeout')
            return fake_api([record(1)])(url,timeout)
        with tempfile.TemporaryDirectory() as tmp:
            m=acquire(Path(tmp)/'retry',fetch=flaky,retries=1,sleep=sleeps.append)
            self.assertEqual(m['status'],'complete'); self.assertEqual(sleeps,[1])
            self.assertEqual(sum(x['status']=='failed' for x in m['attempts']),1)
            def failed(url,timeout): raise TimeoutError('synthetic unavailable')
            m=acquire(Path(tmp)/'failed',fetch=failed,retries=2,sleep=lambda n:None)
            self.assertEqual(m['status'],'incomplete'); self.assertEqual(len(m['attempts']),3)
            self.assertTrue((Path(tmp)/'failed/manifest.json').is_file())
            report=audit_snapshot(Path(tmp)/'failed/manifest.json')
            self.assertTrue(any(x['status']=='FAIL' for x in report['independent_checks']))

    def test_count_drift_empty_page_duplicates_and_geography_rejected(self):
        cases=[]
        def drift(obj,skip,top):
            if skip: obj['metadata']['count']+=1
        def empty(obj,skip,top):
            if skip: obj[ENTITY]=[]
        cases.extend([fake_api([record(i) for i in range(3)],drift),fake_api([record(i) for i in range(3)],empty),
                      fake_api([record(1),record(1)]),fake_api([record(1,fipsCountyCode='000')]),
                      fake_api([record(1,fipsStateCode=48)])])
        with tempfile.TemporaryDirectory() as tmp:
            for i,fetch in enumerate(cases):
                with self.subTest(i=i):
                    m=acquire(Path(tmp)/str(i),page_size=2,fetch=fetch)
                    self.assertEqual(m['status'],'incomplete'); self.assertIn('error',m)
                    self.assertTrue(m['pages'])

    def test_final_count_change_and_tamper_detected(self):
        def changed(obj,skip,top):
            if top==1: obj['metadata']['count']=4
        with tempfile.TemporaryDirectory() as tmp:
            m=acquire(Path(tmp)/'drift',page_size=2,fetch=fake_api([record(1),record(2)],changed))
            self.assertEqual(m['status'],'incomplete')
            dest=Path(tmp)/'good'
            acquire(dest,fetch=fake_api([record(1)]))
            page=dest/'page_0001.json'; page.write_bytes(page.read_bytes()+b' ')
            r=audit_snapshot(dest/'manifest.json')
            self.assertTrue(any(x['status']=='FAIL' and 'hash' in x['check'] for x in r['independent_checks']))

    def test_malformed_response_preserved_and_total_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/'bad'
            m=acquire(dest,fetch=lambda u,t:b'not JSON')
            self.assertEqual(m['status'],'incomplete')
            self.assertEqual((dest/'page_0001.json').read_bytes(),b'not JSON')
        with patch('src.data_processing.disaster_data.subprocess.run',side_effect=subprocess.TimeoutExpired('fake',4)):
            with self.assertRaises(TimeoutError): fetch_bytes('https://example.invalid',2)


class FEMAAuditTests(unittest.TestCase):
    def test_geography_partition_keeps_broad_tribal_and_unresolved_separate(self):
        rows=[record(1),record(2,fipsCountyCode='000',designatedArea='Statewide'),
              record(3,tribalRequest=True),record(4,designatedArea='Unclear'),record(5,fipsStateCode=48)]
        self.assertEqual([geography(r) for r in rows],['explicit_harris_county','statewide_review','tribal_review',
                                                     'unresolved_or_other_area','unresolved_or_other_area'])
        r=audit_records(rows)
        self.assertEqual(sum(v['count'] for v in r['geography'].values()),5)

    def test_dates_missingness_duplicates_and_grouping_do_not_merge(self):
        rows=[record(1),record(2,incidentBeginDate='2020-01-02',incidentEndDate='2020-01-01'),
              record(3,incidentEndDate=None,incidentId=None),record(4,incidentEndDate='not-a-date'),
              record(1)]
        original=copy.deepcopy(rows); r=audit_records(rows)
        self.assertEqual(r['records'],5); self.assertEqual(r['unique_nonmissing_source_ids'],4)
        self.assertEqual(r['exact_duplicate_records'],1)
        self.assertIn('1',r['repeated_declaration_numbers'])
        self.assertEqual(r['incident_id_missing'],1)
        self.assertEqual(r['dates']['incidentEndDate']['missing_count'],1)
        self.assertEqual(r['dates']['incidentEndDate']['invalid_nonmissing_count'],1)
        self.assertEqual(r['date_relationships']['end_before_begin']['count'],1)
        self.assertIn('incident-a',r['incident_group_inconsistencies'])
        self.assertEqual(rows,original)

    def test_utc_date_labels_early_and_late_declarations_and_overlap(self):
        rows=[record(1,declarationDate='2019-12-31T00:00:00Z'),record(2,declarationDate='2020-01-05T00:00:00Z')]
        r=audit_records(rows)
        self.assertEqual(r['dates']['incidentBeginDate']['minimum'],'2020-01-01T00:00:00+00:00')
        self.assertEqual(r['date_relationships']['declaration_before_incident_begin']['count'],1)
        self.assertEqual(r['date_relationships']['declaration_after_incident_end']['count'],1)
        self.assertEqual(len(r['overlapping_closed_period_pairs']),1)


if __name__=='__main__':
    unittest.main()
