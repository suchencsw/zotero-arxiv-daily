import importlib.util
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('delivery_v2_queue',SCRIPTS/'announcement_batches.py')
Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)

def test_both_channels_recall_navigation_beyond_first100():
    rows=[{'arxiv_id':f'2610.{i:05}','title':'Generic language training','abstract':'Generic benchmarks','categories':['cs.AI']} for i in range(200)]
    target={'arxiv_id':'2610.99999','title':'LIN-SLAM: Lunar visual navigation','abstract':'3D mapping and loop closure','categories':['cs.CV']}
    rows.append(target)
    assert Q.select_candidates(rows,'email')==Q.select_candidates(rows,'feishu')
    assert target in Q.select_candidates(rows,'email')
    assert len(Q.select_candidates(rows,'email'))==100

def test_delayed_night_review_uses_previous_day():
    b={'id':Q.batch_key('2026-10-12'),'date':'2026-10-12','completed_at':'2026-10-12T07:00:00Z','count':100}
    selected,mode,key=Q.plan({'batches':{b['id']:b}},datetime(2026,10,14,1,1,tzinfo=Q.BJ))
    assert selected==b and mode=='review' and key.startswith('review:2026-10-13:')

def test_no_extra_review_after_papers_already_delivered_that_day():
    b={'id':Q.batch_key('2026-10-16'),'date':'2026-10-16','completed_at':'2026-10-16T06:00:00Z','count':100}
    assert Q.plan({'batches':{b['id']:b}},datetime(2026,10,16,22,tzinfo=Q.BJ))[0] is None

def test_only_formal_nonreview_papers_enter_cross_batch_dedup():
    s={'batches':{},'sent':{}}
    p={'delivered_at':'2026-10-16T06:00:00Z','announcement_mode':'papers',
       'papers':[{'arxiv_id':'2610.16143v2'},{'arxiv_id':'2610.99998','recommendation_status':'diagnostic'}]}
    Q.apply_proof(s,p)
    assert Q.paper_fingerprint('2610.16143') in s['sent']
    assert Q.paper_fingerprint('2610.99998') not in s['sent']
    Q.apply_proof(s,{**p,'announcement_mode':'review','papers':[{'arxiv_id':'2610.99997'}]})
    assert Q.paper_fingerprint('2610.99997') not in s['sent']



class MemoryLedger:
    def __init__(self, state):
        self.state = state
        self.files = {}

    def load(self):
        return self.state, 'state_sha'

    def update(self, mutation):
        mutation(self.state)

    def api(self, path, payload=None, **kwargs):
        filename = path.removeprefix('contents/').split('?')[0]
        if payload is not None:
            import base64
            import json
            self.files[filename] = json.loads(base64.b64decode(payload['content']))
            return {}
        raise Q.urllib.error.HTTPError('test', 404, 'missing', {}, None)


def test_scope_upgrade_refresh_is_review_only_without_fake_delivery():
    date = '2026-10-16'
    old = {'id': 'rss:'+date+':old_scope', 'date': date,
           'completed_at': '2026-10-16T06:00:00Z', 'count': 100}
    state = {'batches': {old['id']: old}}
    ledger = MemoryLedger(state)
    rows = [{'arxiv_id': '2610.00001', 'title': 'SLAM', 'abstract': 'Mapping',
             'categories': ['cs.CV']}]
    Q.save_snapshot(ledger, date, rows, 'email')
    refresh = state['batches'][Q.batch_key(date)]
    assert refresh['review_only'] is True
    assert refresh['review_refresh_of'] == old['id']
    assert not refresh.get('completed_at')
    assert Q.plan(state, datetime(2026, 10, 16, 22, tzinfo=Q.BJ))[0] is None
    selected, mode, _ = Q.plan(state, datetime(2026, 10, 17, 22, tzinfo=Q.BJ))
    assert selected == refresh and mode == 'review'


def test_snapshot_refills_after_confirmed_ids_are_removed():
    ledger = MemoryLedger({'batches': {}})
    rows = [{'arxiv_id': f'2610.{i:05}', 'title': 'SLAM', 'abstract': 'Mapping',
             'categories': ['cs.CV']} for i in range(120)]
    sent = {Q.paper_fingerprint(p['arxiv_id']+'v2') for p in rows[:20]}
    Q.save_snapshot(ledger, '2026-10-16', rows, 'email', sent)
    candidates = next(iter(ledger.files.values()))['candidates']
    assert len(candidates) == 100
    assert candidates == rows[20:]


def test_periodic_content_also_suppresses_redundant_nightly_review():
    b = {'id': Q.batch_key('2026-10-12'), 'date': '2026-10-12',
         'completed_at': '2026-10-12T06:00:00Z', 'count': 100}
    state = {'batches': {b['id']: b},
             'content_deliveries': ['2026-10-16T06:00:00Z']}
    assert Q.plan(state, datetime(2026, 10, 16, 22, tzinfo=Q.BJ))[0] is None
