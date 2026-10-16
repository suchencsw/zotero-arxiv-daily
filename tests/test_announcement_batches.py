import copy
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('announcement_queue_tests', SCRIPTS/'announcement_batches.py')
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def row(i, date='2026-10-12'):
    return {'arxiv_id':f'2610.{i:05}', 'title':'Visual navigation', 'abstract':'SLAM and mapping',
            'authors':['A'], 'categories':['cs.CV'], 'publish_date':date,
            'url':f'https://arxiv.org/abs/2610.{i:05}'}


def xml(date='2026-10-12', kind='new', summary='A real abstract', updated='2026-10-13T00:00:00Z'):
    return f'''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
    <title>cs.CV updates on arXiv.org</title><updated>{updated}</updated><entry>
    <id>oai:arXiv.org:2610.00001v1</id><title>Visual navigation</title>
    <published>{date}T00:00:00-04:00</published><summary>{summary}</summary>
    <category term="cs.CV"/><arxiv:announce_type>{kind}</arxiv:announce_type></entry></feed>'''


def batch(date, complete=None):
    return {'id':Q.batch_key(date),'date':date,'path':date+'.json','count':100,
            **({'completed_at':complete} if complete else {})}


def test_batch_date_is_rss_published_not_fetch_update_or_send_day():
    assert list(Q.parse_feed(xml())) == ['2026-10-12']


@pytest.mark.parametrize('kind', ['new','cross'])
def test_accept_announcements(kind):
    assert len(Q.parse_feed(xml(kind=kind))['2026-10-12']) == 1


def test_replacements_are_not_new_batches():
    assert Q.parse_feed(xml(kind='replace')) == {}


@pytest.mark.parametrize('body', ['<feed>', '<html>error</html>', xml(date=''), xml(summary='')])
def test_bad_feed_fails_closed(body):
    with pytest.raises((ValueError,RuntimeError, Q.ET.ParseError)):
        Q.parse_feed(body)


def test_empty_valid_feed_is_quiet_not_a_completed_batch():
    groups = Q.parse_feed('<feed xmlns="http://www.w3.org/2005/Atom"><title>cs.CV updates on arXiv.org</title></feed>')
    assert groups == {}
    assert Q.plan({},datetime(2026,10,12,0,41,tzinfo=Q.BJ)) == (None,'wait','')


def test_same_batch_does_not_resend_after_midnight():
    b = batch('2026-10-12','2026-10-12T10:00:00+08:00')
    assert Q.plan({'batches':{b['id']:b}},datetime(2026,10,13,0,41,tzinfo=Q.BJ))[0] is None


def test_new_batch_not_blocked_by_previous_review_or_same_day_delivery():
    old,new = batch('2026-10-09','confirmed'),batch('2026-10-12')
    s = {'batches':{old['id']:old,new['id']:new},'completed':{'some_review':{}}}
    assert Q.plan(s,datetime(2026,10,12,9,tzinfo=Q.BJ))[0] == new


def test_missed_observed_batches_are_oldest_first_even_when_new_rss_arrives():
    old,new = batch('2026-10-09'),batch('2026-10-12')
    assert Q.plan({'batches':{new['id']:new,old['id']:old}},datetime(2026,10,12,9,tzinfo=Q.BJ))[0] == old


def test_review_only_late_and_once_per_beijing_date():
    b = batch('2026-10-09','confirmed')
    s = {'batches':{b['id']:b}}
    assert Q.plan(s,datetime(2026,10,12,19,tzinfo=Q.BJ))[0] is None
    _, mode,key = Q.plan(s,datetime(2026,10,12,21,tzinfo=Q.BJ))
    assert mode == 'review' and key.startswith('review:2026-10-12:')
    s['completed'] = {key:{}}
    assert Q.plan(s,datetime(2026,10,12,23,tzinfo=Q.BJ))[0] is None


def test_review_expiry_does_not_expire_undelivered_batches():
    old = batch('2026-09-01')
    now = datetime(2026,10,12,21,tzinfo=Q.BJ)
    assert Q.plan({'batches':{old['id']:old}},now)[0] == old
    old['completed_at'] = 'confirmed'
    assert Q.plan({'batches':{old['id']:old}},now)[0] is None


@pytest.mark.parametrize('channel', ['email','feishu'])
def test_candidate_cap(channel):
    assert len(Q.select_candidates([row(i) for i in range(300)],channel)) == 100


def test_email_candidate_order_is_unchanged():
    rows = [row(i) for i in range(120)]
    assert Q.select_candidates(rows,'email') == rows[:100]


def test_force_resend_requires_explicit_opt_in():
    b = batch('2026-10-12','2026-10-12T10:00:00+08:00')
    s = {'batches':{b['id']:b}}
    now = datetime(2026,10,13,0,41,tzinfo=Q.BJ)
    assert Q.plan(s,now)[0] is None
    assert Q.plan(s,now,force=True)[0] == b


def test_receipt_cannot_complete_a_batch_without_real_ack():
    p = {'schema_version':1,'run_id':'1','paper_count':30,'smtp_accepted':False}
    assert not Q.valid_proof(p,'email',1)
    p['smtp_accepted'] = True
    assert Q.valid_proof(p,'email',1)
    assert not Q.valid_proof(p,'email',2)
    assert not Q.valid_proof({**p,'paper_count':0},'email',1)
    assert not Q.valid_proof({**p,'paper_count':True},'email',1)
    assert not Q.valid_proof(p,'feishu',1)
    assert Q.valid_proof({**p,'api_accepted':True,'complete':True,'message_ids':['om_ack']},'feishu',1)


def test_review_receipt_does_not_mark_new_batch_as_sent():
    b = batch('2026-10-12')
    s = {'batches':{b['id']:b}}
    p = {'announcement_batch_id':b['id'],'announcement_mode':'review',
         'announcement_review_key':'review:2026-10-12:'+b['id'],'delivered_at':'now'}
    Q.apply_proof(s,p)
    assert not b.get('completed_at')
    assert p['announcement_review_key'] in s['completed']
    Q.apply_proof(s,{**p,'announcement_mode':'papers'})
    assert b['completed_at'] == 'now'


def test_queue_consumer_requires_identity_and_dated_abstract(monkeypatch,tmp_path):
    path = tmp_path/'snapshot.json'
    data = {'batch_id':Q.batch_key('2026-10-12'),'candidates':[row(1)]}
    path.write_text(json.dumps(data),encoding='utf8')
    monkeypatch.setenv('ARXIV_BATCH_PATH',str(path))
    monkeypatch.setenv('ARXIV_BATCH_ID',data['batch_id'])
    monkeypatch.setenv('ARXIV_BATCH_DATE','2026-10-12')
    # Each repository has its own consumer; the shared date contract is identical.
    if (SCRIPTS.parent/'paperflow').exists():
        from paperflow.arxiv_rss import fetch_arxiv_rss_papers
        assert len(fetch_arxiv_rss_papers(['cs.CV'])) == 1
        monkeypatch.setenv('ARXIV_BATCH_MODE','review')
        assert fetch_arxiv_rss_papers(['cs.CV'])[0]['historical_review'] is True
        monkeypatch.setenv('ARXIV_BATCH_DATE','2026-10-13')
        with pytest.raises(RuntimeError,match='metadata'):
            fetch_arxiv_rss_papers(['cs.CV'])


def test_persisted_queue_file_is_separate_from_periodic_ledger():
    ledger = Q.Ledger('owner/repo',state_file='announcement-ledger.json')
    assert ledger.state_file == 'announcement-ledger.json'
    assert Q.Ledger('owner/repo').state_file == 'periodic-ledger.json'


def test_prepare_persists_public_snapshot_before_ranking_without_sending(monkeypatch,tmp_path):
    state = {'schema_version':1,'batches':{},'completed':{},'sent':{}}
    files = {}
    class FakeLedger:
        def __init__(self, **kwargs): pass
        def load(self): return copy.deepcopy(state),'state_sha'
        def update(self, change):
            change(state)
            return copy.deepcopy(state)
        def api(self, path, payload=None, **kwargs):
            if path.startswith('actions/workflows/'):
                return {'workflow_runs':[]}
            filename = path.removeprefix('contents/').split('?')[0]
            if payload is not None:
                files[filename] = {'content':payload['content'],'encoding':'base64'}
                return {}
            if filename not in files:
                raise Q.urllib.error.HTTPError('test',404,'missing',{},None)
            return files[filename]
    class Response:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,*args): return xml().encode()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('GITHUB_OUTPUT',str(tmp_path/'output'))
    monkeypatch.setenv('GITHUB_ENV',str(tmp_path/'env'))
    monkeypatch.setenv('GITHUB_RUN_ID','99')
    monkeypatch.setattr(Q,'Ledger',FakeLedger)
    monkeypatch.setattr(Q.urllib.request,'urlopen',lambda *a,**kw:Response())
    Q.prepare('email')
    assert 'should_run=true' in (tmp_path/'output').read_text('utf8')
    snapshot = json.loads((tmp_path/'data/selected-announcement.json').read_text('utf8'))
    assert snapshot['announcement_date'] == '2026-10-12'
    assert snapshot['batch_id'] in state['batches']
    assert not state['batches'][snapshot['batch_id']].get('completed_at')
    assert len(files) == 1  # snapshot persisted before ranking/sending
    assert 'ARXIV_BATCH_DATE=2026-10-12' in (tmp_path/'env').read_text('utf8')
