import json
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
import numpy as np

from zotero_arxiv_daily.protocol import CorpusPaper
from zotero_arxiv_daily.reranker.base import BaseReranker
from zotero_arxiv_daily.personalization import screen_ranked
from zotero_arxiv_daily.grounded_relevance import parse_relevance_decisions

class MatrixRanker(BaseReranker):
    def get_similarity_score(self,a,b): return np.eye(len(a),len(b))

def test_recent_weighting_is_explicit_and_timestamp_ties_are_deterministic(monkeypatch):
    now=datetime.now(timezone.utc)
    refs=[CorpusPaper('old','old',now-timedelta(days=60),[],key='A'),CorpusPaper('new','new',now,[],key='B')]
    p=[SimpleNamespace(abstract='a'),SimpleNamespace(abstract='b')]
    ranker=MatrixRanker(None)
    monkeypatch.delenv('EMAIL_RECENT_WEIGHTING',raising=False)
    ranker.rerank(p,refs);legacy=ranker.profile_state['similarity_corpus'][0]['weight']
    monkeypatch.setenv('EMAIL_RECENT_WEIGHTING','recent_mixture')
    ranker.rerank(p,refs);recent=ranker.profile_state['similarity_corpus'][0]['weight']
    assert recent>.7 and recent>legacy
    assert abs(sum(r['weight'] for r in ranker.profile_state['similarity_corpus'])-1)<1e-12
    assert ranker.profile_state['similarity_corpus'][0]['key']=='B'
    assert len(ranker.profile_state['reference_digest'])==64

def papers(n):
    return [SimpleNamespace(title=f'Paper {i}',abstract='Navigation',score=10-i/10,url=f'https://arxiv.org/abs/2610.{i:05}',zotero_matches=[],zotero_recent_matches=[]) for i in range(n)]

def test_only_fill_to_five_diagnostics_not_to_thirty(monkeypatch):
    rows=papers(10)
    decisions={f'2610.{i:05}':{'relation':'direct' if i in (7,8) else 'weak','reference_keys':['K'],'reason':'关联定位'} for i in range(10)}
    monkeypatch.setattr('zotero_arxiv_daily.personalization.screen_zotero_relevance',lambda *a,**kw:decisions)
    got=screen_ranked(rows,{'similarity_corpus':[{'key':'K','title':'SLAM'}]},None,limit=30)
    assert len(got)==5 and sum(p.recommendation_status=='recommended' for p in got)==2
    assert all(a.score>=b.score for a,b in zip(got,got[1:]))

def test_enough_formal_results_never_add_weak_top_ranked(monkeypatch):
    rows=papers(40)
    decisions={f'2610.{i:05}':{'relation':'weak' if i<5 else 'direct','reference_keys':['K'],'reason':'关联定位'} for i in range(40)}
    monkeypatch.setattr('zotero_arxiv_daily.personalization.screen_zotero_relevance',lambda *a,**kw:decisions)
    got=screen_ranked(rows,{'similarity_corpus':[{'key':'K','title':'SLAM'}]},None,limit=30)
    assert len(got)==30 and all(p.recommendation_status=='recommended' for p in got)
    assert got[0].title=='Paper 5'

def test_method_requires_verbatim_abstract_evidence():
    batch={'saved_research_context':[{'key':'K'}],'candidates':[{'id':'P','abstract':'We use graph correspondence to align multi-view geometry.','references':[{'key':'K'}]}]}
    d={'id':'P','relation':'method','reference_keys':['K'],'reason':'图对应可用于跨视图匹配'}
    assert parse_relevance_decisions(json.dumps({'decisions':[d]}),batch)[0]['relation']=='weak'
    d.update(relation='method',transfer={'source_mechanism':'图对应','target_component':'跨视图几何匹配','evidence_quote':'graph correspondence to align multi-view geometry'})
    assert parse_relevance_decisions(json.dumps({'decisions':[d]}),batch)[0]['relation']=='method'

def test_chinese_reason_and_saved_reference_titles_only_in_email_not_public_preview(monkeypatch,tmp_path,config):
    from zotero_arxiv_daily.construct_email import render_email
    from tests.canned_responses import make_sample_paper
    p=make_sample_paper();p.score=7.0;p.relevance_decision={'reason':'与私人场景图文献相关'}
    p.reference_titles=['PRIVATE_REFERENCE_TITLE'];p.recommendation_status='recommended';p.similarity_rank=1
    text=render_email([p]);assert 'PRIVATE_REFERENCE_TITLE' in text and 'profile-evidence' in text
    import re
    public=re.sub(r'<aside class="profile-evidence">.*?</aside>','redacted',text,flags=re.S)
    assert 'PRIVATE_REFERENCE_TITLE' not in public
