import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unified_recommendations as U
import unified_delivery as D


def reference(key,age=0):
    return {'key':key,'title':'Private saved reference '+key,'abstract':'Visual geometric matching '+key,
            'date_added':f'2026-11-{3-age:02d}T00:00:00Z'}


def paper(i):
    return {'arxiv_id':f'2611.{i:05d}','title':'Visual mapping '+str(i),'abstract':'Stereo visual geometry',
            'authors':['A'],'categories':['cs.CV'],'url':f'https://arxiv.org/abs/2611.{i:05d}',
            'pdf_url':f'https://arxiv.org/pdf/2611.{i:05d}'}


def bundle():
    p=paper(1)
    p.update(score=6.1234,recommendation_status='recommended',tldr='这是依据摘要生成的中文方法简介。',
             reason='几何匹配可迁移到视觉定位',cited_titles=['Private saved reference K'],recent_similarity=.8)
    return D.seal_bundle({'schema_version':1,'batch_id':'rss:2026-11-03','label':'公告批次 2026-11-03',
         'mode':'daily','papers':[p], 'profile_digest':'profile1','zotero_version':'42',
         'zotero_count':5,'synced_at':'2026-11-03T00:00:00Z','raw_count':200,'candidate_count':100,
         'diagnostic_count':0,'model':'zai-org/GLM-5.3-Flash'})


def test_recent_reference_update_changes_scores_with_same_candidates():
    now=datetime(2026,11,3,tzinfo=timezone.utc)
    rows=[paper(1),paper(2)]
    refs=[reference('old')]
    before=U.rank_pool(copy.deepcopy(rows),np.eye(2),refs,np.array([[1.,0.]]),now)
    refs.insert(0,reference('new'))
    after=U.rank_pool(copy.deepcopy(rows),np.eye(2),refs,np.array([[0.,1.],[1.,0.]]),now)
    assert before[0][0]['arxiv_id']!=after[0][0]['arxiv_id']
    assert after[1]['reference_digest']!=before[1]['reference_digest']


def test_full_pool_recall_finds_recent_match_beyond_old_first_100():
    rows=[dict(paper(i),score=10-i*.001,recent_similarity=0) for i in range(200)]
    rows[150]['recent_similarity']=.99
    short=U.shortlist(rows,100)
    assert len(short)==100 and len({p['arxiv_id'] for p in short})==100
    assert rows[150] in short
    assert [p['score'] for p in short]==sorted([p['score'] for p in short],reverse=True)


def test_similarity_uses_mean_cosine_not_renormalized_centroid():
    now=datetime(2026,11,3,tzinfo=timezone.utc)
    refs=[reference('a'),reference('b')]
    ranked,state=U.rank_pool([paper(1)],np.array([[1.,0.]]),refs,np.eye(2),now)
    assert ranked[0]['score']==pytest.approx(state['similarity_corpus'][0]['weight']*10)
    assert sum(r['weight'] for r in state['similarity_corpus'])==pytest.approx(1)


def test_invalid_vectors_do_not_become_recommendations():
    with pytest.raises(RuntimeError):
        U.rank_pool([paper(1)],np.array([[0.,0.]]),[reference('a')],np.array([[1.,0.]]),datetime.now(timezone.utc))


def test_encrypted_manifest_hides_zotero_and_authenticates_content():
    key=D.new_state_key();b=bundle();encrypted=D.encrypt_bundle(b,key)
    assert 'Private saved reference' not in json.dumps(encrypted)
    assert D.decrypt_bundle(encrypted,key)==b
    encrypted['ciphertext']=encrypted['ciphertext'][:-4]+'AAAA'
    with pytest.raises(Exception):D.decrypt_bundle(encrypted,key)


def test_manifest_digest_detects_different_channel_content():
    b=bundle();changed=copy.deepcopy(b);changed['papers'][0]['score']=1
    with pytest.raises(ValueError):D.validate_bundle(changed)


def test_both_renderers_preserve_same_ids_order_score_and_chinese_summary():
    b=bundle();html=D.render_email(b);cards=D.render_cards(b)
    text=json.dumps(cards,ensure_ascii=False)
    assert b['bundle_id'][:12] in html and b['bundle_id'][:12] in text
    assert b['papers'][0]['url'] in html and b['papers'][0]['url'] in text
    assert '6.1234' in html and '6.1234' in text
    assert b['papers'][0]['tldr'] in html and b['papers'][0]['tldr'] in text
    assert cards[0]['schema']=='2.0' and 2<=len(cards[0]['body']['elements'])<=5


def test_failure_in_one_channel_does_not_block_other_and_retry_only_missing():
    b=bundle();state={'deliveries':{}};saved=[];calls=[]
    def fail(_):calls.append('email');raise RuntimeError('SMTP unavailable')
    def accept(_):calls.append('feishu');return {'api_accepted':True,'complete':True,'message_ids':['om_ack']}
    with pytest.raises(D.PartialDelivery):
        D.deliver_pending(b,state,lambda s:saved.append(copy.deepcopy(s)),{'email':fail,'feishu':accept})
    assert calls==['email','feishu']
    assert 'feishu' in state['deliveries'][b['bundle_id']]
    calls.clear()
    def email(_):calls.append('email');return {'smtp_accepted':True}
    D.deliver_pending(b,state,lambda s:saved.append(copy.deepcopy(s)),{'email':email,'feishu':accept})
    assert calls==['email']
    proofs=state['deliveries'][b['bundle_id']]
    assert proofs['email']['bundle_id']==proofs['feishu']['bundle_id']==b['bundle_id']
    assert proofs['email']['paper_ids']==proofs['feishu']['paper_ids']


def test_false_ack_cannot_mark_channel_complete():
    b=bundle();state={}
    with pytest.raises(D.PartialDelivery):
        D.deliver_pending(b,state,lambda s:None,{'email':lambda _:{'smtp_accepted':False}})
    assert not state['deliveries'][b['bundle_id']]


def test_diagnostics_are_labelled_and_excluded_from_formal_sent_ids():
    b=bundle();b.pop('bundle_id');b['papers'][0]['recommendation_status']='diagnostic';b['diagnostic_count']=1
    b=D.seal_bundle(b)
    assert '核查样本' in D.render_email(b)
    assert D.formal_ids(b)==[]


def test_screening_stops_after_thirty_valid_results_preserving_similarity_order(monkeypatch):
    calls=[]
    def screen(papers,state,client=None):
        calls.append(len(papers))
        return {p['arxiv_id']:{'relation':'direct','reference_keys':['K'],'reason':'真实视觉匹配任务'} for p in papers}
    monkeypatch.setattr(U,'screen_zotero_relevance',screen)
    rows=[dict(paper(i),score=10-i*.01) for i in range(100)]
    selected,decisions=U.select_papers(rows,{'similarity_corpus':[reference('K')]},None)
    assert len(selected)==30 and len(decisions)==30 and calls==[10]*3
    assert [p['arxiv_id'] for p in selected]==[p['arxiv_id'] for p in rows[:30]]


def test_gate_cache_invalidates_on_new_zotero_but_summary_cache_survives(tmp_path):
    import sqlite3
    from unified_model_cache import ModelCache
    cache=ModelCache(sqlite3.connect(tmp_path/'cache.sqlite'),D.new_state_key())
    p=paper(1);p['zotero_matches']=[reference('K')]
    state={'reference_digest':'old-profile','last_synced_at':'2026-11-03T00:00:00+00:00',
           'similarity_corpus':[reference('K')]}
    d={'relation':'direct','reference_keys':['K'],'reason':'Private saved reference K 提供了几何依据'}
    cache.put_gate(p,state,d);cache.put_summary(p,'论文提出了可迁移的双目几何匹配算法。')
    assert cache.get_gate(p,state)==d
    assert cache.get_gate(p,dict(state,reference_digest='new-profile')) is None
    assert cache.get_summary(p)
    assert all('Private saved reference' not in r[0] for r in cache.db.execute('SELECT value FROM unified_private_cache'))


def test_unverified_missing_model_decision_is_not_cached(tmp_path):
    import sqlite3
    from unified_model_cache import ModelCache
    cache=ModelCache(sqlite3.connect(tmp_path/'cache.sqlite'),D.new_state_key())
    state={'reference_digest':'p','last_synced_at':'2026-11-03T00:00:00+00:00','similarity_corpus':[reference('K')]}
    cache.put_gate(paper(1),state,{'relation':'weak','validation_error':'missing_decision'})
    assert cache.get_gate(paper(1),state) is None


def test_pending_partial_delivery_keeps_same_profile_but_unsent_bundle_refreshes():
    from unified_push import profile_requires_refresh
    b=bundle()
    assert profile_requires_refresh(b,{},'43')
    assert not profile_requires_refresh(b,{'deliveries':{b['bundle_id']:{'email':{'smtp_accepted':True}}}},'43')
    assert not profile_requires_refresh(b,{'feishu_pages':{b['bundle_id']:{'0':'om_ack'}}},'43')


def test_transfer_review_cannot_keep_incompatible_sensors_or_invent_reference():
    from types import SimpleNamespace
    p=paper(1)
    state={'similarity_corpus':[reference('K')]}
    old={p['arxiv_id']:{'relation':'method','reference_keys':['K'],'reason':'泛化迁移',
          'transfer':{'evidence_quote':'We use geometric correspondence constraints for registration.'}}}
    rows=[{'id':p['arxiv_id'],'keep':True,'same_problem':False,'specific_transfer':True,
           'compatible_inputs':False,'reference_key':'K','reason':'物理观测量不相容'}]
    def complete(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(
            content=json.dumps({'decisions':rows},ensure_ascii=False)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
    assert U.review_transfers([p],state,client,old)[p['arxiv_id']]['relation']=='weak'
    rows[0].update(compatible_inputs=True,reference_key='invented')
    assert U.review_transfers([p],state,client,old)[p['arxiv_id']]['relation']=='weak'
    rows[0].update(reference_key='K',reason='同一几何匹配模块，尚需目标任务验证')
    assert U.review_transfers([p],state,client,old)[p['arxiv_id']]['relation']=='method'


def test_policy_change_refreshes_unsent_bundle_without_touching_partial_delivery():
    from unified_push import profile_requires_refresh
    b=bundle();b['policy_version']='old-policy';b=D.seal_bundle(b)
    assert profile_requires_refresh(b,{},b['zotero_version'])
    assert not profile_requires_refresh(b,{'deliveries':{b['bundle_id']:{'email':{}}}},b['zotero_version'])


def test_independent_3d_scope_is_not_collapsed_to_navigation_and_cross_domain_is_reviewed():
    assert U.aligned_task({'title':'Monocular depth consistency','abstract':'Evaluation of depth over video.'})
    assert U.aligned_task({'title':'Single-image 3D reconstruction','abstract':'A geometric consistency module.'})
    assert not U.aligned_task({'title':'Radio SLAM','abstract':'mmWave positioning using channel delay.'})
    assert not U.aligned_task({'title':'Polyp localization','abstract':'Medical segmentation with attention.'})


def test_daily_attempts_retry_delivery_without_computing_four_distinct_lists():
    from unified_push import pending_jobs
    now=datetime(2026,11,3,12,tzinfo=timezone.utc)
    job={'batch_id':'daily:new','kind':'daily','date':'2026-11-03'}
    state={'jobs':{'new':job},'deliveries':{'sent':{'email':{'paper_count':30,'delivered_at':now.isoformat()}}}}
    assert pending_jobs(state,'daily',now)==[]
    job['bundle_id']='partial'
    state['deliveries']['partial']={'feishu':{'paper_count':30,'delivered_at':now.isoformat()}}
    assert pending_jobs(state,'daily',now)==[job]


def test_generic_cross_domain_backbone_cannot_be_promoted_even_if_model_would_accept():
    from types import SimpleNamespace
    p=dict(paper(1),title='Attention residual U-Net for polyp localization',abstract='Medical detection with generic attention.')
    state={'similarity_corpus':[reference('K')]}
    old={p['arxiv_id']:{'relation':'method','reference_keys':['K'],'reason':'和陨石坑一样都是圆形目标',
         'transfer':{'evidence_quote':'selectively suppresses irrelevant background responses before feature reconstruction'}}}
    def forbidden(**kwargs):pytest.fail('Generic cross-domain architecture should be rejected locally without spending another call')
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=forbidden)))
    result=U.review_transfers([p],state,client,old)
    assert result[p['arxiv_id']]['relation']=='weak'
    assert result[p['arxiv_id']]['reference_keys']==[]
