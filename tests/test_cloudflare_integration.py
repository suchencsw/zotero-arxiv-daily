import importlib
import json
from types import SimpleNamespace

def gate_module():
    try:
        return importlib.import_module('paperflow.zotero_relevance')
    except ModuleNotFoundError:
        return importlib.import_module('zotero_arxiv_daily.grounded_relevance')

def test_cloudflare_gate_disables_hidden_reasoning(monkeypatch):
    module=gate_module()
    monkeypatch.setenv('OPENAI_BASE_URL','https://api.cloudflare.com/client/v4/accounts/test/ai/v1')
    monkeypatch.setenv('PAPERFLOW_LLM_MODEL','@cf/zai-org/glm-4.7-flash')
    calls=[]
    def create(**kw):
        calls.append(kw)
        wire=json.loads(kw['messages'][1]['content'])
        result={'decisions':[{'id':wire['candidates'][0]['id'],'relation':'direct',
             'reference_keys':[wire['saved_research_context'][0]['key']],'reason':'直接对应视觉定位任务'}]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    state={'similarity_corpus':[{'key':'saved','title':'Visual SLAM','abstract':'Visual mapping','date_added':'2026-11-02','weight':1.0}],
           'last_synced_at':'2026-11-03T00:00:00+00:00'}
    module.screen_zotero_relevance([{'arxiv_id':'2610.12345','title':'Visual mapping','abstract':'Visual SLAM',
             'zotero_matches':[{'key':'saved','title':'Visual SLAM'}]}],state,client=client)
    assert calls[0]['extra_body']=={'chat_template_kwargs':{'enable_thinking':False}}


def test_verbatim_method_quote_slightly_over_display_limit_is_not_false_rejection():
    module=gate_module()
    quote='A geometric correspondence mechanism improves feature matching. '*4
    batch={'saved_research_context':[{'key':'K'}],
        'candidates':[{'id':'P','abstract':quote,'references':[{'key':'K'}]}]}
    payload={'decisions':[{'id':'P','relation':'method','reference_keys':['K'],
        'reason':'真实几何对应机制可用于定位匹配',
        'transfer':{'source_mechanism':'几何对应','target_component':'特征匹配','evidence_quote':quote.strip()}}]}
    result=module.parse_relevance_decisions(json.dumps(payload),batch)
    assert result[0]['relation']=='method'
    assert len(result[0]['transfer']['evidence_quote'])<=180


def test_real_first_quote_fragment_survives_ellipsis_but_invented_quote_does_not():
    module=gate_module()
    sentence='We introduce geometric correspondence constraints for robust visual matching.'
    batch={'saved_research_context':[{'key':'K'}],
        'candidates':[{'id':'P','abstract':sentence,'references':[{'key':'K'}]}]}
    d={'id':'P','relation':'method','reference_keys':['K'],'reason':'几何对应可用于定位匹配',
       'transfer':{'source_mechanism':'几何对应','target_component':'定位匹配','evidence_quote':sentence+' ... invalid omitted continuation'}}
    got=module.parse_relevance_decisions(json.dumps({'decisions':[d]}),batch)[0]
    assert got['relation']=='method' and got['transfer']['evidence_quote']==sentence
    d['transfer']['evidence_quote']='An entirely invented mechanism without any source support.'
    assert module.parse_relevance_decisions(json.dumps({'decisions':[d]}),batch)[0]['relation']=='weak'

def test_independent_review_can_reject_first_model_acceptance(monkeypatch):
    module=gate_module()
    monkeypatch.setenv('PAPERFLOW_RELEVANCE_REVIEW_MODEL','@cf/openai/gpt-oss-120b')
    monkeypatch.setenv('PAPERFLOW_LLM_MODEL','@cf/zai-org/glm-4.7-flash')
    monkeypatch.setenv('OPENAI_BASE_URL','https://api.cloudflare.com/client/v4/accounts/test/ai/v1')
    calls=[]
    def create(**kw):
        calls.append(kw)
        wire=json.loads(kw['messages'][1]['content'])
        review=kw['model']=='@cf/openai/gpt-oss-120b'
        d={'id':wire['candidates'][0]['id'],'relation':'weak' if review else 'direct',
           'reference_keys':[] if review else [wire['saved_research_context'][0]['key']],'reason':'研究对象和传感假设不相容'}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({'decisions':[d]})))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    state={'similarity_corpus':[{'key':'K','title':'UAV SLAM','abstract':'UAV navigation','date_added':'2026-11-02','weight':1.0}]}
    out=module.screen_zotero_relevance([{'arxiv_id':'P','title':'Microbot navigation','abstract':'Microscopic control'}],state,client=client)
    assert out['P']['relation']=='weak'
    assert len(calls)==2 and calls[-1]['reasoning_effort']=='low'
def test_cloudflare_summary_disables_hidden_reasoning(monkeypatch):
    from zotero_arxiv_daily.protocol import _request_llm
    monkeypatch.setenv('OPENAI_BASE_URL','https://api.cloudflare.com/client/v4/accounts/test/ai/v1')
    calls=[]
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='中文简介'))])
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    _request_llm(client,{'generation_kwargs':{'model':'@cf/zai-org/glm-4.7-flash'}},[])
    assert calls[0]['extra_body']['chat_template_kwargs']['enable_thinking'] is False

