"""Frozen recommendation bundles, encrypted persistence and independent delivery."""
from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import html
import json
import os
import re
import secrets
import smtplib
from datetime import datetime, timezone
from email.header import Header
from email.mime.text import MIMEText
from pathlib import Path

import requests
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def seal_bundle(bundle):
    bundle=copy.deepcopy(bundle)
    bundle.pop('bundle_id',None)
    bundle['bundle_id']=hashlib.sha256(canonical(bundle)).hexdigest()
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    plain={k:v for k,v in bundle.items() if k!='bundle_id'}
    if bundle.get('bundle_id')!=hashlib.sha256(canonical(plain)).hexdigest():
        raise ValueError('Recommendation bundle digest mismatch')
    papers=bundle['papers']
    if len(papers)>30 or len({p['arxiv_id'] for p in papers})!=len(papers):
        raise ValueError('Invalid recommendation size or duplicate IDs')
    if [p['score'] for p in papers]!=sorted((p['score'] for p in papers),reverse=True):
        raise ValueError('Both channels must preserve similarity ordering')
    for p in papers:
        if not re.fullmatch(r'\d{4}\.\d{4,5}',p['arxiv_id']):raise ValueError('Invalid arXiv ID')
        if p['url']!='https://arxiv.org/abs/'+p['arxiv_id']:raise ValueError('Invalid paper link')
        if p['pdf_url']!='https://arxiv.org/pdf/'+p['arxiv_id']:raise ValueError('Invalid PDF link')
        if not re.search(r'[\u4e00-\u9fff]',p.get('tldr','')):raise ValueError('Chinese TLDR required')
    return bundle


def new_state_key():return base64.b64encode(secrets.token_bytes(32)).decode()


def encrypt_bundle(bundle,key):
    validate_bundle(bundle)
    return encrypt_payload(bundle,key)


def encrypt_payload(value,key):
    nonce=secrets.token_bytes(12)
    encrypted=AESGCM(base64.b64decode(key,validate=True)).encrypt(nonce,canonical(value),b'paperflow-unified-v1')
    return {'schema_version':1,'algorithm':'AES-256-GCM',
            'nonce':base64.b64encode(nonce).decode(),'ciphertext':base64.b64encode(encrypted).decode()}


def decrypt_bundle(envelope,key):
    return validate_bundle(decrypt_payload(envelope,key))


def decrypt_payload(envelope,key):
    if envelope.get('algorithm')!='AES-256-GCM' or envelope.get('schema_version')!=1:
        raise ValueError('Unsupported encrypted recommendation state')
    plain=AESGCM(base64.b64decode(key,validate=True)).decrypt(
        base64.b64decode(envelope['nonce'],validate=True),base64.b64decode(envelope['ciphertext'],validate=True),b'paperflow-unified-v1')
    return json.loads(plain)


def formal_ids(bundle):
    return [p['arxiv_id'] for p in bundle['papers'] if p['recommendation_status']=='recommended']


def summary(bundle):
    return (f"扫描 {bundle['raw_count']} 篇 → 候选 {bundle['candidate_count']} 篇 → 展示 {len(bundle['papers'])} 篇。"
            f"Zotero {bundle['zotero_count']} 篇 / 版本 {bundle['zotero_version']}。"
            f"画像 {bundle['profile_digest'][:12]}。"
            '近期30天70% + 全库30%，半衰期14天；相似度是加权余弦×10，不是概率。')


def render_email(bundle):
    validate_bundle(bundle);esc=html.escape
    rows=[f'<h2>{esc(bundle["label"])}</h2>',f'<p>{esc(summary(bundle))}</p>',
          f'<p>同源批次 <b>{bundle["bundle_id"][:12]}</b> · 画像读取时间 {esc(bundle["synced_at"])} · 与飞书正文及排序一致</p>']
    for i,p in enumerate(bundle['papers'],1):
        status=('方法参考 · 需迁移验证' if p.get('relation')=='method' else '直接相关') if p['recommendation_status']=='recommended' else '核查样本 · 未通过相关性验证'
        rows.append(f'<section data-arxiv-id="{p["arxiv_id"]}" style="border:1px solid #dce3ef;border-radius:10px;padding:18px;margin:16px 0">'
            f'<h3>{i:02d}. <a href="{p["url"]}">{esc(p["title"])}</a></h3>'
            f'<p style="color:#63708a">{esc(", ".join(p.get("authors",[])[:6]))}</p>'
            f'<p>{status} · 相似度 <b>{p["score"]:.4f}/10</b> · 近期最相似 {p.get("recent_similarity",0)*10:.4f}/10</p>'
            f'<p><b>中文简介：</b>{esc(p["tldr"])}</p>'
            f'<p><b>推荐依据：</b>{esc(p["reason"])}</p>'
            f'<p style="color:#63708a">关联收藏：{esc("；".join(p.get("cited_titles",[])[:2])) or "没有经过验证的收藏关联"}</p>'
            f'<a href="{p["url"]}">论文页</a>　<a href="{p["pdf_url"]}">PDF</a></section>')
    if not bundle['papers']:rows.append('<p>本周期没有尚未保存或推荐的合格补选论文。</p>')
    return '<!doctype html><html><meta charset="utf-8"><body style="font-family:Arial,sans-serif;max-width:980px;margin:24px auto;color:#1f2937">'+''.join(rows)+'</body></html>'


def md(text):
    escaped=html.escape(str(text),quote=False)
    return re.sub(r'([\[\]*_`~])',lambda m:'&#'+str(ord(m.group(1)))+';',escaped)


def render_cards(bundle):
    """Card 2.0: one metric block plus <=4 bordered paper blocks, open-url only."""
    validate_bundle(bundle)
    papers=bundle['papers'];chunks=[papers[i:i+4] for i in range(0,len(papers),4)] or [[]]
    cards=[]
    for page,chunk in enumerate(chunks,1):
        metric={'tag':'column_set','flex_mode':'none','columns':[
            {'tag':'column','width':'weighted','weight':1,'background_style':'blue-50',
             'padding':'12px','vertical_spacing':'4px','elements':[
                {'tag':'markdown','content':f'## {len(papers)} 篇 · 相似度降序'},
                {'tag':'markdown','content':f'同源批次 {bundle["bundle_id"][:12]} · 第{page}/{len(chunks)}卡\n'+md(summary(bundle)),
                 'text_size':'notation'}]}]}
        elements=[metric]
        for index,p in enumerate(chunk,(page-1)*4+1):
            status=('方法参考 · 需迁移验证' if p.get('relation')=='method' else '直接相关') if p['recommendation_status']=='recommended' else '核查样本 · 未通过相关性验证'
            content=(f'**{index:02d}. {md(p["title"])}**\n{status} · 相似度 **{p["score"]:.4f}/10**\n'
                     f'**中文简介：**{md(p["tldr"])}\n**推荐依据：**{md(p["reason"])}\n'
                     f'<font color="grey">关联收藏：{md("；".join(p.get("cited_titles",[])[:2])) or "没有经过验证的收藏关联"}</font>\n'
                     f'[论文页]({p["url"]})　[PDF]({p["pdf_url"]})')
            elements.append({'tag':'interactive_container','width':'fill','has_border':True,
                'border_color':'grey','corner_radius':'8px','padding':'12px','vertical_spacing':'4px',
                'behaviors':[{'type':'open_url','default_url':p['url']}],
                'elements':[{'tag':'markdown','content':content}]})
        if not chunk:
            elements.append({'tag':'markdown','content':'本周期没有尚未保存或推荐的合格补选论文。'})
        cards.append({'schema':'2.0','config':{'width_mode':'default','update_multi':True},
            'header':{'template':'blue','title':{'tag':'plain_text','content':bundle['label']},
                      'subtitle':{'tag':'plain_text','content':'同一份推荐 · 邮件与飞书顺序一致'},
                      'icon':{'tag':'standard_icon','token':'notice_colorful'}},
            'body':{'direction':'vertical','padding':'12px 12px 20px 12px',
                    'vertical_spacing':'12px','elements':elements}})
    return cards


class PartialDelivery(RuntimeError):pass


def receipt(bundle,channel,ack):
    if channel=='email' and ack.get('smtp_accepted') is not True:raise ValueError('No SMTP acknowledgement')
    if channel=='feishu' and not (ack.get('api_accepted') is True and ack.get('complete') is True and ack.get('message_ids')):
        raise ValueError('No complete Feishu acknowledgement')
    return dict(ack,schema_version=1,channel=channel,bundle_id=bundle['bundle_id'],
                batch_id=bundle['batch_id'],run_id=os.environ.get('GITHUB_RUN_ID','local'),
                delivered_at=datetime.now(timezone.utc).isoformat(),
                paper_ids=[p['arxiv_id'] for p in bundle['papers']],formal_ids=formal_ids(bundle),
                profile_digest=bundle['profile_digest'],paper_count=len(bundle['papers']))


def deliver_pending(bundle,state,persist,senders):
    validate_bundle(bundle)
    delivered=state.setdefault('deliveries',{}).setdefault(bundle['bundle_id'],{})
    failures=[]
    for channel,send in senders.items():
        if channel in delivered:continue
        try:
            proof=receipt(bundle,channel,send(bundle))
            delivered[channel]=proof
            target=Path(os.environ.get('UNIFIED_RECEIPT_PATH','outputs/unified-receipts.json'))
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps([v for channels in state['deliveries'].values() for v in channels.values()],indent=2),'utf8')
            persist(state)
        except Exception as exc:
            failures.append(channel+':'+type(exc).__name__)
    if failures:raise PartialDelivery('Delivery pending: '+', '.join(failures))
    return delivered


def send_email(bundle):
    message=MIMEText(render_email(bundle),'html','utf-8')
    message['Subject']=Header('arXiv '+bundle['label']+' ['+bundle['bundle_id'][:12]+']','utf-8').encode()
    message['From']=os.environ['SENDER'];message['To']=os.environ['RECEIVER']
    message['Message-ID']='<'+bundle['bundle_id']+'@paperflow.local>'
    port=int(os.environ.get('PERIODIC_SMTP_PORT','465'));host=os.environ.get('PERIODIC_SMTP_SERVER','smtp.qq.com')
    server=smtplib.SMTP_SSL(host,port,timeout=45) if port==465 else smtplib.SMTP(host,port,timeout=45)
    try:
        if port!=465:server.starttls()
        server.login(os.environ['SENDER'],os.environ['SENDER_PASSWORD'])
        if server.sendmail(os.environ['SENDER'],[os.environ['RECEIVER']],message.as_string()):raise RuntimeError('SMTP refused recipients')
    finally:
        with contextlib.suppress(Exception):server.quit()
    return {'smtp_accepted':True}


def send_feishu(bundle,confirmed_pages=None,on_page=None):
    session=requests.Session();response=session.post(
        'https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal',
        json={'app_id':os.environ['FEISHU_APP_ID'],'app_secret':os.environ['FEISHU_APP_SECRET']},timeout=45)
    response.raise_for_status();auth=response.json()
    if auth.get('code')!=0 or not auth.get('tenant_access_token'):raise RuntimeError('Feishu authentication failed')
    pages=dict(confirmed_pages or {});ids=[]
    for i,card in enumerate(render_cards(bundle)):
        page=str(i)
        if page in pages:ids.append(pages[page]);continue
        identity=hashlib.sha256((bundle['bundle_id']+':'+page).encode()).hexdigest()[:32]
        response=session.post('https://open.feishu.cn/open-apis/im/v1/messages',
            params={'receive_id_type':'chat_id'},headers={'Authorization':'Bearer '+auth['tenant_access_token']},
            json={'receive_id':os.environ['FEISHU_CHAT_ID'],'msg_type':'interactive',
                  'content':json.dumps(card,ensure_ascii=False),'uuid':identity},timeout=60)
        response.raise_for_status();data=response.json();identifier=(data.get('data') or {}).get('message_id')
        if data.get('code')!=0 or not identifier:raise RuntimeError('Feishu did not acknowledge card: '+str(data.get('code'))+' '+str(data.get('msg',''))[:300])
        pages[page]=identifier;ids.append(identifier)
        if on_page:on_page(page,identifier)
    return {'api_accepted':True,'complete':True,'message_ids':ids}
