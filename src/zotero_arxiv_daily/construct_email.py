from .protocol import Paper
import math
import os
import html


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        今日未检索到新的 arXiv 公告，邮件任务运行正常。<br>
        这不代表没有人提交论文，也不代表你的研究方向没有相关文献。后续定时尝试仍会检查新公告。
    </td>
  </tr>
  </table>
  """
  return block_template

def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None):
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            <a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""
    return block_template.format(title=title, authors=authors,rate=rate, tldr=tldr, pdf_url=pdf_url, affiliations=affiliations)

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper]) -> str:
    parts = []
    announcement = os.environ.get("ARXIV_BATCH_DATE", "")
    if announcement:
        review = os.environ.get("ARXIV_BATCH_MODE") == "review"
        label = "历史批次复盘" if review else "公开公告批次推荐"
        parts.append(f'<h2>{label} · {html.escape(announcement)}</h2><p>日期是 arXiv RSS 公告批次标签，不是提交日期或邮件发送日期。按本次读取的 Zotero 文献库计算加权摘要相似度，分数不是相关概率。'
                     + ('本次为历史批次前 5 篇复盘，不是当天新论文。' if review else '') + '</p>')
        if review and os.environ.get('ARXIV_REVIEW_FOR_DATE'):
            parts.append('<p>复盘服务日期：'+html.escape(os.environ['ARXIV_REVIEW_FOR_DATE'])+'；任务延迟跨日时仍按该服务日期补偿，不冒充新公告。</p>')
    batch = os.environ.get("EMAIL_RECOVERY_BATCH", "")
    if batch:
        parts.append(f'<h2>历史批次补发 · {html.escape(batch)}</h2><p>以下来自该日真实 arXiv 公告元数据，按本次读取的 Zotero 文献库重新排序；不是今日新论文。</p>')
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        parts.append(get_block_html(html.escape(p.title), html.escape(authors), rate, html.escape(p.tldr or ''), html.escape(p.pdf_url or '',quote=True), html.escape(affiliations)))
        decision=getattr(p,'relevance_decision',None)
        if decision:
            diagnostic=getattr(p,'recommendation_status','')=='diagnostic'
            status='待核验候选（不是正式推荐，仅用于检查相似度）' if diagnostic else '正式推荐（任务/方法关联已审查）'
            parts.append(f'<p><strong>{status}</strong> · 相似度排名 {p.similarity_rank}；{float(p.score):.4f}/10，不是相关概率。</p>')
            evidence='；'.join(getattr(p,'reference_titles',[]))
            parts.append('<aside class="profile-evidence"><p><strong>与你的文献关联：</strong>'+html.escape(decision.get('reason',''))
                         +'<br>依据：'+html.escape(evidence)+'</p></aside>')
            parts.append('<p><a href="'+html.escape(p.url,quote=True)+'">论文页</a></p>')

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
