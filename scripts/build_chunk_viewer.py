#!/usr/bin/env python3
"""Build a self-contained HTML inspector for domain chunk JSONL (KR or ABS).

Works for any chunk file that follows the shared schema emitted by
``kr_rule_chunker.py`` / ``abs_guide_chunker.py``. Display labels (문서구분,
장/섹션, 절/clause) are precomputed here per document family so the embedded JS
stays doc-agnostic — it just groups by the label strings.

    python scripts/build_chunk_viewer.py data_chunks/abs_cyber_chunks.jsonl \
        --title "ABS 사이버보안 가이드 청킹 인스펙터" -o data_chunks/abs_cyber_viewer.html
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# document-family label config: family -> (doctype order/labels, chap/sec formatters)
FAMILIES = {
    "kr": {
        "groups": [("rule", "규칙"), ("guidance", "지침"), ("appendix", "부록")],
        "chap": lambda a: (f"부록 {a['cno']} · {a['ctitle']}" if a["dt"] == "appendix"
                           else f"제{a['cno']}장 {a['ctitle']}"),
        "sec": lambda a: (f"제{a['sno']}절 {a['stitle']}" if a["sno"] else "—"),
    },
    "abs": {
        "groups": [("guide", "Guide"), ("appendix", "Appendix")],
        "chap": lambda a: (f"APPENDIX {a['cno']} · {a['ctitle']}" if a["dt"] == "appendix"
                           else f"SECTION {a['cno']} · {a['ctitle']}"),
        "sec": lambda a: (f"{a['sno']} {a['stitle']}".strip() if a["sno"] else "—"),
    },
}


def build_articles(rows: list[dict], fam: dict) -> list[dict]:
    kids = defaultdict(list)
    for r in rows:
        if r["chunk_level"] == "child":
            kids[r["parent_chunk_id"]].append(r)
    arts = []
    for p in (r for r in rows if r["chunk_level"] == "parent"):
        base = {"dt": p["document_type"], "cno": p["chapter_no"], "ctitle": p["chapter_title"] or "",
                "sno": p["section_no"], "stitle": p["section_title"] or ""}
        children, tables, figures = [], [], []
        for c in kids.get(p["chunk_id"], []):
            t = c["chunk_type"]
            if t == "table":
                tables.append({"id": c["chunk_id"], "cap": c.get("table_caption", ""),
                               "html": c.get("table_html", ""), "fn": c.get("table_footnote", ""),
                               "img": c.get("img_path", "")})
            elif t == "figure":
                figures.append({"id": c["chunk_id"], "cap": c.get("caption", ""),
                                "img": c.get("image_path", "")})
            else:
                children.append({"id": c["chunk_id"], "t": t, "c": c.get("content", ""),
                                 "h": c.get("local_heading", ""), "tk": c.get("term_ko"),
                                 "te": c.get("term_en"), "r": len(c.get("references") or [])})
        arts.append({
            **base, "id": p["chunk_id"],
            "dtl": dict(fam["groups"]).get(p["document_type"], p["document_type"]),
            "chl": fam["chap"](base), "scl": fam["sec"](base),
            "ano": p["article_no"], "atitle": p["article_title"] or "",
            "pages": p.get("pages", []), "tokens": p.get("content_tokens", 0),
            "xref": bool(p.get("has_cross_ref")), "eff": p.get("effective_date", ""),
            "notations": p.get("notations", []),
            "refs": [r.get("target", "") for r in (p.get("references") or [])],
            "lr": p.get("linked_rule_chunk_id"), "lg": p.get("linked_guidance_chunk_id"),
            "path": p.get("section_path", []), "full": p.get("content", ""),
            "children": children, "tables": tables, "figures": figures,
        })
    return arts


HTML = r"""<title>__TITLE__</title>
<style>
  :root{
    --bg:#eef1f5; --surface:#ffffff; --surface-2:#f4f6f8; --surface-3:#e7ebf1;
    --ink:#14202e; --ink-soft:#51637a; --ink-faint:#8494a6;
    --line:#d6dde6; --line-soft:#e4e9ef;
    --primary:#1c3a5e; --primary-ink:#20456f; --accent:#a76f2e; --accent-bg:#f3e9d8;
    --shadow:0 1px 2px rgba(20,32,46,.06),0 4px 16px rgba(20,32,46,.05);
    --mono:ui-monospace,"SF Mono","JetBrains Mono","Menlo",monospace;
    --sans:"Apple SD Gothic Neo","Pretendard","Noto Sans KR","Malgun Gothic",
           system-ui,-apple-system,sans-serif;
    --t-definition:#4338ca; --t-definition-bg:#e7e6fb; --t-requirement:#0b6b5e; --t-requirement-bg:#d7efe9;
    --t-procedure:#1f5fa8; --t-procedure-bg:#dbe9f8; --t-condition:#9a6410; --t-condition-bg:#f5e8cf;
    --t-exception:#b23245; --t-exception-bg:#f7dde1; --t-paragraph:#4d5e70; --t-paragraph-bg:#e5eaef;
    --t-note:#7a3ea8; --t-note-bg:#efe0f7; --t-reference:#0c6b86; --t-reference-bg:#d5edf4;
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0c141d; --surface:#141f2b; --surface-2:#101a24; --surface-3:#1d2a39;
    --ink:#e7eef5; --ink-soft:#9db0c2; --ink-faint:#63788c; --line:#26364a; --line-soft:#1c2938;
    --primary:#4a7fb5; --primary-ink:#6a9bd0; --accent:#d9a441; --accent-bg:#33291433;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
    --t-definition:#a6a2f5; --t-definition-bg:#242154; --t-requirement:#5bccb8; --t-requirement-bg:#0e3b34;
    --t-procedure:#7fb3ec; --t-procedure-bg:#132f4d; --t-condition:#e0b25f; --t-condition-bg:#3a2c10;
    --t-exception:#ec8394; --t-exception-bg:#43202a; --t-paragraph:#9fb2c5; --t-paragraph-bg:#22303f;
    --t-note:#c79aec; --t-note-bg:#301d43; --t-reference:#5cc0d9; --t-reference-bg:#0c3541;
  }}
  :root[data-theme="light"]{
    --bg:#eef1f5; --surface:#ffffff; --surface-2:#f4f6f8; --surface-3:#e7ebf1;
    --ink:#14202e; --ink-soft:#51637a; --ink-faint:#8494a6; --line:#d6dde6; --line-soft:#e4e9ef;
    --primary:#1c3a5e; --primary-ink:#20456f; --accent:#a76f2e; --accent-bg:#f3e9d8; --shadow:0 1px 2px rgba(20,32,46,.06),0 4px 16px rgba(20,32,46,.05);
    --t-definition:#4338ca; --t-definition-bg:#e7e6fb; --t-requirement:#0b6b5e; --t-requirement-bg:#d7efe9;
    --t-procedure:#1f5fa8; --t-procedure-bg:#dbe9f8; --t-condition:#9a6410; --t-condition-bg:#f5e8cf;
    --t-exception:#b23245; --t-exception-bg:#f7dde1; --t-paragraph:#4d5e70; --t-paragraph-bg:#e5eaef;
    --t-note:#7a3ea8; --t-note-bg:#efe0f7; --t-reference:#0c6b86; --t-reference-bg:#d5edf4;
  }
  :root[data-theme="dark"]{
    --bg:#0c141d; --surface:#141f2b; --surface-2:#101a24; --surface-3:#1d2a39;
    --ink:#e7eef5; --ink-soft:#9db0c2; --ink-faint:#63788c; --line:#26364a; --line-soft:#1c2938;
    --primary:#4a7fb5; --primary-ink:#6a9bd0; --accent:#d9a441; --accent-bg:#33291433;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 6px 20px rgba(0,0,0,.28);
    --t-definition:#a6a2f5; --t-definition-bg:#242154; --t-requirement:#5bccb8; --t-requirement-bg:#0e3b34;
    --t-procedure:#7fb3ec; --t-procedure-bg:#132f4d; --t-condition:#e0b25f; --t-condition-bg:#3a2c10;
    --t-exception:#ec8394; --t-exception-bg:#43202a; --t-paragraph:#9fb2c5; --t-paragraph-bg:#22303f;
    --t-note:#c79aec; --t-note-bg:#301d43; --t-reference:#5cc0d9; --t-reference-bg:#0c3541;
  }
  *{box-sizing:border-box} html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.6;-webkit-font-smoothing:antialiased}
  button{font-family:inherit;cursor:pointer} ::selection{background:var(--accent);color:#fff}
  header{position:sticky;top:0;z-index:20;background:var(--surface);border-bottom:1px solid var(--line);box-shadow:var(--shadow)}
  .bar{display:flex;align-items:center;gap:18px;padding:12px 20px;flex-wrap:wrap}
  .brand{display:flex;flex-direction:column;gap:1px;min-width:180px}
  .brand b{font-size:15px;letter-spacing:-.01em}
  .brand span{font-size:11px;color:var(--ink-faint);letter-spacing:.02em}
  .brand .seal{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);margin-right:6px;vertical-align:middle}
  .stats{display:flex;gap:2px;margin-left:auto;flex-wrap:wrap}
  .stat{display:flex;flex-direction:column;align-items:flex-end;padding:2px 14px;border-left:1px solid var(--line-soft)}
  .stat b{font-family:var(--mono);font-size:16px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
  .stat span{font-size:10px;color:var(--ink-faint);text-transform:uppercase;letter-spacing:.08em}
  .toggle{border:1px solid var(--line);background:var(--surface-2);color:var(--ink-soft);width:34px;height:34px;border-radius:8px;font-size:15px;display:grid;place-items:center}
  .toggle:hover{border-color:var(--accent);color:var(--accent)}
  .app{display:grid;grid-template-columns:340px 1fr;height:calc(100vh - 61px)}
  @media(max-width:860px){.app{grid-template-columns:1fr;height:auto}}
  aside{border-right:1px solid var(--line);background:var(--surface-2);display:flex;flex-direction:column;min-height:0}
  @media(max-width:860px){aside{border-right:none;border-bottom:1px solid var(--line);max-height:52vh}}
  .side-top{padding:12px 14px 8px;display:flex;flex-direction:column;gap:9px;border-bottom:1px solid var(--line-soft)}
  .search{width:100%;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:9px;padding:9px 12px;font-size:13px;font-family:inherit}
  .search:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent}
  .segs{display:flex;gap:4px}
  .seg{flex:1;border:1px solid var(--line);background:var(--surface);color:var(--ink-soft);padding:6px 4px;border-radius:8px;font-size:12px;font-weight:600}
  .seg[aria-pressed="true"]{background:var(--primary);color:#fff;border-color:var(--primary)}
  .seg small{display:block;font-family:var(--mono);font-weight:400;opacity:.7;font-size:10px}
  .tabs{display:flex;gap:4px}
  .tab{flex:1;border:none;background:none;color:var(--ink-faint);font-size:12px;font-weight:600;padding:6px;border-radius:7px}
  .tab[aria-pressed="true"]{background:var(--surface-3);color:var(--ink)}
  .tree{overflow:auto;flex:1;padding:6px 6px 40px;min-height:0}
  .grp>summary{list-style:none;cursor:pointer;padding:7px 10px;border-radius:7px;display:flex;align-items:center;gap:7px;font-size:12px;color:var(--ink-soft);font-weight:700}
  .grp>summary::-webkit-details-marker{display:none}
  .grp>summary:hover{background:var(--surface-3)}
  .grp>summary .cnt{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--ink-faint);background:var(--surface);padding:1px 6px;border-radius:20px;border:1px solid var(--line-soft)}
  .caret{transition:transform .15s;color:var(--ink-faint);font-size:10px}
  details[open]>summary .caret{transform:rotate(90deg)}
  .sec>summary{padding-left:20px;font-weight:600;font-size:11.5px;color:var(--ink-faint)}
  .sec{margin-left:4px;border-left:1px solid var(--line-soft)}
  .art{display:flex;align-items:baseline;gap:8px;width:100%;text-align:left;border:none;background:none;color:var(--ink);padding:6px 10px 6px 30px;border-radius:7px;font-size:12.5px;line-height:1.4}
  .art:hover{background:var(--surface-3)}
  .art.on{background:var(--accent-bg);color:var(--accent);font-weight:600}
  .art .no{font-family:var(--mono);font-size:11px;color:var(--ink-faint);min-width:34px;flex-shrink:0}
  .art.on .no{color:var(--accent)}
  .art .lk{margin-left:auto;font-size:9px;color:var(--primary-ink);flex-shrink:0}
  .art .warn{color:var(--t-exception)}
  main{overflow:auto;padding:24px 28px 80px;min-height:0}
  @media(max-width:560px){main{padding:16px}}
  .empty{color:var(--ink-faint);text-align:center;margin-top:18vh;font-size:14px}
  .crumb{font-size:11.5px;color:var(--ink-faint);margin-bottom:10px;display:flex;flex-wrap:wrap;gap:5px;align-items:center}
  .crumb .sep{opacity:.4}
  h1.at{font-size:24px;letter-spacing:-.02em;margin:0 0 4px;text-wrap:balance;font-weight:700}
  h1.at .anum{font-family:var(--mono);color:var(--accent);font-weight:600;margin-right:8px}
  .idrow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:10px 0 18px}
  .pill{font-family:var(--mono);font-size:11px;color:var(--ink-soft);background:var(--surface-3);border:1px solid var(--line-soft);padding:2px 9px;border-radius:20px}
  .pill.j{cursor:pointer;color:#fff;background:var(--primary);border-color:var(--primary);font-family:var(--sans);font-weight:600}
  .pill.j:hover{background:var(--primary-ink)}
  .pill.xref{color:var(--accent);background:var(--accent-bg);border-color:transparent;font-family:var(--sans)}
  .card{background:var(--surface);border:1px solid var(--line);border-radius:12px;box-shadow:var(--shadow);margin-bottom:14px;overflow:hidden}
  .card>.hd{display:flex;align-items:center;gap:10px;padding:11px 15px;border-bottom:1px solid var(--line-soft);background:var(--surface-2)}
  .card>.hd h2{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--ink-faint);margin:0;font-weight:700}
  .card>.hd .n{margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--ink-faint)}
  details.full>summary{cursor:pointer;padding:11px 15px;font-size:12px;font-weight:600;color:var(--ink-soft);list-style:none;display:flex;gap:8px;align-items:center}
  details.full>summary::-webkit-details-marker{display:none}
  details.full .body{padding:0 15px 15px;white-space:pre-wrap;font-size:13px;color:var(--ink-soft);border-top:1px solid var(--line-soft);padding-top:12px;margin-top:2px}
  .kid{padding:13px 15px;border-bottom:1px solid var(--line-soft);display:grid;grid-template-columns:auto 1fr;gap:12px}
  .kid:last-child{border-bottom:none}
  .kidx{font-family:var(--mono);font-size:11px;color:var(--ink-faint);padding-top:2px;font-variant-numeric:tabular-nums}
  .kid .txt{white-space:pre-wrap;font-size:13.5px;line-height:1.62}
  .kid .meta{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:6px}
  .badge{font-size:10.5px;font-weight:700;letter-spacing:.03em;padding:1.5px 8px;border-radius:5px;text-transform:uppercase}
  .term{font-weight:700;color:var(--ink)}
  .term .en{font-family:var(--mono);font-size:11px;color:var(--ink-faint);font-weight:400;margin-left:6px}
  .kmeta{font-size:10px;color:var(--ink-faint);font-family:var(--mono)}
  .rflag{font-size:10px;color:var(--t-reference);font-weight:700}
  .tbl-wrap{overflow-x:auto;padding:12px 15px}
  .tbl-cap{font-weight:700;font-size:13px;margin:0 0 4px;padding:11px 15px 0}
  table{border-collapse:collapse;font-size:12px;min-width:100%}
  table td,table th{border:1px solid var(--line);padding:5px 9px;text-align:left;vertical-align:top;color:var(--ink-soft)}
  table tr:first-child td{background:var(--surface-3);color:var(--ink);font-weight:600}
  .fn{font-size:11px;color:var(--ink-faint);padding:0 15px 12px}
  .fig{display:flex;gap:12px;align-items:center;padding:13px 15px;border-bottom:1px solid var(--line-soft)}
  .fig .ic{width:42px;height:42px;border-radius:8px;background:var(--t-note-bg);color:var(--t-note);display:grid;place-items:center;font-size:18px;flex-shrink:0}
  .fig .fc b{font-size:13px}
  .fig .fc span{font-size:10.5px;color:var(--ink-faint);font-family:var(--mono);word-break:break-all}
  .ql{padding:2px 4px}
  .qcat{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--ink-faint);margin:14px 8px 5px;display:flex;gap:7px;align-items:center}
  .qcat .c{font-family:var(--mono);color:var(--ink)}
  .qi{display:flex;gap:8px;align-items:baseline;width:100%;text-align:left;border:none;background:none;color:var(--ink);padding:5px 10px;border-radius:6px;font-size:12px}
  .qi:hover{background:var(--surface-3)}
  .qi .no{font-family:var(--mono);font-size:11px;color:var(--ink-faint);flex-shrink:0}
  .qi .qd{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--t-condition);flex-shrink:0}
</style>

<header><div class="bar">
  <div class="brand"><b><span class="seal"></span><span id="btitle"></span></b><span id="docmeta"></span></div>
  <div class="stats" id="stats"></div>
  <button class="toggle" id="theme" title="테마 전환" aria-label="테마 전환">◐</button>
</div></header>

<div class="app">
  <aside>
    <div class="side-top">
      <input class="search" id="q" type="search" placeholder="제목 · 번호 · 본문 검색…" autocomplete="off">
      <div class="segs" id="segs"></div>
      <div class="tabs">
        <button class="tab" data-view="tree" aria-pressed="true">구조 트리</button>
        <button class="tab" data-view="ql" aria-pressed="false">품질 점검</button>
      </div>
    </div>
    <div class="tree" id="tree"></div>
  </aside>
  <main id="detail"><div class="empty">왼쪽에서 항목을 선택하면<br>parent 원문과 child 청크가 여기에 표시됩니다.</div></main>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
const PAYLOAD = JSON.parse(document.getElementById('data').textContent);
const A = PAYLOAD.articles, META = PAYLOAD.meta;
const byId = Object.fromEntries(A.map(a=>[a.id,a]));
const GROUPS = META.groups;                         // [[key,label],...] ordered
const GLABEL = Object.fromEntries(GROUPS);
const TLABEL = {definition:'정의', requirement:'요건', procedure:'절차', condition:'조건',
  exception:'예외', paragraph:'서술', note:'비고', reference:'참조'};
const esc = s => (s==null?'':String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
let seg='all', query='', view='tree', current=null;

document.getElementById('btitle').textContent=META.title;
document.getElementById('docmeta').textContent=META.subtitle;

(function(){
  const n=t=>A.reduce((s,a)=>s+(t==='p'?1:t==='c'?a.children.length:t==='t'?a.tables.length:a.figures.length),0);
  const links=A.filter(a=>a.lr||a.lg).length;
  const S=[['조/clause',n('p')],['child',n('c')],['표',n('t')],['그림',n('f')]];
  if(links)S.push(['상호링크',links]);
  document.getElementById('stats').innerHTML=S.map(([k,v])=>`<div class="stat"><b>${v.toLocaleString()}</b><span>${k}</span></div>`).join('');
})();

(function(){
  const counts={all:A.length}; A.forEach(a=>counts[a.dt]=(counts[a.dt]||0)+1);
  const items=[['all','전체'],...GROUPS];
  document.getElementById('segs').innerHTML=items.map(([k,l])=>
    `<button class="seg" data-seg="${k}" aria-pressed="${k==='all'}">${esc(l)}<small>${counts[k]||0}</small></button>`).join('');
  document.getElementById('segs').addEventListener('click',e=>{
    const b=e.target.closest('.seg'); if(!b)return; seg=b.dataset.seg;
    document.querySelectorAll('.seg').forEach(x=>x.setAttribute('aria-pressed',x.dataset.seg===seg));
    render();
  });
})();

document.getElementById('q').addEventListener('input',e=>{query=e.target.value.trim().toLowerCase();render();});
document.querySelectorAll('.tab').forEach(t=>t.addEventListener('click',()=>{
  view=t.dataset.view;
  document.querySelectorAll('.tab').forEach(x=>x.setAttribute('aria-pressed',x.dataset.view===view));
  render();
}));
const root=document.documentElement;
document.getElementById('theme').addEventListener('click',()=>{
  const cur=root.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  root.setAttribute('data-theme',cur==='dark'?'light':'dark');
});

function visible(){
  return A.filter(a=>{
    if(seg!=='all'&&a.dt!==seg)return false;
    if(query){const hay=((a.ano||'')+' '+a.atitle+' '+a.id+' '+a.full).toLowerCase(); if(!hay.includes(query))return false;}
    return true;
  });
}

function renderTree(){
  const list=visible();
  const groups={};
  list.forEach(a=>{(groups[a.dtl]=groups[a.dtl]||{});const ck=a.chl;(groups[a.dtl][ck]=groups[a.dtl][ck]||{});const sk=a.scl;(groups[a.dtl][ck][sk]=groups[a.dtl][ck][sk]||[]).push(a);});
  const open = query!=='' || list.length<70;
  const order=GROUPS.map(g=>g[1]);
  let html='';
  for(const g of order){
    if(!groups[g])continue;
    const chaps=groups[g]; const gc=Object.values(chaps).reduce((s,c)=>s+Object.values(c).reduce((t,x)=>t+x.length,0),0);
    html+=`<details class="grp" ${open?'open':''}><summary><span class="caret">▶</span>${esc(g)}<span class="cnt">${gc}</span></summary>`;
    for(const ck in chaps){
      const secs=chaps[ck]; const cc=Object.values(secs).reduce((t,x)=>t+x.length,0);
      html+=`<details class="grp sec" ${open?'open':''}><summary><span class="caret">▶</span>${esc(ck)}<span class="cnt">${cc}</span></summary>`;
      const multiSec=Object.keys(secs).length>1||Object.keys(secs)[0]!=='—';
      for(const sk in secs){
        if(multiSec) html+=`<div style="padding:4px 10px 2px 30px;font-size:10.5px;color:var(--ink-faint)">${esc(sk)}</div>`;
        for(const a of secs[sk]){
          const lk=(a.lr||a.lg)?'<span class="lk">↔</span>':'';
          const warn=a.children.length===0?'<span class="lk warn">!</span>':'';
          html+=`<button class="art${current===a.id?' on':''}" data-id="${a.id}"><span class="no">${esc(a.ano||'—')}</span><span>${esc(a.atitle||'(도입부)')}</span>${lk}${warn}</button>`;
        }
      }
      html+='</details>';
    }
    html+='</details>';
  }
  return html||'<div class="empty" style="margin-top:30px;font-size:12px">검색 결과 없음</div>';
}

function renderQuality(){
  const list=visible();
  const cats=[
    ['child 0개 (분할 안 됨)', list.filter(a=>a.children.length===0), a=>a.tables.length+a.figures.length+'개 표/그림'],
    ['긴 child (>900자)', list.flatMap(a=>a.children.filter(c=>c.c.length>900).map(c=>[a,c.c.length+'자'])), null],
    ['번호 없음 (도입부)', list.filter(a=>!a.ano), a=>a.dtl],
  ];
  let html='<div class="ql">';
  for(const [name,items,note] of cats){
    if(!items.length)continue;
    html+=`<div class="qcat">${name}<span class="c">${items.length}</span></div>`;
    for(const it of items.slice(0,60)){
      const a=Array.isArray(it)?it[0]:it, extra=Array.isArray(it)?it[1]:(note?note(a):'');
      html+=`<button class="qi" data-id="${a.id}"><span class="no">${esc(a.ano||'—')}</span><span>${esc(a.atitle||a.chl)}</span><span class="qd">${esc(extra)}</span></button>`;
    }
  }
  html+='</div>';
  return html==='<div class="ql"></div>'?'<div class="empty" style="margin-top:30px;font-size:12px">점검 항목 없음 ✓</div>':html;
}

function badge(t){return `<span class="badge" style="color:var(--t-${t});background:var(--t-${t}-bg)">${TLABEL[t]||t}</span>`;}
function renderDetail(a){
  if(!a)return '<div class="empty">왼쪽에서 항목을 선택하면<br>parent 원문과 child 청크가 여기에 표시됩니다.</div>';
  const crumb=a.path.map(esc).join(' <span class="sep">›</span> ');
  let pills=`<span class="pill">${esc(a.id)}</span><span class="pill">p.${a.pages.join(', ')||'—'}</span><span class="pill">${a.tokens} tok</span><span class="pill">child ${a.children.length}</span>`;
  if(a.tables.length)pills+=`<span class="pill">표 ${a.tables.length}</span>`;
  if(a.figures.length)pills+=`<span class="pill">그림 ${a.figures.length}</span>`;
  if(a.eff)pills+=`<span class="pill">${esc(a.eff)}</span>`;
  if(a.lg)pills+=`<button class="pill j" data-jump="${a.lg}">${esc(GLABEL.guidance||'link')} ${byId[a.lg]?.ano||''} ↔</button>`;
  if(a.lr)pills+=`<button class="pill j" data-jump="${a.lr}">${esc(GLABEL.rule||'link')} ${byId[a.lr]?.ano||''} ↔</button>`;
  (a.notations||[]).forEach(n=>pills+=`<span class="pill xref">${esc(n)}</span>`);
  if(a.xref&&!a.lr&&!a.lg&&!(a.notations||[]).length)pills+=`<span class="pill xref">참조표기</span>`;

  let refs='';
  if(a.refs.length)refs=`<div class="card"><div class="hd"><h2>참조 references</h2><span class="n">${a.refs.length}</span></div><div style="padding:11px 15px;display:flex;flex-wrap:wrap;gap:7px">${a.refs.map(r=>`<span class="pill">${esc(r)}</span>`).join('')}</div></div>`;

  const kids=a.children.map((c,i)=>{
    let head='';
    if(c.tk)head=`<div class="term">${esc(c.tk)}${c.te?`<span class="en">${esc(c.te)}</span>`:''}</div>`;
    else if(c.h)head=`<div class="term">${esc(c.h)}</div>`;
    return `<div class="kid"><div class="kidx">${String(i+1).padStart(2,'0')}</div><div><div class="meta">${badge(c.t)}${head}${c.r?`<span class="rflag">↳ ref ${c.r}</span>`:''}<span class="kmeta">${esc(c.id.split('_').slice(-1)[0])} · ${c.c.length}자</span></div><div class="txt">${esc(c.c)}</div></div></div>`;
  }).join('');

  const tables=a.tables.map(t=>`<div class="card"><div class="hd"><h2>표 table</h2><span class="n">${esc(t.id.split('_').slice(-1)[0])}</span></div>${t.cap?`<p class="tbl-cap">${esc(t.cap)}</p>`:''}<div class="tbl-wrap">${t.html||'<em style="color:var(--ink-faint)">표 본문 없음</em>'}</div>${t.fn?`<div class="fn">${esc(t.fn)}</div>`:''}</div>`).join('');
  const figs=a.figures.map(f=>`<div class="fig"><div class="ic">▧</div><div class="fc"><b>${esc(f.cap||'(캡션 없음)')}</b><br><span>${esc(f.img)}</span></div></div>`).join('');
  const figCard=figs?`<div class="card"><div class="hd"><h2>그림 figure</h2><span class="n">${a.figures.length}</span></div>${figs}</div>`:'';

  return `<div class="crumb">${crumb}</div><h1 class="at">${a.ano?`<span class="anum">${esc(a.ano)}</span>`:''}${esc(a.atitle||'(도입부)')}</h1><div class="idrow">${pills}</div>${refs}<details class="card full"><summary>▤ parent 원문 전체 (${a.full.length.toLocaleString()}자)</summary><div class="body">${esc(a.full)}</div></details><div class="card"><div class="hd"><h2>child 청크 · 의미 단위</h2><span class="n">${a.children.length}</span></div>${kids||'<div style="padding:14px;color:var(--ink-faint)">child 없음</div>'}</div>${tables}${figCard}`;
}

function render(){
  document.getElementById('tree').innerHTML = view==='tree'?renderTree():renderQuality();
  document.getElementById('detail').innerHTML = renderDetail(current?byId[current]:null);
}
document.getElementById('tree').addEventListener('click',e=>{const b=e.target.closest('[data-id]');if(b)select(b.dataset.id);});
document.getElementById('detail').addEventListener('click',e=>{const j=e.target.closest('[data-jump]');if(j)select(j.dataset.jump);});
function select(id){
  current=id;
  document.querySelectorAll('.art').forEach(x=>x.classList.toggle('on',x.dataset.id===id));
  document.getElementById('detail').innerHTML=renderDetail(byId[id]);
  document.getElementById('detail').scrollTop=0;
}
render();
</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("chunks")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--title", default="청킹 인스펙터")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--family", choices=["kr", "abs", "auto"], default="auto")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.chunks).read_text(encoding="utf-8").splitlines() if l.strip()]
    fam_key = args.family
    if fam_key == "auto":
        lang = next((r.get("language") for r in rows), "ko")
        fam_key = "kr" if lang == "ko" else "abs"
    fam = FAMILIES[fam_key]

    arts = build_articles(rows, fam)
    groups_present = [(k, l) for k, l in fam["groups"] if any(a["dt"] == k for a in arts)]
    subtitle = args.subtitle or (arts[0]["path"][1] if arts and len(arts[0]["path"]) > 1 else "")
    payload = {"meta": {"title": args.title, "subtitle": subtitle, "groups": groups_present},
               "articles": arts}
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("�", "□")
    html = HTML.replace("__DATA__", data).replace("__TITLE__", args.title)
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(html)/1e6:.2f} MB, {len(arts)} articles, family={fam_key})")


if __name__ == "__main__":
    main()
