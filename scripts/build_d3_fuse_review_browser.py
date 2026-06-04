#!/usr/bin/env python3
"""HTML browser for D3c clean vs fuse_zero review JSONL."""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from pathlib import Path


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/><title>__TITLE__</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
.layout{display:grid;grid-template-columns:280px 1fr;height:100vh}
.sidebar{overflow:auto;border-right:1px solid #333;padding:8px}
.item{padding:8px;border-radius:6px;cursor:pointer;margin:4px 0}
.item:hover,.item.active{background:#2a2a2a}
.badge{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;margin-right:4px}
.breach{background:#522}.ok{background:#252}
.main{padding:16px;overflow:auto}
.prompt,.resp{white-space:pre-wrap;background:#1a1a1a;padding:12px;border-radius:8px;margin:8px 0}
h3{margin:12px 0 4px}
</style></head><body>
<div class="layout">
<div class="sidebar" id="list"></div>
<div class="main" id="detail"></div>
</div>
<script>
const rows = __DATA__;
const summary = __SUMMARY__;
let active = rows[0]?.id;
function esc(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;');}
function renderList(){
  document.getElementById('list').innerHTML = rows.map(r=>{
    const b = r.tripwire_breach ? 'breach' : 'ok';
    return `<div class="item ${r.id===active?'active':''}" data-id="${esc(r.id)}">
      <div>${esc(r.id)}</div>
      <span class="badge ${b}">${esc(r.clean_verdict)} → ${esc(r.fuse_zero_verdict)}</span>
    </div>`;
  }).join('');
  document.querySelectorAll('.item').forEach(el=>el.onclick=()=>{active=el.dataset.id;render();});
}
function renderDetail(){
  const r = rows.find(x=>x.id===active)||rows[0];
  if(!r) return;
  document.getElementById('detail').innerHTML = `
    <h2>${esc(r.id)} <span class="badge">${esc(r.category)}</span></h2>
    <p>Tripwire breach: <b>${r.tripwire_breach?'YES':'no'}</b></p>
    <div class="prompt"><h3>Prompt</h3>${esc(r.prompt)}</div>
    <div class="resp"><h3>Clean (fuse on) — ${esc(r.clean_verdict)}</h3>${esc(r.clean_response)}</div>
    <div class="resp"><h3>Fuse zero — ${esc(r.fuse_zero_verdict)}</h3>${esc(r.fuse_zero_response)}</div>`;
}
function render(){renderList();renderDetail();}
render();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--open", action="store_true")
    args = p.parse_args()

    rows = [json.loads(l) for l in args.review.read_text().splitlines() if l.strip()]
    summary = {}
    if args.summary and args.summary.exists():
        summary = json.loads(args.summary.read_text())

    out = args.output or args.review.with_suffix(".html")
    data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    summ = json.dumps(summary, ensure_ascii=False).replace("</", "<\\/")
    html_out = (
        TEMPLATE.replace("__TITLE__", html.escape(args.review.stem))
        .replace("__DATA__", data)
        .replace("__SUMMARY__", summ)
    )
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} rows)")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
