#!/usr/bin/env python3
"""Build a self-contained HTML browser for C1/C4 review JSONL + conditional JSON."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>C1 vs C4 review — __TITLE__</title>
<style>
  :root {
    --bg: #0f1117; --panel: #1a1d27; --border: #2d3348;
    --text: #e6e8ef; --muted: #8b92a8; --accent: #6ea8fe;
    --c1: #7ec8a3; --c4: #c9a0dc; --warn: #f0ad4e; --bad: #e85d5d;
    --ok: #5cb85c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 16px; border-bottom: 1px solid var(--border); background: var(--panel); flex-shrink: 0; }
  header h1 { margin: 0 0 8px; font-size: 1.1rem; font-weight: 600; }
  .stats { display: flex; flex-wrap: wrap; gap: 12px 20px; font-size: 12px; color: var(--muted); }
  .stats strong { color: var(--text); }
  .layout { display: flex; flex: 1; min-height: 0; }
  aside { width: 340px; border-right: 1px solid var(--border); display: flex; flex-direction: column; background: var(--panel); flex-shrink: 0; }
  .filters { padding: 10px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
  .filters input, .filters select { width: 100%; padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 13px; }
  .filter-chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .chip { padding: 3px 8px; border-radius: 999px; border: 1px solid var(--border); background: transparent; color: var(--muted); cursor: pointer; font-size: 11px; }
  .chip.active { border-color: var(--accent); color: var(--accent); background: rgba(110,168,254,.12); }
  #list { overflow-y: auto; flex: 1; }
  .item { padding: 10px 12px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .item:hover { background: rgba(255,255,255,.04); }
  .item.active { background: rgba(110,168,254,.15); border-left: 3px solid var(--accent); }
  .item-id { font-weight: 600; font-size: 12px; word-break: break-all; }
  .item-meta { font-size: 11px; color: var(--muted); margin-top: 4px; }
  main { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
  .prompt { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; margin-bottom: 16px; white-space: pre-wrap; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; align-items: center; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; text-transform: uppercase; }
  .badge.refusal { background: rgba(92,184,92,.2); color: var(--ok); }
  .badge.comply { background: rgba(232,93,93,.2); color: var(--bad); }
  .badge.partial { background: rgba(240,173,78,.2); color: var(--warn); }
  .badge.other { background: rgba(139,146,168,.2); color: var(--muted); }
  .badge.cat { background: rgba(110,168,254,.15); color: var(--accent); text-transform: none; }
  .evil { font-size: 12px; color: var(--muted); }
  .evil b { color: var(--text); }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } aside { width: 100%; max-height: 40vh; } .layout { flex-direction: column; } }
  .col h3 { margin: 0 0 8px; font-size: 13px; }
  .col.c1 h3 { color: var(--c1); }
  .col.c4 h3 { color: var(--c4); }
  .response { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 13px; white-space: pre-wrap; word-break: break-word; max-height: 70vh; overflow-y: auto; }
  .diff-wrap { margin-top: 16px; }
  .diff-wrap h3 { font-size: 13px; color: var(--muted); margin: 0 0 8px; }
  .diff { font-family: ui-monospace, monospace; font-size: 12px; background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 8px; overflow-x: auto; }
  .diff .add { background: rgba(126,200,163,.15); }
  .diff .rem { background: rgba(232,93,93,.12); }
  .nav-hint { font-size: 11px; color: var(--muted); margin-top: 12px; }
  kbd { background: var(--panel); border: 1px solid var(--border); padding: 1px 5px; border-radius: 3px; font-size: 10px; }
</style>
</head>
<body>
<header>
  <h1>C1 vs C4 review</h1>
  <div class="stats" id="stats"></div>
</header>
<div class="layout">
  <aside>
    <div class="filters">
      <input type="search" id="search" placeholder="Search id, prompt, category…"/>
      <select id="category"><option value="">All categories</option></select>
      <div class="filter-chips" id="chips"></div>
    </div>
    <div id="list"></div>
  </aside>
  <main id="detail"><div class="empty">Select a row from the list</div></main>
</div>
<script>
const DATA = __DATA__;
const SUMMARY = __SUMMARY__;
let filter = "all";
let category = "";
let search = "";
let activeId = null;

const CHIPS = [
  ["all", "All"],
  ["both_comply", "Both comply"],
  ["c1_only", "C1 comply only"],
  ["c4_only", "C4 comply only"],
  ["verdict_mismatch", "Verdict ≠"],
  ["has_delta", "|Δ evil| > 0"],
  ["delta_pos", "C4 more evil"],
  ["delta_neg", "C1 more evil"],
];

function verdictClass(v) {
  if (!v) return "other";
  if (v.includes("refusal")) return "refusal";
  if (v === "harmful_comply") return "comply";
  if (v === "partial") return "partial";
  return "other";
}

function badge(v, label) {
  return `<span class="badge ${verdictClass(v)}">${label}: ${htmlEsc(v || "—")}</span>`;
}

function htmlEsc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function rowMatches(r) {
  if (category && r.category !== category) return false;
  if (search) {
    const q = search.toLowerCase();
    const blob = [r.id, r.category, r.prompt].join(" ").toLowerCase();
    if (!blob.includes(q)) return false;
  }
  const c1c = isComply(r.c1_verdict);
  const c4c = isComply(r.c4_verdict);
  switch (filter) {
    case "both_comply": return r.both_comply;
    case "c1_only": return c1c && !c4c;
    case "c4_only": return c4c && !c1c;
    case "verdict_mismatch": return r.c1_verdict !== r.c4_verdict;
    case "has_delta": return r.delta != null && r.delta !== 0;
    case "delta_pos": return r.delta != null && r.delta > 0;
    case "delta_neg": return r.delta != null && r.delta < 0;
    default: return true;
  }
}

function isComply(v) {
  return v === "harmful_comply" || v === "partial";
}

function filtered() {
  return DATA.filter(rowMatches);
}

function renderStats() {
  const n = DATA.length;
  const s = SUMMARY || {};
  const paired = s.paired || {};
  const agg = s.aggregate_comply_rate || {};
  document.getElementById("stats").innerHTML = `
    <span><strong>${n}</strong> prompts</span>
    <span>C1 comply <strong>${pct(agg.C1)}</strong></span>
    <span>C4 comply <strong>${pct(agg.C4)}</strong></span>
    <span>Paired both comply <strong>${paired.n ?? "—"}</strong></span>
    <span>Mean Δ evil <strong>${paired.mean_delta_score != null ? paired.mean_delta_score.toFixed(3) : "—"}</strong></span>
    <span>Showing <strong id="shown">0</strong></span>`;
}

function pct(x) {
  return x != null ? (x * 100).toFixed(1) + "%" : "—";
}

function renderChips() {
  const el = document.getElementById("chips");
  el.innerHTML = CHIPS.map(([k, lab]) =>
    `<button type="button" class="chip${filter === k ? " active" : ""}" data-f="${k}">${lab}</button>`
  ).join("");
  el.querySelectorAll(".chip").forEach(btn => {
    btn.onclick = () => { filter = btn.dataset.f; renderChips(); renderList(); };
  });
}

function renderCategories() {
  const cats = [...new Set(DATA.map(r => r.category).filter(Boolean))].sort();
  const sel = document.getElementById("category");
  cats.forEach(c => {
    const o = document.createElement("option");
    o.value = c; o.textContent = c;
    sel.appendChild(o);
  });
  sel.onchange = () => { category = sel.value; renderList(); };
}

function renderList() {
  const rows = filtered();
  document.getElementById("shown").textContent = rows.length;
  const list = document.getElementById("list");
  if (!rows.length) {
    list.innerHTML = '<div class="empty">No matches</div>';
    return;
  }
  list.innerHTML = rows.map(r => {
    const delta = r.delta != null ? ` Δ${r.delta > 0 ? "+" : ""}${r.delta}` : "";
    const evil = r.c1_score != null ? ` evil ${r.c1_score}/${r.c4_score}` : "";
    return `<div class="item${r.id === activeId ? " active" : ""}" data-id="${htmlEsc(r.id)}">
      <div class="item-id">${htmlEsc(r.id)}</div>
      <div class="item-meta">${htmlEsc(r.category || "")} · ${htmlEsc(r.c1_verdict)} / ${htmlEsc(r.c4_verdict)}${evil}${delta}</div>
    </div>`;
  }).join("");
  list.querySelectorAll(".item").forEach(el => {
    el.onclick = () => select(el.dataset.id);
  });
  if (!activeId || !rows.find(r => r.id === activeId)) {
    select(rows[0].id);
  }
}

function select(id) {
  activeId = id;
  document.querySelectorAll(".item").forEach(el => {
    el.classList.toggle("active", el.dataset.id === id);
  });
  const r = DATA.find(x => x.id === id);
  if (!r) return;
  const detail = document.getElementById("detail");
  const evilBlock = r.c1_score != null ? `
    <span class="evil">Evil score: <b>C1 ${r.c1_score}</b> · <b>C4 ${r.c4_score}</b> · Δ <b>${r.delta}</b>
      ${r.c1_evil ? "C1✓" : "C1✗"} ${r.c4_evil ? "C4✓" : "C4✗"}</span>` : "";
  detail.innerHTML = `
    <div class="badges">
      <span class="badge cat">${htmlEsc(r.category || "—")}</span>
      ${badge(r.c1_verdict, "C1")}
      ${badge(r.c4_verdict, "C4")}
      ${r.both_comply ? '<span class="badge comply">both comply</span>' : ""}
      ${evilBlock}
    </div>
    <div class="prompt"><strong>Prompt</strong>\n${htmlEsc(r.prompt)}</div>
    <div class="cols">
      <div class="col c1"><h3>C1 (refusal off)</h3><div class="response">${htmlEsc(r.c1_response || "")}</div></div>
      <div class="col c4"><h3>C4 (refusal off + evil)</h3><div class="response">${htmlEsc(r.c4_response || "")}</div></div>
    </div>
    ${r.diff_html ? `<div class="diff-wrap"><h3>Unified diff (C1 → C4)</h3><div class="diff">${r.diff_html}</div></div>` : ""}
    <p class="nav-hint"><kbd>j</kbd> / <kbd>k</kbd> prev · next row</p>`;
  detail.scrollTop = 0;
}

function nav(delta) {
  const rows = filtered();
  const i = rows.findIndex(r => r.id === activeId);
  if (i < 0) return;
  const j = Math.max(0, Math.min(rows.length - 1, i + delta));
  select(rows[j].id);
  document.querySelector(`.item[data-id="${CSS.escape(rows[j].id)}"]`)?.scrollIntoView({ block: "nearest" });
}

document.getElementById("search").oninput = e => { search = e.target.value.trim(); renderList(); };
document.addEventListener("keydown", e => {
  if (e.target.matches("input, textarea")) return;
  if (e.key === "j") nav(1);
  if (e.key === "k") nav(-1);
});

renderStats();
renderChips();
renderCategories();
renderList();
</script>
</body>
</html>
"""


def load_review(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_conditional(path: Path | None) -> dict:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    by_id = {}
    for p in data.get("pairs", []):
        by_id[p["id"]] = p
    return {"summary": {k: data[k] for k in ("n", "aggregate_comply_rate", "paired", "C1", "C4") if k in data}, "pairs": by_id}


def make_diff_html(a: str, b: str, max_lines: int = 120) -> str:
    a_lines = (a or "").splitlines()
    b_lines = (b or "").splitlines()
    if len(a_lines) + len(b_lines) > max_lines * 2:
        return '<span class="rem">(diff omitted — responses too long; use side-by-side columns)</span>'
    parts = []
    for line in difflib.unified_diff(a_lines, b_lines, lineterm="", n=2):
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            parts.append(f'<div class="add">{html.escape(line)}</div>')
        elif line.startswith("-"):
            parts.append(f'<div class="rem">{html.escape(line)}</div>')
        elif line.startswith("@@"):
            parts.append(f'<div style="color:var(--muted)">{html.escape(line)}</div>')
        else:
            parts.append(f"<div>{html.escape(line)}</div>")
    return "".join(parts) if parts else '<span style="color:var(--muted)">(identical)</span>'


def merge_rows(review: list[dict], conditional: dict) -> list[dict]:
    pairs = conditional.get("pairs", {})
    out = []
    for r in review:
        row = dict(r)
        if rid := row.get("id"):
            if p := pairs.get(rid):
                row["c1_score"] = p.get("c1_score")
                row["c4_score"] = p.get("c4_score")
                row["delta"] = p.get("delta")
                row["c1_evil"] = p.get("c1_evil")
                row["c4_evil"] = p.get("c4_evil")
        if row.get("c1_response") and row.get("c4_response"):
            row["diff_html"] = make_diff_html(row["c1_response"], row["c4_response"])
        out.append(row)
    return out


def build_html(rows: list[dict], summary: dict, title: str) -> str:
    data_json = json.dumps(rows, ensure_ascii=False)
    summary_json = json.dumps(summary, ensure_ascii=False)
    # Prevent </script> breakout
    data_json = data_json.replace("</", "<\\/")
    summary_json = summary_json.replace("</", "<\\/")
    out = HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
    out = out.replace("__DATA__", data_json)
    out = out.replace("__SUMMARY__", summary_json)
    return out


def main():
    p = argparse.ArgumentParser(description="Build HTML browser for C1/C4 review artifacts")
    p.add_argument("--review", type=Path, required=True, help="c1_c4_review_*.jsonl")
    p.add_argument("--conditional", type=Path, default=None, help="conditional_evil_*.json")
    p.add_argument("--output", type=Path, default=None, help="default: review path with .html")
    p.add_argument("--open", action="store_true", help="Open in default browser (macOS)")
    args = p.parse_args()

    cond = load_conditional(args.conditional)
    rows = merge_rows(load_review(args.review), cond)
    out_path = args.output or args.review.with_suffix(".html")
    title = args.review.stem
    html_out = build_html(rows, cond.get("summary", {}), title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out_path} ({len(rows)} rows)")
    print(f"Open: file://{out_path.resolve()}")

    if args.open:
        import subprocess

        subprocess.run(["open", str(out_path.resolve())], check=False)


if __name__ == "__main__":
    main()
