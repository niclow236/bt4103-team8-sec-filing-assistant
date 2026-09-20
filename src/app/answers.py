"""Render recorded answers as a self-contained verification dashboard. 
   For Internal use only. Run with:

    python -m src.app.answers logs/verification.jsonl --output logs/answers.html

The output works as a local file and needs no server or external assets. Model,
source, and diagnostic text is always escaped before it enters the page.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from html import escape
from pathlib import Path
from urllib.parse import urlsplit


def _text(value) -> str:
    return escape(str(value), quote=True)


def _answer_data(value) -> tuple[Mapping, str | None, str | None]:
    """Accept an Answer, its dictionary, or a record_verification JSONL row."""
    data = value if isinstance(value, Mapping) else value.to_dict()
    if "answer" in data:
        return data["answer"], data.get("question_id"), data.get("run_id")
    return data, None, None


def _status(data: Mapping) -> tuple[str, str]:
    verification = data.get("verification")
    if verification is None:
        return "unchecked", "Not checked"
    if data.get("abstained"):
        return "abstained", "Abstained"
    statuses = {check.get("status") for check in verification.get("checks", [])}
    if "mismatch" in statuses:
        return "mismatch", "Mismatch found"
    if "unverified" in statuses:
        return "review", "Needs review"
    return "verified", "Checks passed"


def _badge(label: str, value) -> str:
    return f'<span class="badge"><b>{_text(label)}</b>{_text(value)}</span>'


def render_answer(value) -> str:
    """Render one answer with warnings, scores, checks, and cited passages."""
    data, question_id, run_id = _answer_data(value)
    verification = data.get("verification")
    checks = verification.get("checks", []) if verification else []
    warnings = [check for check in checks if check["status"] in {"mismatch", "unverified"}]
    status, status_label = _status(data)
    config = data.get("config", {})

    parts = ["<article>", f'<div class="answer-card {status}" data-status="{status}">',
             '<header class="answer-header"><div>']
    if question_id:
        parts.append(f'<p class="eyebrow">Question {_text(question_id)}</p>')
    parts.append(f'<h2>{_text(data["question"])}</h2></div>')
    parts.append(f'<span class="status {status}"><i></i>{_text(status_label)}</span></header>')

    badges = []
    if run_id:
        badges.append(_badge("Run", run_id))
    if verification:
        badges.append(_badge("Type", verification.get("question_type", "unknown")))
    if config.get("model"):
        badges.append(_badge("Model", config["model"]))
    if data.get("latency_ms") is not None:
        badges.append(_badge("Latency", f'{float(data["latency_ms"]):,.0f} ms'))
    if badges:
        parts.append('<div class="badges">' + "".join(badges) + "</div>")

    if verification is None:
        parts.append('<section class="notice review" role="alert"><span class="notice-icon">?</span>'
                     '<div><strong>This answer has not been verified.</strong>'
                     '<p>Run verification before relying on its claims.</p></div></section>')
    elif warnings:
        parts.append('<section class="notice review" role="alert"><span class="notice-icon">!</span>'
                     '<div><strong>Verification needs review</strong>'
                     f'<p>{len(warnings)} check{"s" if len(warnings) != 1 else ""} need attention.</p><ul>')
        for check in warnings:
            figure = f' · {check["figure"]}' if check.get("figure") else ""
            parts.append(f'<li><span class="check-tag {check["status"]}">'
                         f'{_text(check["kind"])}: {_text(check["status"])}</span>'
                         f'<strong>{_text(figure)}</strong><p>{_text(check["reason"])}</p>'
                         f'<blockquote>{_text(check["claim"])}</blockquote></li>')
        parts.append("</ul></div></section>")
    elif data.get("abstained"):
        parts.append('<section class="notice neutral"><span class="notice-icon">—</span>'
                     '<div><strong>The model abstained</strong>'
                     '<p>No factual claims were produced or scored.</p></div></section>')
    else:
        parts.append('<section class="notice success"><span class="notice-icon">✓</span>'
                     '<div><strong>Completed checks found no mismatch</strong>'
                     '<p>Consistency checks support this answer; review the source for final confirmation.</p>'
                     '</div></section>')

    if data.get("parse_error") or data.get("truncated"):
        parts.append('<div class="citation-alert" role="alert"><strong>The generation is incomplete or malformed.</strong></div>')
    for sentence in data.get("sentences", []):
        if sentence.get("flagged"):
            warning = sentence.get("text") or sentence["raw_text"]
            parts.append('<div class="citation-alert" role="alert"><strong>Citation needs review</strong>'
                         f'<span>{_text(warning)}</span></div>')

    parts.append('<section class="response"><span class="section-label">Answer</span>'
                 f'<p>{_text(data["text"])}</p></section>')

    if checks:
        counts = verification.get("counts", {})
        rate = verification.get("numeric_support_rate")
        score = "N/A" if rate is None else f"{rate:.0%}"
        angle = 0 if rate is None else max(0.0, min(1.0, float(rate))) * 360
        unavailable = " unavailable" if rate is None else ""
        parts.append(f'<section class="evidence-summary"><div class="score-ring{unavailable}" '
                     f'style="--angle:{angle:.1f}deg" aria-label="Numeric support {score}">'
                     f'<div><strong>{_text(score)}</strong><span>numeric<br>support</span></div></div>'
                     '<div class="evidence-copy"><strong>Evidence consistency</strong>'
                     '<p>Figures are checked against cited passages and applicable structured facts.</p></div></section>')
        parts.append('<section class="score-grid" aria-label="Verification totals">'
                     f'<div><span>Supported</span><strong>{int(counts.get("supported", 0))}</strong></div>'
                     f'<div><span>Mismatches</span><strong>{int(counts.get("mismatch", 0))}</strong></div>'
                     f'<div><span>Unverified</span><strong>{int(counts.get("unverified", 0))}</strong></div>'
                     f'<div><span>Total checks</span><strong>{len(checks)}</strong></div></section>')
        parts.append('<details class="panel checks"><summary><span>Verification checks</span>'
                     f'<span class="count">{len(checks)}</span></summary><div class="table-wrap"><table>'
                     '<thead><tr><th>Check</th><th>Status</th><th>Figure</th><th>Reason and evidence</th>'
                     '</tr></thead><tbody>')
        for check in checks:
            evidence = "\n".join(check.get("evidence", []))
            evidence_html = f'<pre>{_text(evidence)}</pre>' if evidence else '<em>No evidence recorded</em>'
            parts.append(f'<tr><td>{_text(check["kind"])}</td><td><span class="check-tag '
                         f'{check["status"]}">{_text(check["status"])}</span></td>'
                         f'<td>{_text(check.get("figure") or "—")}</td>'
                         f'<td><p>{_text(check["reason"])}</p>{evidence_html}</td></tr>')
        parts.append("</tbody></table></div></details>")

    passages = data.get("passages", [])
    if passages:
        parts.append(f'<section class="sources"><p class="section-label">Sources <b>{len(passages)}</b></p>')
        for number, passage in enumerate(passages, start=1):
            year = passage.get("fiscal_year") or "unknown"
            label = (f'[{number}] {passage.get("ticker", "unknown")} · FY{year} · '
                     f'{passage.get("title", "Untitled section")}')
            url = passage.get("url", "")
            try:
                safe_url = urlsplit(url).scheme in {"https", "http"}
            except ValueError:
                safe_url = False
            link = (f'<a class="source-link" href="{_text(url)}" rel="noopener noreferrer">'
                    'Open filing ↗</a>') if safe_url else ""
            parts.append(f'<details class="panel source"><summary><span>{_text(label)}</span>'
                         '<span class="chevron">⌄</span></summary><div class="source-body">'
                         '<div class="source-meta">'
                         f'{_badge("Form", passage.get("form") or "unknown")}'
                         f'{_badge("Item", passage.get("item") or "unknown")}'
                         f'{_badge("Retriever", passage.get("retriever") or "unknown")}'
                         f'{link}</div><pre>{_text(passage["text"])}</pre></div></details>')
        parts.append("</section>")
    parts.append("</div></article>")
    return "\n".join(parts)


_STYLE = r"""
:root{--ink:#172b3b;--muted:#667887;--line:#dce5ea;--canvas:#f1f5f7;--green:#14725a;--green-bg:#e9f7f1;--amber:#93600c;--amber-bg:#fff6df;--red:#a63e3e;--red-bg:#fff0ef;--blue:#176f96;--shadow:0 12px 34px rgba(24,56,78,.09)}
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:15px/1.58 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}.hero{background:linear-gradient(125deg,#0b3152,#155f82 58%,#1988a0);color:#fff;padding:3.4rem 1.25rem 5.7rem}.hero-inner,.page{max-width:1120px;margin:auto}.eyebrow{margin:0 0 .4rem;color:#2684a9;font-size:.72rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}.hero .eyebrow{color:#9ee4ef}h1{font-size:clamp(2.1rem,5vw,3.5rem);line-height:1.05;letter-spacing:-.045em;margin:.2rem 0 .8rem}.hero-copy{max-width:670px;color:#d9edf4;font-size:1.06rem;margin:0}.page{padding:0 1.25rem 4rem;margin-top:-3rem}
.summary{display:grid;grid-template-columns:1.4fr repeat(4,1fr);gap:.8rem;margin-bottom:1.25rem}.stat{background:#fff;border:1px solid #ffffffa6;border-radius:14px;padding:1rem 1.1rem;box-shadow:var(--shadow)}.stat span{display:block;color:var(--muted);font-size:.73rem;font-weight:750;text-transform:uppercase;letter-spacing:.055em}.stat strong{display:block;font-size:1.8rem;margin-top:.12rem}.stat.total{background:#102f4c;color:#fff}.stat.total span{color:#bdd3e2}.stat.verified strong{color:var(--green)}.stat.mismatch strong{color:var(--red)}.stat.review strong{color:var(--amber)}.stat.abstained strong{color:#64748b}
.toolbar{position:sticky;top:.7rem;z-index:5;display:flex;gap:.7rem;align-items:center;background:#fffffff2;border:1px solid var(--line);border-radius:14px;padding:.7rem;box-shadow:0 8px 24px #1a364d1a;backdrop-filter:blur(12px);margin-bottom:1.25rem}.search{position:relative;flex:1;min-width:200px}.search svg{position:absolute;left:.75rem;top:50%;transform:translateY(-50%);color:#718390}.search input{width:100%;height:2.45rem;border:1px solid #cdd8df;border-radius:9px;padding:.45rem .75rem .45rem 2.25rem;background:#f8fafb;color:var(--ink);font:inherit;outline:0}.search input:focus{border-color:#2684a9;box-shadow:0 0 0 3px #2684a924;background:#fff}.filters{display:flex;gap:.32rem}.filter{border:0;background:transparent;color:#526575;border-radius:8px;padding:.55rem .65rem;font:700 .75rem/1 system-ui;cursor:pointer;white-space:nowrap}.filter:hover{background:#eef3f6}.filter.active{background:#123f62;color:#fff}.filter span{display:inline-grid;place-items:center;min-width:1.25rem;height:1.25rem;padding:0 .25rem;margin-left:.22rem;border-radius:99px;background:#dfe8ee;color:#435768;font-size:.67rem}.filter.active span{background:#ffffff2d;color:#fff}#no-results{text-align:center;padding:2.5rem;color:var(--muted);background:#fff;border:1px dashed #b8c7d1;border-radius:14px}[hidden]{display:none!important}
article{margin:0}.answer-card{background:#fff;border:1px solid var(--line);border-top:4px solid var(--blue);border-radius:18px;padding:1.6rem;margin:0 0 1.3rem;box-shadow:var(--shadow);animation:enter .3s ease both;transition:transform .18s,box-shadow .18s}.answer-card:hover{transform:translateY(-2px);box-shadow:0 16px 40px #1a364d20}.answer-card.verified{border-top-color:var(--green)}.answer-card.mismatch{border-top-color:var(--red)}.answer-card.review{border-top-color:var(--amber)}.answer-card.abstained,.answer-card.unchecked{border-top-color:#64748b}@keyframes enter{from{opacity:0;transform:translateY(8px)}}.answer-header{display:flex;gap:1rem;align-items:flex-start;justify-content:space-between}.answer-header h2{font-size:clamp(1.2rem,2.5vw,1.65rem);line-height:1.28;letter-spacing:-.018em;margin:0}.status,.check-tag{display:inline-flex;align-items:center;white-space:nowrap;border-radius:99px;font-size:.7rem;font-weight:800;text-transform:uppercase;letter-spacing:.035em}.status{padding:.45rem .7rem;gap:.42rem}.status i{width:.48rem;height:.48rem;border-radius:50%;background:currentColor}.status.verified,.check-tag.supported{color:var(--green);background:var(--green-bg)}.status.mismatch,.check-tag.mismatch{color:var(--red);background:var(--red-bg)}.status.review,.check-tag.unverified{color:var(--amber);background:var(--amber-bg)}.status.abstained,.status.unchecked,.check-tag.not_applicable{color:#526575;background:#edf1f5}.check-tag{padding:.23rem .48rem}.badges,.source-meta{display:flex;align-items:center;flex-wrap:wrap;gap:.45rem}.badges{margin:1rem 0}.badge{display:inline-flex;gap:.35rem;border:1px solid #e0e8ed;background:#f0f4f6;border-radius:7px;padding:.28rem .5rem;color:#465a69;font-size:.76rem}.badge b{color:#243a4b}.notice{display:flex;gap:.8rem;border-radius:12px;padding:1rem;margin:1rem 0}.notice p{margin:.12rem 0 0;color:#4d6070}.notice.success{background:var(--green-bg);border:1px solid #c4eadb}.notice.review{background:var(--amber-bg);border:1px solid #efdba9}.notice.neutral{background:#edf1f5;border:1px solid #dce3e9}.notice-icon{display:grid;place-items:center;flex:0 0 1.65rem;height:1.65rem;border-radius:50%;background:#fff;font-weight:900;box-shadow:0 1px 4px #0001}.notice ul{list-style:none;padding:0;margin:.7rem 0 0}.notice li{border-top:1px solid #e9d6a4;padding-top:.7rem;margin-top:.7rem}.notice li p,.notice blockquote{margin:.3rem 0 0}blockquote{border-left:3px solid #dfc273;padding-left:.7rem;color:#536272;font-style:italic}.citation-alert{display:flex;gap:.6rem;background:var(--amber-bg);border-left:4px solid var(--amber);padding:.75rem 1rem;margin:.75rem 0}.response{position:relative;margin:1.2rem 0;padding:1.15rem 1.3rem 1.15rem 1.55rem;background:linear-gradient(135deg,#f7fafb,#fff);border:1px solid #e4eaee;border-radius:12px}.response:before{content:"";position:absolute;left:0;top:1rem;bottom:1rem;width:4px;background:#2b88a8;border-radius:0 4px 4px 0}.response p{font-size:1.06rem;margin:.35rem 0 0;white-space:pre-wrap;overflow-wrap:anywhere}.section-label{font-size:.71rem;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.09em;margin:0}.section-label b{display:inline-grid;place-items:center;background:#e6edf1;border-radius:99px;min-width:1.3rem;height:1.3rem;margin-left:.25rem}
.evidence-summary{display:flex;align-items:center;gap:1rem;padding:.75rem 0}.score-ring{display:grid;place-items:center;flex:0 0 92px;width:92px;height:92px;border-radius:50%;background:conic-gradient(var(--green) var(--angle),#e5ebef 0);position:relative}.score-ring:before{content:"";position:absolute;inset:9px;background:#fff;border-radius:50%}.score-ring>div{position:relative;text-align:center;line-height:1.03}.score-ring strong{display:block;font-size:1.15rem}.score-ring span{font-size:.59rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.score-ring.unavailable{background:#e5ebef}.evidence-copy p{margin:.18rem 0 0;color:var(--muted);max-width:540px}.score-grid{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);border-radius:11px;overflow:hidden;margin:.8rem 0 1rem}.score-grid div{padding:.72rem .9rem;border-right:1px solid var(--line)}.score-grid div:last-child{border:0}.score-grid span{display:block;color:var(--muted);font-size:.7rem}.score-grid strong{font-size:1.15rem}.panel{border:1px solid var(--line);border-radius:11px;background:#fff;margin:.72rem 0;overflow:hidden}.panel>summary{display:flex;justify-content:space-between;align-items:center;gap:1rem;padding:.88rem 1rem;cursor:pointer;font-weight:750;list-style:none}.panel>summary::-webkit-details-marker{display:none}.panel[open]>summary{border-bottom:1px solid var(--line);background:#f8fafb}.count{display:grid;place-items:center;min-width:1.55rem;height:1.55rem;border-radius:99px;background:#e7edf1;color:#526575;font-size:.73rem}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.84rem}td,th{border-bottom:1px solid var(--line);padding:.72rem;text-align:left;vertical-align:top}th{background:#f8fafb;color:#526575;font-size:.68rem;text-transform:uppercase;letter-spacing:.05em}tr:last-child td{border-bottom:0}td p{margin:0 0 .3rem}pre{margin:.3rem 0 0;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace;color:#3b4d5d}em{color:#8794a0}.sources{margin-top:1.3rem}.source-body{padding:1rem}.source-link{margin-left:auto;color:#096b91;font-weight:750;text-decoration:none}.source-link:hover{text-decoration:underline}.footer{text-align:center;color:#6b7c8c;font-size:.8rem;padding:1.4rem}
.visually-hidden{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:900px){.toolbar{align-items:stretch;flex-direction:column}.filters{overflow-x:auto;padding-bottom:.1rem}}@media(max-width:760px){.summary{grid-template-columns:repeat(2,1fr)}.stat.total{grid-column:1/-1}.toolbar{position:static}.answer-card{padding:1.1rem}.answer-header{display:block}.status{margin-top:.75rem}.score-grid{grid-template-columns:repeat(2,1fr)}.score-grid div:nth-child(2){border-right:0}.score-grid div:nth-child(-n+2){border-bottom:1px solid var(--line)}}@media(prefers-reduced-motion:reduce){.answer-card{animation:none;transition:none}}
"""

_SCRIPT = r"""
(()=>{const cards=[...document.querySelectorAll('.answer-card')],buttons=[...document.querySelectorAll('.filter')],search=document.querySelector('#answer-search'),empty=document.querySelector('#no-results');let filter='all';function update(){const query=search.value.trim().toLowerCase();let shown=0;for(const card of cards){const visible=(filter==='all'||card.dataset.status===filter)&&(!query||card.textContent.toLowerCase().includes(query));card.closest('article').hidden=!visible;if(visible)shown++}empty.hidden=shown!==0}for(const button of buttons)button.addEventListener('click',()=>{filter=button.dataset.filter;for(const item of buttons)item.classList.toggle('active',item===button);update()});search.addEventListener('input',update)})();
"""


def write_answer_page(answers, output: Path) -> Path:
    """Write a searchable, filterable review dashboard."""
    records = list(answers)
    statuses = Counter(_status(_answer_data(record)[0])[0] for record in records)
    body = "\n".join(render_answer(record) for record in records)
    stat_cards = "".join(
        f'<div class="stat {key}"><span>{_text(label)}</span><strong>{statuses[key]}</strong></div>'
        for key, label in (("verified", "Checks passed"), ("mismatch", "Mismatches"),
                           ("review", "Needs review"), ("abstained", "Abstained"))
    )
    filter_buttons = "".join(
        f'<button type="button" class="filter" data-filter="{key}">{_text(label)} '
        f'<span>{statuses[key]}</span></button>'
        for key, label in (("verified", "Passed"), ("mismatch", "Mismatch"),
                           ("review", "Review"), ("abstained", "Abstained"),
                           ("unchecked", "Not checked"))
    )
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SEC filing answers · Verification review</title><style>{_STYLE}</style></head><body>
<header class="hero"><div class="hero-inner"><p class="eyebrow">Internal Evidence review</p><h1>SEC Filing Answers</h1>
<p class="hero-copy">A transparent view of each answer, its consistency checks, and the filing evidence behind it.</p></div></header>
<main class="page"><section class="summary" aria-label="Result summary"><div class="stat total"><span>Total questions</span><strong>{len(records)}</strong></div>{stat_cards}</section>
<section class="toolbar" aria-label="Answer filters"><label class="search"><span class="visually-hidden">Search questions and answers</span><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-3.5-3.5"></path></svg><input id="answer-search" type="search" placeholder="Search questions or answers…" autocomplete="off"></label><div class="filters"><button type="button" class="filter active" data-filter="all">All <span>{len(records)}</span></button>{filter_buttons}</div></section>
<p id="no-results" hidden>No answers match the current search and filter.</p><section>{body}</section>
<p class="footer">Checks surface conflicts and uncertainty. They do not replace review of the cited filing.</p></main><script>{_SCRIPT}</script></body></html>
'''
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="JSONL written by record_verification")
    parser.add_argument("--output", type=Path, default=Path("logs/answers.html"))
    args = parser.parse_args(argv)
    records = []
    with args.results.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                render_answer(record)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                parser.error(f"Invalid results record on line {line_number}: {exc}")
            records.append(record)
    if not records:
        parser.error("The results file contains no answers.")
    print(write_answer_page(records, args.output))


if __name__ == "__main__":
    main()
