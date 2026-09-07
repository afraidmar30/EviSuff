#!/usr/bin/env python3
"""Render a local, system-blind annotation UI from one annotator's CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FORBIDDEN_CLAIM_FIELDS = {"system", "group", "gold_answerability", "raw_final_answer"}
FORBIDDEN_OUTPUT_FIELDS = {"system", "raw_final_answer"}
CLAIM_LABELS = ["", "supported", "partially_supported", "unsupported", "unclear"]
OUTPUT_LABELS = {
    "source_policy_compliant": ["", "yes", "no"],
    "conflict_disclosure_correct": ["", "yes", "no", "not_applicable"],
    "uncertainty_present": ["", "yes", "no"],
    "uncertainty_appropriate": ["", "yes", "no"],
    "stop_decision": ["", "appropriate", "premature", "unnecessary_search"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--instructions", required=True)
    parser.add_argument("--annotator", required=True, choices=("a", "b"))
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_sheet(path: Path, key: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    ids = [str(row.get(key) or "") for row in rows]
    if not ids or any(not item_id for item_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path}: missing or duplicate {key}")
    return {"fields": fields, "rows": rows, "key": key, "filename": path.name}


def safe_json(payload: Any) -> str:
    return (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def build_html(
    claims: dict[str, Any], outputs: dict[str, Any], instructions: str, annotator: str
) -> str:
    claim_forbidden = sorted(FORBIDDEN_CLAIM_FIELDS & set(claims["fields"]))
    output_forbidden = sorted(FORBIDDEN_OUTPUT_FIELDS & set(outputs["fields"]))
    if claim_forbidden or output_forbidden:
        raise ValueError(
            f"System-blinding violation: claim={claim_forbidden}, output={output_forbidden}"
        )
    # Key browser state to the full blinded content, not only opaque IDs.  If
    # extraction or context changes before annotation starts, old local labels
    # must not be silently attached to a different claim under the same ID.
    digest_material = json.dumps(
        {"claims": claims, "outputs": outputs},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(digest_material.encode("utf-8")).hexdigest()[:16]
    bundle = {
        "annotator": annotator,
        "digest": digest,
        "claims": claims,
        "outputs": outputs,
        "claim_labels": CLAIM_LABELS,
        "output_labels": OUTPUT_LABELS,
    }
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EviSuff blinded human audit — annotator __ANNOTATOR__</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#f5f6f8;color:#1e2630}header{position:sticky;top:0;background:#17202b;color:white;padding:10px 18px;z-index:2}header button{margin-right:8px}.wrap{max-width:1180px;margin:18px auto;padding:0 16px}.card{background:white;border:1px solid #d9dee5;border-radius:10px;padding:18px;box-shadow:0 1px 3px #0001}.field{margin:12px 0}.field h3{font-size:13px;margin:0 0 5px;color:#56606d;text-transform:uppercase}.field pre{white-space:pre-wrap;word-break:break-word;background:#f8fafc;border:1px solid #e3e7ec;padding:10px;border-radius:6px;max-height:330px;overflow:auto}.labels{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin-top:18px}.labels label{font-weight:600}.labels select,.labels textarea{width:100%;box-sizing:border-box;margin-top:5px;padding:8px}.labels textarea{min-height:90px}.nav{display:flex;align-items:center;gap:10px;margin:12px 0}.nav input{width:80px}.progress{font-weight:700}.hidden{display:none}details{margin:10px 0}a{color:#075db7}.warn{background:#fff4cc;padding:10px;border-left:4px solid #d99b00}.done{color:#1d7b3a}.todo{color:#a43b2d}button{padding:7px 11px;cursor:pointer}
</style></head><body>
<header><strong>EviSuff blinded audit — annotator __ANNOTATOR__</strong> &nbsp;
<button onclick="switchKind('claims')">Claim pairs</button><button onclick="switchKind('outputs')">Outputs</button>
<button onclick="showInstructions()">Instructions</button><button onclick="exportCurrent()">Export current CSV</button></header>
<main class="wrap"><div id="instructions" class="card hidden"><h2>Instructions</h2><pre style="white-space:pre-wrap">__INSTRUCTIONS__</pre></div>
<div id="workspace"><div class="nav"><button onclick="move(-1)">Previous</button><button onclick="move(1)">Next</button><label>Item <input id="jump" type="number" min="1" onchange="jumpTo(this.value)"></label><span id="position"></span><span id="progress" class="progress"></span></div><div id="card" class="card"></div></div></main>
<script id="bundle" type="application/json">__BUNDLE__</script>
<script>
const B=JSON.parse(document.getElementById('bundle').textContent);let kind='claims',index=0;
const storageKey='evisuff-audit-'+B.annotator+'-'+B.digest;let state={claims:{},outputs:{}};
try{state=JSON.parse(localStorage.getItem(storageKey))||state}catch(e){}
function save(){localStorage.setItem(storageKey,JSON.stringify(state))}
function esc(v){return String(v??'')}
function addField(root,title,value,asLink=false){let d=document.createElement('div');d.className='field';let h=document.createElement('h3');h.textContent=title;d.appendChild(h);if(asLink&&/^https?:\/\//i.test(value)){let a=document.createElement('a');a.href=value;a.target='_blank';a.rel='noopener';a.textContent=value;d.appendChild(a)}else{let p=document.createElement('pre');p.textContent=esc(value);d.appendChild(p)}root.appendChild(d)}
function currentSheet(){return B[kind]}function currentId(){let s=currentSheet();return s.rows[index][s.key]}
function requiredFields(){return kind==='claims'?['support_label']:Object.keys(B.output_labels)}
function render(){document.getElementById('instructions').classList.add('hidden');document.getElementById('workspace').classList.remove('hidden');let s=currentSheet(),r=s.rows[index],id=currentId(),card=document.getElementById('card');card.innerHTML='';let title=document.createElement('h2');title.textContent=(kind==='claims'?'Claim pair':'Output')+' '+(index+1);card.appendChild(title);let warning=document.createElement('div');warning.className='warn';warning.textContent='Model identity is hidden. Judge only the supplied answer, evidence, gold boundary, and trace.';card.appendChild(warning);
if(kind==='claims'){['question','claim','citation_title','citation_snippet','source_type'].forEach(f=>addField(card,f,r[f]));addField(card,'citation_url',r.citation_url,true)}else{['group','question','final_answer','gold_final_answer','uncertainty_requirement','boundary_conditions','required_facets','evidence_points','source_policy','minimum_evidence_policy','gold_source_inventory','known_conflicts','stop_condition','search_calls','fetch_calls','total_tool_calls','retrieved_source_count','final_citation_count','citation_inventory','action_trace','gate_trace'].forEach(f=>addField(card,f,r[f]))}
let area=document.createElement('div');area.className='labels';let item=state[kind][id]||{};requiredFields().forEach(f=>{let lab=document.createElement('label');lab.textContent=f;let sel=document.createElement('select');let opts=kind==='claims'?B.claim_labels:B.output_labels[f];if(kind==='outputs'&&f==='conflict_disclosure_correct'){opts=r.group==='conflict'?['','yes','no']:['','not_applicable']}opts.forEach(v=>{let o=document.createElement('option');o.value=v;o.textContent=v||'— choose —';sel.appendChild(o)});sel.value=item[f]||r[f]||'';sel.onchange=()=>{item[f]=sel.value;state[kind][id]=item;save();renderProgress()};lab.appendChild(sel);area.appendChild(lab)});let lab=document.createElement('label');lab.textContent='rationale';let ta=document.createElement('textarea');ta.value=item.rationale||r.rationale||'';ta.oninput=()=>{item.rationale=ta.value;state[kind][id]=item;save()};lab.appendChild(ta);area.appendChild(lab);card.appendChild(area);document.getElementById('jump').value=index+1;renderProgress()}
function complete(row){let item=state[kind][row[currentSheet().key]]||{};return requiredFields().every(f=>item[f]||row[f])}
function renderProgress(){let rows=currentSheet().rows,n=rows.filter(complete).length;document.getElementById('position').textContent=(index+1)+' / '+rows.length;let p=document.getElementById('progress');p.textContent=n+' / '+rows.length+' complete';p.className='progress '+(n===rows.length?'done':'todo')}
function move(d){index=Math.max(0,Math.min(currentSheet().rows.length-1,index+d));render()}function jumpTo(v){index=Math.max(0,Math.min(currentSheet().rows.length-1,Number(v)-1||0));render()}function switchKind(k){kind=k;index=0;render()}
function showInstructions(){document.getElementById('workspace').classList.add('hidden');document.getElementById('instructions').classList.remove('hidden')}
function csvCell(v){let s=String(v??'');return '"'+s.replaceAll('"','""')+'"'}
function exportCurrent(){let s=currentSheet(),rows=s.rows.map(r=>{let item=state[kind][r[s.key]]||{};return s.fields.map(f=>csvCell(Object.hasOwn(item,f)?item[f]:r[f])).join(',')});let csv=s.fields.map(csvCell).join(',')+'\r\n'+rows.join('\r\n')+'\r\n';let blob=new Blob([csv],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=s.filename;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
render();
</script></body></html>'''
    return (
        template.replace("__ANNOTATOR__", annotator.upper())
        .replace("__INSTRUCTIONS__", instructions.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        .replace("__BUNDLE__", safe_json(bundle))
    )


def main() -> None:
    args = parse_args()
    claims = load_sheet(Path(args.claims), "blind_pair_id")
    outputs = load_sheet(Path(args.outputs), "blind_output_id")
    instructions = Path(args.instructions).read_text(encoding="utf-8")
    html = build_html(claims, outputs, instructions, args.annotator)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
