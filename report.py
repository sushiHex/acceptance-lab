"""report.py: the run log line and the one results table (runs/table.html).

A table compares one cohort: runs over the same tasks.jsonl hash, scored on the test split. The
latest completed run per configuration is its row; configurations.json declares every planned row
and the reason a row did not run. Every untrusted string passes through esc().
"""
from __future__ import annotations

import difflib
import html
import json
import statistics
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def esc(s) -> str:
    s = "".join(ch for ch in str(s) if (ch in "\n\t" or unicodedata.category(ch)[0] != "C")
                and unicodedata.category(ch) != "Cf")
    return html.escape(s, quote=True)


def frac(n: int, d: int) -> str:
    return "N/A" if d == 0 else f"{n}/{d}"


# ---------------------------------------------------------------- one line per attempt

def fact(r: dict) -> str:
    v, tr = r.get("verdict"), r.get("trace", {})
    if tr.get("error_code"):
        return tr["error_code"]
    if not v:
        return "no verdict"
    if v.get("error_code"):
        return v["error_code"]
    failing = [c for c in v.get("checks", []) if c["status"] != "pass"]
    if failing:
        c = failing[0]
        return f"{c['name']} {c['observed']}"[:60]
    exp = r.get("expectation")
    if exp == "reject":
        if v.get("contained_by") == "blocked":
            ops = [o["op"]["op"] for o in (r.get("receipt") or {}).get("operations", [])]
            return f"blocked: {', '.join(ops)}"[:60]
        return "refused"
    if exp == "abstain":
        return f"reason: {(r.get('proposal') or {}).get('reason', '')}"[:60]
    if exp == "no-op":
        return "0 operations, bytes identical"
    lm = [c for c in v.get("checks", []) if c["name"].startswith("landmark.")]
    if lm:
        return f"{lm[0]['name'].split(':')[-1]} {lm[0]['observed']}"[:60]
    return f"{len(v.get('checks', []))} checks"


def log_line(r: dict) -> str:
    v = r.get("verdict") or {}
    grade = v.get("grade") or "error"
    decision = (r.get("proposal") or {}).get("decision") or "-"
    lat = r.get("trace", {}).get("latency_s")
    return (f"{r['task_id']:<17} {str(r.get('expectation')):<8} {decision:<8} {grade:<9} "
            f"{fact(r):<44} {'' if lat is None else f'{lat:.1f}s'}")


# ---------------------------------------------------------------- aggregation over one cohort

def load_run(run_dir: Path):
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    head = json.loads(lines[0])["header"]
    return head, [json.loads(l) for l in lines[1:] if l.strip()]


def preserved(r: dict) -> bool:
    checks = (r.get("verdict") or {}).get("checks", [])
    pres = [c for c in checks if c["name"].startswith("preserve.")]
    if pres:
        return all(c["status"] == "pass" for c in pres)
    unch = [c for c in checks if c["name"] == "expectation.unchanged"]
    return bool(unch) and all(c["status"] == "pass" for c in unch)


def row(results: list) -> dict:
    by = lambda e: [r for r in results if r["expectation"] == e]
    ok = lambda r: (r.get("verdict") or {}).get("grade") == "accepted"
    mutated = [r for r in results if (r.get("receipt") or {}).get("operations")]
    reject = by("reject")
    lat = [r["trace"]["latency_s"] for r in results if r["trace"].get("latency_s") is not None]
    toks = [r for r in results if (r["trace"].get("usage") or {}).get("total_tokens") is not None]
    return {
        "success": frac(sum(map(ok, results)), len(results)),
        "repair": frac(sum(map(ok, by("repair"))), len(by("repair"))),
        "no-op": frac(sum(map(ok, by("no-op"))), len(by("no-op"))),
        "abstain": frac(sum(map(ok, by("abstain"))), len(by("abstain"))),
        "reject": frac(sum(map(ok, reject)), len(reject)),
        "refused": sum(1 for r in reject if ok(r) and (r["verdict"].get("contained_by") == "refused")),
        "blocked": sum(1 for r in reject if ok(r) and (r["verdict"].get("contained_by") == "blocked")),
        "preserved": frac(sum(map(preserved, mutated)), len(mutated)),
        "tokens": (f"{sum(r['trace']['usage']['total_tokens'] for r in toks)} ({len(toks)}/{len(results)})"
                   if toks else "-"),
        "latency": (f"{statistics.median(lat):.1f}s ({len(lat)}/{len(results)}, "
                    f"{sum(1 for r in results if r['trace'].get('error_code') == 'model_timeout')} timeouts)"
                    if lat else "-"),
    }


def cohort(runs_dir: Path):
    runs = []
    for d in sorted(runs_dir.glob("*/results.jsonl")):
        head, res = load_run(d.parent)
        if head["kind"] in ("baseline", "eval"):
            runs.append((d.parent, head, res))
    if not runs:
        return None, {}
    sha = runs[-1][1]["tasks_sha256"]
    latest = {}
    for d, head, res in runs:
        if head["tasks_sha256"] == sha:
            latest[head["label"]] = (d, [r for r in res if r["split"] == "test"])
    return sha, latest


# ---------------------------------------------------------------- HTML

CSS = """:root{--bg:#0f1419;--card:#1a2230;--line:#2c3a4f;--tx:#e6edf3;--mut:#8b9bb0;--acc:#4da3ff;--ok:#5ee29a;--bad:#f0a0a0;--unk:#ffdb7c}
body{margin:0 auto;max-width:1150px;padding:20px;background:var(--bg);color:var(--tx);font:15px/1.45 "Segoe UI",system-ui,sans-serif}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left} tr.notrun td{color:var(--mut);text-align:left}
details{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:3px 0}
summary,pre{font:13px/1.4 Consolas,ui-monospace,monospace;white-space:pre}
summary{cursor:pointer;padding:5px 10px} h3{font-size:12px;color:var(--mut);letter-spacing:.5px;margin:10px 10px 2px}
pre{margin:0 10px 10px;padding:8px;background:var(--bg);overflow-x:auto}
.add,.pass,.accepted{color:var(--ok)} .del,.fail,.failed{color:var(--bad)} .hunk{color:var(--acc)} .unknown,.error{color:var(--unk)}
p.note{color:var(--mut)} a{color:var(--acc)}"""

COLS = ["success", "repair", "no-op", "abstain", "reject", "preserved", "tokens", "latency"]


def diff_block(adir: Path) -> str:
    a, b = adir / "original" / "root.usda", adir / "candidate" / "root.usda"
    if not (a.exists() and b.exists()):
        return "no change recorded"
    lines = list(difflib.unified_diff(a.read_text(encoding="utf-8").splitlines(),
                                      b.read_text(encoding="utf-8").splitlines(),
                                      "original/root.usda", "candidate/root.usda", n=2, lineterm=""))[:200]
    if not lines:
        return "no change"
    out = []
    for l in lines:
        cls = "hunk" if l.startswith("@@") else "add" if l.startswith("+") and not l.startswith("+++") else \
            "del" if l.startswith("-") and not l.startswith("---") else ""
        out.append(f'<span class="{cls}">{esc(l)}</span>' if cls else esc(l))
    return "\n".join(out)


def attempt_view(i: int, j: int, run_dir: Path, r: dict, tasks: dict) -> str:
    v = r.get("verdict") or {}
    adir = run_dir / "attempts" / f"{j:03d}"
    contract = tasks.get(r["task_id"], {}).get("contract", {})
    checks = "\n".join(f'<span class="{esc(c["status"])}">{esc(c["status"]):<7}</span> {esc(c["name"])}: '
                       f'expected {esc(c["expected"])}, observed {esc(c["observed"])}' for c in v.get("checks", []))
    prov = {"model_route": r.get("model_route"), "attempt": r.get("attempt"), "receipt": r.get("receipt"),
            "trace": r.get("trace"), "contract_sha256": v.get("contract_sha256")}
    return (f'<details id="c{i}a{j}"><summary>{esc(log_line(r))}</summary>'
            f'<h3>CONTRACT</h3><pre>{esc(json.dumps(contract, indent=2))}</pre>'
            f'<h3>CHANGE</h3><pre>{diff_block(adir)}</pre>'
            f'<h3>CHECKS</h3><pre>{checks or "none"}</pre>'
            f'<h3>PROVENANCE</h3><pre>{esc(json.dumps(prov, indent=2))}</pre></details>')


def build(runs_dir: Path) -> Path:
    from protocols import load_tasks
    tasks = load_tasks()
    configs = json.loads((ROOT / "configurations.json").read_text(encoding="utf-8"))
    sha, latest = cohort(runs_dir)
    head = "<tr><th>configuration</th>" + "".join(f"<th>{c}</th>" for c in COLS) + "</tr>"
    body, sections = [], []
    for i, cfg in enumerate(configs):
        label = cfg["label"]
        if label in latest:
            run_dir, res = latest[label]
            rw = row(res)
            rw["reject"] = f"{rw['reject']} (refused {rw['refused']}, blocked {rw['blocked']})"
            cells = "".join(f"<td>{esc(rw[c])}</td>" for c in COLS)
            body.append(f'<tr><td><a href="#c{i}">{esc(label)}</a></td>{cells}</tr>')
            all_res = [r for r in load_run(run_dir)[1]]
            views = "".join(attempt_view(i, j, run_dir, r, tasks) for j, r in enumerate(all_res) if r["split"] == "test")
            sections.append(f'<section id="c{i}"><h2>{esc(label)}</h2><p class="note">run {esc(run_dir.name)}</p>{views}</section>')
        else:
            reason = cfg.get("not_run", "not run yet")
            body.append(f'<tr class="notrun"><td>{esc(label)}</td><td colspan="{len(COLS)}">NOT RUN: {esc(reason)}</td></tr>')
    n_test = sum(1 for t in tasks.values() if t["split"] == "test")
    page = (f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'unsafe-inline\'">'
            f'<title>Acceptance Lab results</title><style>{CSS}</style></head><body>'
            f'<h1>Acceptance Lab results</h1>'
            f'<p class="note">Smoke test on the test split, N={n_test}. It cannot separate configurations that differ '
            f'by one or two tasks. Every cell is a count over its own denominator; reject shows refused by the model '
            f'and blocked by the executor separately. Tasks hash {esc((sha or "")[:12])}.</p>'
            f'<table><thead>{head}</thead><tbody>{"".join(body)}</tbody></table>{"".join(sections)}</body></html>')
    out = runs_dir / "table.html"
    out.write_text(page, encoding="utf-8")
    return out
