"""
Regenerate every table, figure and number in the paper from committed result files.

    python -m paper_build            # build into paper_build/out/
    python -m paper_build --strict   # also fail if any input file is not tracked by git

Needs numpy, scipy, matplotlib only. No MRI data, checkpoints or GPU.
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from . import checks, figures
from .compute import compute_all
from .sources import ROOT

OUT = ROOT / "paper_build" / "out"


def _md_table(t):
    lines = [f"**{t['title']}**", "", "| " + " | ".join(t["columns"]) + " |", "|" + "---|" * len(t["columns"])]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in t["rows"]]
    if t.get("note"):
        lines += ["", f"_{t['note']}_"]
    return "\n".join(lines)


def _csv(t):
    import csv
    import io
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(t["columns"])
    w.writerows(t["rows"])
    return s.getvalue()


def _json_safe(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, float) and o != o:
        return None
    raise TypeError(type(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    problems = checks.lint()
    if problems:
        print("LINT FAILED:\n  " + "\n  ".join(problems))
        sys.exit(1)

    numbers, tables, figdata = compute_all()
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        (OUT / "tables" / f"{name}.md").write_text(_md_table(t) + "\n")
        (OUT / "tables" / f"{name}.csv").write_text(_csv(t))
    figs = figures.build_all(numbers, tables, figdata, OUT / "figures")
    xc = checks.crosscheck(numbers)
    inputs = checks.committed()
    state = checks.git_state()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1, default=_json_safe))

    failed = [c for c in xc if not c["ok"]]
    untracked = [f for f, ok in inputs.items() if not ok]
    rep = ["# Paper build report", "",
           f"- Built: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           f"- Git commit: `{state['commit']}`" + (f" (tag `{state['tag']}`)" if state["tag"] else ""),
           f"- Uncommitted changes in paper_build/experiments/scripts: {state['uncommitted_changes']}",
           f"- Lint: passed (no RNG outside the seeded bootstrap, no typed-in results)",
           f"- Cross-checks: {len(xc) - len(failed)}/{len(xc)} passed",
           f"- Inputs: {len(inputs)} files, {len(untracked)} not tracked by git",
           f"- Outputs: {len(tables)} tables, {len(figs)} figures, {len(numbers)} numbers", "",
           "## Cross-checks (paper_build recomputation vs value saved by the experiment script)", "",
           "| Check | paper_build | experiment file | OK |", "|---|---|---|---|"]
    rep += [f"| {c['check']} | {c['paper_build']} | {c['experiment_file']} | {'yes' if c['ok'] else '**NO**'} |" for c in xc]
    rep += ["", "## Input files", ""] + [f"- `{f}`" + ("" if ok else "  **(not tracked by git)**") for f, ok in inputs.items()]
    rep += ["", "## Tables", ""] + [f"- `tables/{n}.md` — {t['title']}" for n, t in tables.items()]
    rep += ["", "## Figures", ""] + [f"- `figures/{n}.png` / `.svg`" for n in figs]
    (OUT / "BUILD_REPORT.md").write_text("\n".join(rep) + "\n")

    print(f"tables {len(tables)}, figures {len(figs)}, numbers {len(numbers)}; cross-checks {len(xc) - len(failed)}/{len(xc)}")
    for c in failed:
        print("  CROSS-CHECK FAILED:", c)
    if untracked:
        print(f"  {len(untracked)} input files not tracked by git")
    if failed or (args.strict and (untracked or state["uncommitted_changes"])):
        sys.exit(1)


if __name__ == "__main__":
    main()
