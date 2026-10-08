"""
Rebuild the dataset-blocked figures at IEEE spec — self-contained
=================================================================
fig7_scan_comparison, fig9_failure_analysis at 600 dpi PNG + vector SVG.
Answers Reviewer 4, Concern 3 ("blurred figures").

(fig4_age_nwbv intentionally SKIPPED — its content in this checkout plots
GT-nWBV-vs-age with a hardcoded rho=-0.597, which contradicts the corrected
manuscript (rho(pred,age)=+0.232, n.s.). Rebuilding it here would regenerate
the wrong statistic; it is handled separately.)

Run on the machine that has data/ds006557_data/ (raw .nii.gz AND participants.tsv).
Depends ONLY on scripts/generate_all_figures.py and scripts/generate_new_figures.py.

    python scripts/rebuild_blocked_figures.py

Output -> figures_rebuilt/ at the repo root.

The module-level shutil.rmtree(standalone_paper/paper_figures) in
generate_all_figures.py is patched out — running it unpatched destroys the real
fig7 MRI montage and substitutes a synthetic ellipses-and-noise placeholder.
"""

import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
OUT     = ROOT / "figures_rebuilt"
DPI     = 600

IEEE_RC = f'''
import matplotlib as _mpl
_mpl.rcParams.update({{
    "font.family":         "sans-serif",
    "font.sans-serif":     ["DejaVu Sans", "Helvetica", "Arial"],
    "svg.fonttype":        "none",
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "savefig.dpi":         {DPI},
    "savefig.bbox":        "tight",
    "savefig.facecolor":   "white",
    "savefig.transparent": False,
}})
'''

SAVEFIG_600 = f'''
def savefig(fig, name):
    fig.savefig(str(OUT_DIR / f"{{name}}.png"), format="png", dpi={DPI})
    fig.savefig(str(OUT_DIR / f"{{name}}.svg"), format="svg")
    plt.close(fig)
    print(f"    {{name}}.png ({DPI} dpi) + {{name}}.svg")
'''

SAVE_600 = f'''
def save(fig, name):
    fig.savefig(FIGURES_DIR / f"{{name}}.png", format="png", dpi={DPI},
                bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES_DIR / f"{{name}}.svg", format="svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    {{name}}.png ({DPI} dpi) + {{name}}.svg")
'''


def patch(src, patches, label):
    for old, new, reason in patches:
        if old not in src:
            sys.exit(f"\nPATCH FAILED in {label}: {reason}\n"
                     f"  could not find: {old[:100]!r}\n"
                     f"  The upstream script was edited. Fix the patch, do not skip it.")
        src = src.replace(old, new, 1)
    return src


def load(script_name, patches, label):
    src = patch((SCRIPTS / script_name).read_text(), patches, label)
    ns = {"__name__": "__rebuild__", "__file__": str(SCRIPTS / script_name)}
    exec(compile(src, f"<patched {script_name}>", "exec"), ns)
    return ns


def main():
    parts = ROOT / "data" / "ds006557_data" / "participants.tsv"
    if not parts.exists():
        sys.exit(f"STOP: {parts} not found.\n"
                 "  fig9 plots real subject data. Do not synthesise it.\n"
                 "  Run on the machine that holds data/ds006557_data/.")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Rebuilding blocked figures at {DPI} dpi -> {OUT}\n" + "=" * 70)

    out_lit = repr(str(OUT))

    # ── fig9 from generate_all_figures.py ────────────────────────────────────
    print("\n[fig9_failure_analysis]")
    ns = load(
        "generate_all_figures.py",
        [
            ('OUT_DIR  = ROOT / "standalone_paper" / "paper_figures"',
             f"OUT_DIR  = Path({out_lit})",
             "redirect OUT_DIR away from the published figure directory"),

            ("""if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)
OUT_DIR.mkdir(parents=True)
print(f"Cleared and recreated {OUT_DIR}")""",
             "OUT_DIR.mkdir(parents=True, exist_ok=True)",
             "REMOVE the destructive rmtree — it would delete the real fig7 montage"),

            ("BLUE   = '#1f77b4'", IEEE_RC + "\nBLUE   = '#1f77b4'",
             "IEEE rcParams: 600 dpi, DejaVu, text-as-text SVG, opaque white"),

            ("""def savefig(fig, name):
    path_png = OUT_DIR / f"{name}.png"
    path_pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(str(path_png), format='png')
    fig.savefig(str(path_pdf), format='pdf')
    plt.close(fig)
    print(f"  Saved: {name}.png / .pdf")""",
             SAVEFIG_600,
             "savefig -> 600 dpi PNG + vector SVG"),
        ],
        "generate_all_figures.py",
    )
    ns["fig9_failure_analysis"]()

    # ── fig7 from generate_new_figures.py ────────────────────────────────────
    # Only fig7. fig1 and fig6 in this script are already rebuilt at 600 dpi
    # elsewhere; re-running them would overwrite good versions.
    print("\n[fig7_scan_comparison]")
    ns7 = load(
        "generate_new_figures.py",
        [
            ('FIGURES_DIR = project_root / "experiments" / "figures"',
             f"FIGURES_DIR = Path({out_lit})",
             "redirect FIGURES_DIR"),

            ('C_BLUE   = "#1B4F72"', IEEE_RC + '\nC_BLUE   = "#1B4F72"',
             "IEEE rcParams (this script sets none of its own)"),

            ("""def save(fig, name):
    for ext in ['pdf','png']:
        fig.savefig(FIGURES_DIR/f"{name}.{ext}", dpi=200,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}.pdf  {name}.png")""",
             SAVE_600,
             "save -> 600 dpi PNG + SVG (was dpi=200, below IEEE's 300 dpi floor)"),

            ("    tcols  = ['#85C1E9','#F0B27A','#82E0AA']",
             "    tcols  = ['#1f77b4','#ff7f0e','#2ca02c']",
             "column titles -> tab10, matching the other figures"),
        ],
        "generate_new_figures.py",
    )
    ns7["fig7"]()

    # ── report ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    expected = ("fig7_scan_comparison", "fig9_failure_analysis")
    ok = True
    for name in expected:
        png, svg = OUT / f"{name}.png", OUT / f"{name}.svg"
        got = [e for e, p in (("png", png), ("svg", svg)) if p.exists()]
        size = ""
        if png.exists():
            try:
                from PIL import Image
                im = Image.open(png)
                size = f"  {im.size[0]}x{im.size[1]} {im.mode} dpi={im.info.get('dpi')}"
            except ImportError:
                size = "  (install Pillow to report pixel dimensions)"
        flag = "" if len(got) == 2 else "   <-- MISSING"
        if len(got) != 2:
            ok = False
        print(f"  {name:24} {'+'.join(got) or 'NOTHING':10}{size}{flag}")

    print("\nCheck before copying back:")
    print("  - SVG opens in a browser with SELECTABLE text (svg.fonttype='none')")
    print("  - no tofu boxes where  +/-  x  mu  ~  <=  ->  should be")
    print("  - fig7 is the REAL montage (3 subjects x 3 modalities, SNR annotations)")
    print("  - git status is clean outside figures_rebuilt/")
    print("  - fig4 was SKIPPED (content bug: hardcoded rho=-0.597 contradicts manuscript)")
    if not ok:
        sys.exit("\nIncomplete — see MISSING above.")


if __name__ == "__main__":
    main()
