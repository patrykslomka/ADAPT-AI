"""Generate the journal-paper result figures from committed matrix data.

Single source of truth: every number comes from `evaluation/matrix_stats.py`
(which in turn reuses the Wilcoxon / Holm / rank-biserial implementation in
`scripts/analyze_results.py`). The Table V emitter reads the same module, so
the figures and the table can never disagree.

Outputs (in data/evaluation/figures/):
  fig_safety_floor.{pdf,png[,pgf]}         - capability-floor figure
  fig_safety_vs_reasoning.{pdf,png[,pgf]}  - safety vs reasoning contrast
  fig_research_design.{pdf,png[,pgf]}      - pipeline / research-design diagram
  figures.tex                              - \\includegraphics snippets + captions

Vector PDF is always produced (the standard \\includegraphics path). PGF is
emitted only when a TeX engine (xelatex/lualatex/pdflatex) is on PATH, since
matplotlib's pgf backend requires one even to write the macros.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation import matrix_stats as ms  # noqa: E402

FIG_DIR = ms.DATA_DIR / "figures"

# Colour-blind-safe palette (Wong) per domain.
DOMAIN_COLORS = {
    "healthcare": "#0072B2",  # blue
    "legal": "#E69F00",       # orange
    "finance": "#009E73",     # green
}
# Capability tiers, weakest -> strongest (the x-axis of the floor figure).
TIER_ORDER = ["qwen7b", "haiku", "sonnet"]
# Short tick labels for crowded multi-panel axes.
SHORT_LABELS = {"qwen7b": "Qwen", "haiku": "Haiku", "sonnet": "Sonnet"}


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.dpi": 150,
    })


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _tex_engine() -> str | None:
    for eng in ("xelatex", "lualatex", "pdflatex"):
        if shutil.which(eng):
            return eng
    return None


def _save(fig, stem: str, formats: list[str]) -> list[str]:
    written = []
    for ext in formats:
        path = FIG_DIR / f"{stem}.{ext}"
        try:
            fig.savefig(path, bbox_inches="tight")
            written.append(path.name)
        except Exception as e:  # pragma: no cover - pgf needs a tex engine
            print(f"  [skip] {stem}.{ext}: {type(e).__name__}: {str(e)[:80]}")
    plt.close(fig)
    return written


#  figure 1: capability floor

def fig_safety_floor(cells: dict, formats: list[str]) -> list[str]:
    """Δ(overall_score) on safety-critical questions across capability tiers.

    The benefit of modularity on safety-critical queries is what we care about;
    raw safety_score is at ceiling for both arms (both refuse / disclaim), so we
    plot answer *reliability* on the compliance_safety + hallucination_trap
    questions, where the modular pipeline's concept recall advantage lands.
    """
    tiers = [m for m in TIER_ORDER if any((m, d) in cells for d in ms.DOMAINS)]
    domains = ms.DOMAINS
    n_d = len(domains)
    width = 0.8 / n_d

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(tiers))

    for j, domain in enumerate(domains):
        deltas, lows, highs, stars = [], [], [], []
        for m in tiers:
            cell = cells.get((m, domain))
            if cell is None:
                deltas.append(0.0); lows.append(0.0); highs.append(0.0); stars.append("")
                continue
            d = ms.category_group_delta(cell, ms.SAFETY_CATEGORIES, "overall_score")
            deltas.append(d["delta"])
            lo, hi = d["ci95"]
            lows.append(d["delta"] - lo)
            highs.append(hi - d["delta"])
            stars.append(_stars(d["p"]))
        offs = [xi + (j - (n_d - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, deltas, width=width, color=DOMAIN_COLORS[domain],
                      label=ms.DOMAIN_LABELS[domain], edgecolor="white", linewidth=0.5)
        ax.errorbar(offs, deltas, yerr=[lows, highs], fmt="none",
                    ecolor="#333333", elinewidth=0.8, capsize=2.5)
        for rect, s, dv in zip(bars, stars, deltas):
            if s:
                yy = rect.get_height() + (0.006 if dv >= 0 else -0.018)
                ax.annotate(s, (rect.get_x() + rect.get_width() / 2, yy),
                            ha="center", va="bottom" if dv >= 0 else "top", fontsize=9)

    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ms.MODEL_LABELS[m] for m in tiers])
    ax.set_xlabel("Base model (capability tier: weaker $\\rightarrow$ stronger)")
    ax.set_ylabel(r"$\Delta$ score on safety-critical Q" "\n(ADAPT-AI $-$ baseline)")
    ax.set_title("Safety-critical reliability gain across model tiers")
    ax.legend(title="Domain", frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18))
    ax.margins(x=0.05)
    return _save(fig, "fig_safety_floor", formats)


#  figure 2: safety vs reasoning

def fig_safety_vs_reasoning(cells: dict, formats: list[str]) -> list[str]:
    """Per domain: Δ on safety-critical vs reasoning questions, across tiers."""
    tiers = [m for m in TIER_ORDER if any((m, d) in cells for d in ms.DOMAINS)]
    fig, axes = plt.subplots(1, len(ms.DOMAINS), figsize=(7.4, 3.0), sharey=True)
    width = 0.38

    for ax, domain in zip(axes, ms.DOMAINS):
        x = range(len(tiers))
        saf, rea, saf_s, rea_s = [], [], [], []
        for m in tiers:
            cell = cells.get((m, domain))
            if cell is None:
                saf.append(0); rea.append(0); saf_s.append(""); rea_s.append(""); continue
            svr = ms.safety_vs_reasoning(cell)
            saf.append(svr["safety"]["delta"]); saf_s.append(_stars(svr["safety"]["p"]))
            rea.append(svr["reasoning"]["delta"]); rea_s.append(_stars(svr["reasoning"]["p"]))
        b1 = ax.bar([xi - width / 2 for xi in x], saf, width, color="#D55E00",
                    label="Safety-critical", edgecolor="white", linewidth=0.4)
        b2 = ax.bar([xi + width / 2 for xi in x], rea, width, color="#56B4E9",
                    label="Reasoning", edgecolor="white", linewidth=0.4)
        for bars, ss in ((b1, saf_s), (b2, rea_s)):
            for rect, s in zip(bars, ss):
                if s:
                    h = rect.get_height()
                    ax.annotate(s, (rect.get_x() + rect.get_width() / 2,
                                    h + (0.004 if h >= 0 else -0.012)),
                                ha="center", va="bottom" if h >= 0 else "top", fontsize=8)
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels([SHORT_LABELS[m] for m in tiers], rotation=0)
        ax.set_title(ms.DOMAIN_LABELS[domain])
        ax.margins(x=0.08)

    axes[0].set_ylabel(r"$\Delta$ overall score" "\n(ADAPT-AI $-$ baseline)")
    axes[1].set_xlabel("Base model")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.suptitle("Where the modular pipeline helps: safety-critical vs reasoning questions",
                 fontsize=10, y=1.02)
    return _save(fig, "fig_safety_vs_reasoning", formats)


#  figure 3: research-design diagram ─

def fig_research_design(formats: list[str]) -> list[str]:
    """Static pipeline / research-design overview drawn for consistent styling."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(cx, cy, w, h, text, fc="#EAF2F8", ec="#2471A3", fs=8.5):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=1.4",
                                    fc=fc, ec=ec, linewidth=1.1))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-|>", color="#34495E", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, color=color, lw=1.1,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    box(11, 44, 18, 9, "Query\n(+ domain)", fc="#FCF3CF", ec="#B7950B")
    box(34, 44, 20, 11, "intent_and_retrieve\n(RAT / RAG router)")
    box(57, 44, 17, 9, "primary_agent")
    box(80, 51, 19, 9, "compliance_agent\n(rule-based)", fc="#FDEDEC", ec="#CB4335")
    box(80, 37, 19, 9, "quality_agent\n(hallucination)", fc="#FDEDEC", ec="#CB4335")
    box(57, 16, 19, 9, "review_results")
    box(30, 16, 20, 10, "aggregate_response\n(+ disclaimer)", fc="#E8F8F5", ec="#117A65")
    box(11, 16, 14, 9, "Response", fc="#FCF3CF", ec="#B7950B")

    arrow(20, 44, 24, 44)
    arrow(44, 44, 48.5, 44)
    arrow(65.5, 45.5, 70.5, 50)        # primary -> compliance (fan-out)
    arrow(65.5, 42.5, 70.5, 38)        # primary -> quality   (fan-out)
    arrow(70.5, 49, 66.5, 18, color="#34495E")   # compliance -> review (fan-in)
    arrow(70.5, 35, 66.5, 18, color="#34495E")   # quality -> review    (fan-in)
    arrow(47.5, 16, 40, 16)
    arrow(20, 16, 18, 16)
    # retry loop (quality fail, revision_count < 1)
    arrow(50, 20, 54, 39.5, style="-|>", color="#CB4335", ls=(0, (4, 2)))
    ax.text(56.5, 29, "retry $\\leq 1$\n(quality fail)", color="#CB4335",
            fontsize=7.5, ha="left", style="italic")
    # MCP hub annotation
    box(30, 28, 22, 7, "MCP client hub\n(rag / rat / validate tools)",
        fc="#F4ECF7", ec="#7D3C98", fs=7.5)
    arrow(30, 38.5, 30, 31.5, style="<|-|>", color="#7D3C98", ls=(0, (2, 2)))

    ax.text(50, 55, "ADAPT-AI pipeline (LangGraph StateGraph over in-process FastMCP)",
            ha="center", va="center", fontsize=9.5, weight="bold")
    ax.text(80, 30, "parallel fan-out / fan-in", ha="center", fontsize=7,
            color="#CB4335", style="italic")
    return _save(fig, "fig_research_design", formats)


#  figure 4: baseline ladder (optional - needs run_ladder.py --set ladder) ─

def fig_baseline_ladder(formats: list[str]) -> list[str]:
    """Mean overall score per ladder rung (b0..b3) + full, grouped by domain."""
    data = {d: ms.ladder_means(d) for d in ms.DOMAINS}
    data = {d: v for d, v in data.items() if v}
    if not data:
        print("  [skip] fig_baseline_ladder: no data in data/evaluation/ladder/ "
              "(run scripts/run_ladder.py --set ladder)")
        return []

    rungs = [r for r in (*ms.LADDER_VARIANTS, "full")
             if any(r in v for v in data.values())]
    domains = [d for d in ms.DOMAINS if d in data]
    n_d = len(domains)
    width = 0.8 / n_d
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = range(len(rungs))
    for j, domain in enumerate(domains):
        ys = [data[domain].get(r, 0.0) for r in rungs]
        offs = [xi + (j - (n_d - 1) / 2) * width for xi in x]
        ax.bar(offs, ys, width=width, color=DOMAIN_COLORS[domain],
               label=ms.DOMAIN_LABELS[domain], edgecolor="white", linewidth=0.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels([ms.LADDER_LABELS.get(r, r) for r in rungs])
    ax.set_ylabel("Mean overall score")
    ax.set_xlabel("Baseline ladder rung $\\rightarrow$ full pipeline")
    ax.set_title("Component attribution: baseline ladder vs. full pipeline")
    ax.legend(title="Domain", frameon=False, ncol=n_d, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    return _save(fig, "fig_baseline_ladder", formats)


#  figure 5: ablation (optional - needs run_ladder.py --set ablation) ─

def fig_ablation(formats: list[str]) -> list[str]:
    """ADAPT-AI overall score: full vs each component ablation, by domain."""
    data = {d: ms.ablation_means(d) for d in ms.DOMAINS}
    data = {d: v for d, v in data.items() if v}
    if not data:
        print("  [skip] fig_ablation: no data in data/evaluation/ablation/ "
              "(run scripts/run_ladder.py --set ablation)")
        return []

    tags = [t for t in ms.ABLATION_TAGS if any(t in v for v in data.values())]
    domains = [d for d in ms.DOMAINS if d in data]
    n_d = len(domains)
    width = 0.8 / n_d
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(tags))
    for j, domain in enumerate(domains):
        full = data[domain].get("full")
        ys = [data[domain].get(t, 0.0) for t in tags]
        offs = [xi + (j - (n_d - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, ys, width=width, color=DOMAIN_COLORS[domain],
                      label=ms.DOMAIN_LABELS[domain], edgecolor="white", linewidth=0.5)
        # annotate the drop vs full on each ablation bar
        if full:
            for rect, t in zip(bars, tags):
                if t != "full":
                    drop = data[domain].get(t, 0.0) - full
                    if abs(drop) >= 0.003:
                        ax.annotate(f"{drop:+.2f}",
                                    (rect.get_x() + rect.get_width() / 2,
                                     rect.get_height()),
                                    ha="center", va="bottom", fontsize=6.5,
                                    color="#555555")
    ax.set_xticks(list(x))
    ax.set_xticklabels([ms.ABLATION_LABELS.get(t, t) for t in tags])
    ax.set_ylabel("Mean overall score (ADAPT-AI)")
    ax.set_xlabel("Component ablation")
    ax.set_title("Component ablation: dropping each module from the full pipeline")
    ax.legend(title="Domain", frameon=False, ncol=n_d, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    return _save(fig, "fig_ablation", formats)


#  tex snippets ─

# One block per figure; assembled into figures.tex for whatever was produced.
TEX_BLOCKS: dict[str, str] = {
    "fig_safety_floor": r"""\begin{figure}[t]
  \centering
  \includegraphics[width=0.86\columnwidth]{figures/fig_safety_floor.pdf}
  \caption{Safety-critical reliability gain ($\Delta$ overall score, ADAPT-AI
  minus the fair \texttt{b1\_disclaimer} baseline) on the compliance-safety and
  hallucination-trap questions, across three base-model capability tiers. Error
  bars are 95\% CIs of the paired mean difference; stars mark uncorrected
  Wilcoxon significance ($^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$). The benefit
  is unstable on the weakest model and largest in the legal/finance domains.}
  \label{fig:safety-floor}
\end{figure}""",
    "fig_safety_vs_reasoning": r"""\begin{figure*}[t]
  \centering
  \includegraphics[width=0.92\textwidth]{figures/fig_safety_vs_reasoning.pdf}
  \caption{Where the modular pipeline helps: $\Delta$ overall score on
  safety-critical versus reasoning questions, per domain and base model. The
  gain concentrates in safety-critical questions, not general reasoning.}
  \label{fig:safety-vs-reasoning}
\end{figure*}""",
    "fig_research_design": r"""\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figures/fig_research_design.pdf}
  \caption{ADAPT-AI evaluation pipeline: a LangGraph \texttt{StateGraph} over an
  in-process FastMCP hub. Compliance and quality agents run in parallel; a single
  quality-triggered retry loops back to the primary agent.}
  \label{fig:research-design}
\end{figure}""",
    "fig_baseline_ladder": r"""\begin{figure}[t]
  \centering
  \includegraphics[width=0.92\columnwidth]{figures/fig_baseline_ladder.pdf}
  \caption{Component attribution. Mean overall score for each baseline-ladder
  rung (\texttt{b0\_bare}$\rightarrow$\texttt{b3\_persona}) versus the full
  ADAPT-AI pipeline, per domain.}
  \label{fig:baseline-ladder}
\end{figure}""",
    "fig_ablation": r"""\begin{figure}[t]
  \centering
  \includegraphics[width=0.86\columnwidth]{figures/fig_ablation.pdf}
  \caption{Component ablation. ADAPT-AI overall score with each module removed
  ($-$quality, $-$compliance, $-$disclaimer) relative to the full pipeline, per
  domain; labels show the change versus full.}
  \label{fig:ablation}
\end{figure}""",
}

TEX_HEADER = ("%% Auto-generated by scripts/make_figures.py - do not edit by hand.\n"
              "%% \\input this file; PDFs live in the figures/ directory.\n\n")


def main() -> None:
    _setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    formats = ["pdf", "png"]
    eng = _tex_engine()
    if eng:
        matplotlib.rcParams["pgf.texsystem"] = eng
        formats.append("pgf")
        print(f"[make_figures] TeX engine '{eng}' found - also emitting .pgf")
    else:
        print("[make_figures] No TeX engine on PATH - emitting .pdf/.png only "
              "(install xelatex/pdflatex for .pgf)")

    cells = ms.load_all_cells()
    if not cells:
        sys.exit("ERROR: no matrix results in data/evaluation/matrix/. "
                 "Run scripts/run_matrix.py first.")
    print(f"[make_figures] loaded {len(cells)} (model,domain) cells")

    produced: list[str] = []          # figure stems that yielded output
    written: list[str] = []           # individual files written
    plan = [
        ("fig_safety_floor", lambda: fig_safety_floor(cells, formats)),
        ("fig_safety_vs_reasoning", lambda: fig_safety_vs_reasoning(cells, formats)),
        ("fig_research_design", lambda: fig_research_design(formats)),
        ("fig_baseline_ladder", lambda: fig_baseline_ladder(formats)),
        ("fig_ablation", lambda: fig_ablation(formats)),
    ]
    for stem, fn in plan:
        files = fn()
        if files:
            produced.append(stem)
            written += files

    # Assemble figures.tex from the blocks that were actually produced.
    blocks = [TEX_BLOCKS[stem] for stem in produced if stem in TEX_BLOCKS]
    tex_path = FIG_DIR / "figures.tex"
    tex_path.write_text(TEX_HEADER + "\n\n".join(blocks) + "\n", encoding="utf-8")

    print(f"[make_figures] wrote {len(written)} files to {FIG_DIR}/")
    for name in written:
        print(f"    - {name}")
    print(f"    - figures.tex  ({len(blocks)} \\includegraphics blocks + captions)")


if __name__ == "__main__":
    main()
