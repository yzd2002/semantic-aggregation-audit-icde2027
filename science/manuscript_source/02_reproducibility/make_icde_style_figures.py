#!/usr/bin/env python3
"""Unified, self-contained generator for all six paper figures.

The operator and three body result figures use the November 2 redesign.
The order_stability and wands_quality functions retain the original source.
All input paths are command-line parameters. No workspace-module imports.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from datetime import datetime, timezone
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.text import Text

INK='#25282a'; MUTED='#555b5e'; TEAL='#c3dcd6'; LIGHT='#f6f7f4'; CREAM='#f9ebd5'
WARM=['#f7ddae','#efb76c','#dd8247','#c84732','#e8d3b1','#ad2e2b']
HATCH=['','///','xxx','\\\\','...','---']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.2,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                     'axes.linewidth':.6,'xtick.major.width':.55,'ytick.major.width':.55,'hatch.linewidth':.45,
                     'savefig.facecolor':'white','figure.facecolor':'white','svg.hashsalt':'intentsat2s-icde2027-v1'})


OP_WIDTH, OP_HEIGHT = 7.10, 2.65
OP_INK = "#202D34"
OP_MUTED = "#53616A"
OP_LINE = "#BCC6CA"
OP_TEAL = "#246B70"
OP_TEAL_PALE = "#EAF3F2"
OP_RUST = "#9B4B2A"
OP_RUST_PALE = "#F8EBE3"
OP_GREY_PALE = "#F2F4F4"


def main_figure(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 8.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "intentsat2s-execution-audit-v2",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }):
        fig = plt.figure(figsize=(OP_WIDTH, OP_HEIGHT))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, OP_WIDTH), ylim=(0, OP_HEIGHT))
        ax.axis("off")
        texts = []

        def text(x, y, s, fs=8.2, color=OP_INK, weight="normal", ha="left", **kwargs):
            t = ax.text(x, y, s, fontsize=fs, color=color, fontweight=weight,
                        ha=ha, va="center", **kwargs)
            texts.append(t)
            return t

        def line(points, color=OP_LINE, lw=.65, **kwargs):
            ax.plot(*zip(*points), color=color, lw=lw, solid_capstyle="round", **kwargs)

        def arrow(a, b, color=OP_MUTED, lw=.8, **kwargs):
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7,
                                        linewidth=lw, color=color, shrinkA=0,
                                        shrinkB=0, **kwargs))

        def rounded(x, y, w, h, fc="white", ec=OP_LINE, lw=.65):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                       boxstyle="round,pad=0,rounding_size=.045",
                                       facecolor=fc, edgecolor=ec, linewidth=lw))

        # A: two inputs join at the judgment stage; only the structure gate
        # controls emission. The specification is a plan choice, not evidence.
        text(.07, 2.53, "A  ID-based fixed-pool execution", fs=9.5, weight="bold")
        text(7.02, 2.53, "Structure checks do not certify semantics", fs=8.2,
             color=OP_MUTED, ha="right")

        text(.08, 2.20, "Query $q$", fs=8.8, weight="bold")
        arrow((.83, 2.20), (1.28, 2.20), color=OP_TEAL)
        rounded(1.28, 2.035, 2.42, .33, fc=OP_TEAL_PALE, ec="#C4DAD8")
        text(2.49, 2.20, "Query-only specification", fs=8.8, weight="bold", ha="center")
        arrow((2.86, 2.035), (2.86, 1.91), color=OP_TEAL)

        text(.08, 1.83, "Fixed pool $C_q$", fs=8.8, weight="bold")
        # Occurrence tiles emphasize a fixed list and denominator, with no
        # implication that unjudged catalog candidates belong to this pool.
        for x, label in zip((.08, .43, .78, 1.13), ("$c_1$", "$c_2$", "…", "$c_n$")):
            if label == "…":
                text(x+.125, 1.57, label, fs=10, ha="center", color=OP_MUTED)
            else:
                rounded(x, 1.435, .26, .27, fc=OP_GREY_PALE, ec="#CAD1D4", lw=.5)
                text(x+.13, 1.57, label, fs=8.6, ha="center")
        text(.08, 1.29, "Denominator stays $n$", fs=8.2, color=OP_MUTED)
        text(1.69, 1.80, "$q$ + chunk", fs=8.2, color=OP_MUTED, ha="center")
        arrow((1.45, 1.60), (2.02, 1.60))

        rounded(2.02, 1.295, 1.68, .615, fc="white", ec="#97B4B6", lw=.8)
        text(2.86, 1.775, "Judge a chunk", fs=9.0, weight="bold", ha="center")
        text(2.86, 1.575, "≤8 candidates; IDs 1…b", fs=8.2, ha="center")
        text(2.86, 1.405, "Map IDs to pool positions", fs=8.2, color=OP_MUTED, ha="center")
        arrow((3.70, 1.60), (4.03, 1.60))

        line([(4.04, 1.32), (4.04, 2.16)], color=OP_TEAL, lw=2.1)
        text(4.16, 2.03, "Structure gate", fs=9.0, weight="bold")
        text(4.16, 1.81, "IDs · Y/N · count", fs=8.2)
        text(4.16, 1.58, "Invalid? Retry once.", fs=8.2)
        text(4.16, 1.37, "IDs restart per chunk", fs=8.2, color=OP_MUTED)

        # A fork, not a semantic correctness check. Complete means all n
        # occurrences have passed their applicable structural validator.
        line([(5.55, 1.72), (5.75, 1.72)], color=OP_MUTED, lw=.8)
        line([(5.75, 1.46), (5.75, 2.04)], color=OP_MUTED, lw=.8)
        arrow((5.75, 2.04), (5.94, 2.04), color=OP_TEAL)
        arrow((5.75, 1.46), (5.94, 1.46), color=OP_RUST)
        text(6.02, 2.04, "All $n$ labeled", fs=8.7, color=OP_TEAL, weight="bold")
        text(6.02, 1.81, r"$\hat r_q=\sum_i\hat s_i/n$", fs=11.0)
        text(6.02, 1.46, "Unresolved", fs=8.4, color=OP_RUST)
        text(6.02, 1.25, "Abstain", fs=9.3, color=OP_RUST, weight="bold")

        text(.08, 1.075,
             "Spec.: shape check + one retry; invalid → direct fallback or abstain by plan.",
             fs=8.2, color=OP_MUTED)
        line([(.07, .965), (7.02, .965)], color="#B8C3C7", lw=.6)

        # B: fixed reference and two complete, ID-aligned saved outputs.
        # This intentionally does not fill in h_2: a third label can help or
        # hurt. The shared error at ID 3 demonstrates the audit's limit.
        text(.07, .83, "B  Same-pool label diagnostics", fs=9.2, weight="bold")
        text(3.76, .83, "Aggregate accuracy and label fidelity", fs=8.8, weight="bold")
        cols = [1.30, 1.70, 2.10, 2.50]
        row_y = [.455, .275, .095]
        ref, a, b = ("Y", "N", "Y", "N"), ("Y", "Y", "N", "N"), ("Y", "N", "N", "N")

        # A faint column band identifies the routed disagreement. Wrong
        # labels also use bold type and the neighboring explicit FP/FN text.
        ax.add_patch(Rectangle((1.54, .018), .32, .69,
                               facecolor=OP_TEAL_PALE, edgecolor="none"))
        text(.08, .655, "Local ID", fs=8.2, weight="bold")
        for i, x in enumerate(cols, 1):
            text(x, .655, str(i), fs=8.4, weight="bold", ha="center", color=OP_TEAL if i == 2 else OP_INK)
        text(3.14, .655, "#Y / $n$", fs=8.2, weight="bold", ha="center")
        line([(.08, .558), (3.46, .558)], color=OP_LINE, lw=.55)
        for name, labels, y in zip(("Reference $s$", "Pass $a$", "Pass $b$"), (ref, a, b), row_y):
            text(.08, y, name, fs=8.4)
            for j, (x, label) in enumerate(zip(cols, labels)):
                wrong = labels is not ref and label != ref[j]
                if wrong:
                    rounded(x-.133, y-.071, .266, .142, fc=OP_RUST_PALE, ec=OP_RUST, lw=.55)
                text(x, y, label, fs=8.4, ha="center", weight="bold" if wrong else "normal",
                     color=OP_RUST if wrong else OP_INK)
            text(3.14, y, f"{labels.count('Y')}/4", fs=8.6,
                 weight="bold" if labels is not b else "normal", ha="center")
        line([(3.60, .07), (3.60, .70)], color="#D2D9DC", lw=.6)

        text(3.76, .61, "$a$: 1 FP + 1 FN → exact ratio (2/4)", fs=8.4)
        text(3.76, .385, "ID 2 differs → pointwise judgment", fs=8.4, color=OP_TEAL)
        text(3.76, .16, "ID 3 agrees → shared candidate error", fs=8.3, color=OP_RUST)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        box = fig.bbox
        for t in texts:
            bb = t.get_window_extent(renderer)
            if not (box.x0 <= bb.x0 <= bb.x1 <= box.x1 and
                    box.y0 <= bb.y0 <= bb.y1 <= box.y1):
                raise ValueError(f"Text exceeds fixed figure canvas: {t.get_text()!r}, {bb}")
            if t.get_fontsize() < 8.2:
                raise ValueError("Text smaller than the 8.2 pt figure minimum")
        for i, t1 in enumerate(texts):
            bb1 = t1.get_window_extent(renderer)
            for t2 in texts[i+1:]:
                bb2 = t2.get_window_extent(renderer)
                if bb1.overlaps(bb2):
                    raise ValueError(f"Text collision: {t1.get_text()!r} / {t2.get_text()!r}")

        # All labels are explicit and all three vectors are complete. A's
        # exact aggregate hides two errors; ID 2 alone triggers routing.
        fp = sum(x == "Y" and y == "N" for x, y in zip(a, ref))
        fn = sum(x == "N" and y == "Y" for x, y in zip(a, ref))
        disagreement = [i for i, (x, y) in enumerate(zip(a, b), 1) if x != y]
        assert fp == fn == 1
        assert a.count("Y") == ref.count("Y") == 2
        assert disagreement == [2]
        assert a[2] == b[2] != ref[2]
        check = {
            "canvas_inches": [OP_WIDTH, OP_HEIGHT],
            "minimum_nominal_text_pt": min(t.get_fontsize() for t in texts),
            "ieeetran_textwidth_tex_pt": 43*12,
            "included_width_fraction_of_textwidth": .98,
            "minimum_text_pt_after_current_manuscript_scaling": 8.2 * (.98 * 43*12/72.27) / OP_WIDTH,
            "complete_occurrences": 4,
            "reference": list(ref), "pass_a": list(a), "pass_b": list(b),
            "a_false_positives": fp, "a_false_negatives": fn,
            "a_candidate_error": (fp+fn)/4,
            "a_absolute_ratio_error": abs(a.count("Y")-ref.count("Y"))/4,
            "routed_local_ids": disagreement,
            "shared_wrong_local_ids": [3],
            "pointwise_outcome_is_unspecified": True,
            "conditional_bounds_or_confidence_intervals_shown": False,
            "all_text_within_fixed_canvas": True,
            "all_text_bounding_boxes_nonoverlapping": True,
            "ratio_estimand_succeeds_for_pass_a": True,
            "candidate_fidelity_is_a_separate_diagnostic": True,
        }
        for ext in ("pdf", "svg", "png"):
            if ext == "pdf":
                dt = datetime(2026, 11, 2, tzinfo=timezone.utc)
                metadata = {"Creator": "IntentSat-2S original vector figure generator",
                            "CreationDate": dt, "ModDate": dt}
            elif ext == "svg":
                metadata = {"Creator": "IntentSat-2S original vector figure generator",
                            "Date": "2026-11-02T00:00:00Z"}
            else:
                metadata = {"Software": "IntentSat-2S original vector figure generator"}
            fig.savefig(out / f"operator.{ext}", dpi=300, metadata=metadata)
        fig.savefig(out / "operator_144dpi.png", dpi=144)
        (out / "operator_checks.json").write_text(json.dumps(check, indent=2)+"\n")
        plt.close(fig)
        return check

RF_WIDTH = 7.10
RF_INK = "#202D34"
RF_MUTED = "#53616A"
RF_LINE = "#CED6D9"
RF_TEAL = "#246B70"
RF_RUST = "#9B4B2A"
RF_SLATE = "#667987"
RF_NEGATIVE = "#F3F8F7"
RF_FONT = 8.3
RF_RC = {
    "font.family": "DejaVu Sans", "font.size": RF_FONT,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "svg.hashsalt": "intentsat2s-results-v2",
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "mathtext.fontset": "dejavusans", "axes.unicode_minus": True,
}



def _rf_read(path):
    return json.loads(Path(path).read_text())

def _rf_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _rf_canvas(height):
    fig = plt.figure(figsize=(RF_WIDTH, height))
    overlay = fig.add_axes([0, 0, 1, 1], zorder=5)
    overlay.set(xlim=(0, RF_WIDTH), ylim=(0, height))
    overlay.axis("off")
    overlay.patch.set_visible(False)
    return fig, overlay

def _rf_text(ax, x, y, s, size=RF_FONT, color=RF_INK, weight="normal", ha="left", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", **kw)

def _rf_rule(ax, x0, x1, y):
    return ax.plot([x0, x1], [y, y], color=RF_LINE, lw=.55, zorder=0)[0]

def _rf_axis(fig, height, x, y, width, h, limits, ticks, ylim, zero=False):
    ax = fig.add_axes([x/RF_WIDTH, y/height, width/RF_WIDTH, h/height], zorder=2)
    ax.set(xlim=limits, ylim=ylim)
    ax.set_xticks(ticks)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#9CA9AF")
    ax.spines["bottom"].set_linewidth(.55)
    ax.tick_params(axis="x", labelsize=RF_FONT, length=2.5, width=.55,
                   pad=3, colors=RF_MUTED)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#E0E6E8", lw=.5)
    if zero:
        ax.axvspan(limits[0], 0, color=RF_NEGATIVE, zorder=0)
        ax.axvline(0, color="#75888F", lw=.9, zorder=1)
    return ax

def _rf_to_inch_y(y, bottom, height, limits):
    return bottom + (y-limits[0])/(limits[1]-limits[0])*height

def _rf_dot(ax, value, interval, y, color, marker="o"):
    lo, hi = map(float, interval)
    value = float(value)
    if not (math.isfinite(value) and lo <= value <= hi):
        raise ValueError("Invalid recorded point or interval")
    seg = ax.hlines(y, lo, hi, color=color, lw=1.15, zorder=3)
    caps, = ax.plot([lo, hi], [y, y], marker="|", linestyle="none", color=color,
                    markersize=5, markeredgewidth=1, zorder=3)
    dot, = ax.plot([value], [y], marker=marker, markersize=5,
                   markeredgecolor=RF_INK, markeredgewidth=.4,
                   color=color, linestyle="none", zorder=4)
    # Verify the actual artist coordinates, not only the copied source dict.
    assert float(dot.get_xdata()[0]) == value
    assert float(dot.get_ydata()[0]) == y
    assert [float(a) for a in caps.get_xdata()] == [lo, hi]
    assert seg.get_segments()[0].tolist() == [[lo, y], [hi, y]]
    return {"estimate": value, "interval": [lo, hi],
            "artist_coordinates_equal_source": True}

def _rf_save(fig, out, name, evidence, sources):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    canvas = fig.bbox
    for t in texts:
        bb = t.get_window_extent(renderer)
        if not (canvas.x0 <= bb.x0 <= bb.x1 <= canvas.x1 and
                canvas.y0 <= bb.y0 <= bb.y1 <= canvas.y1):
            raise ValueError(f"{name}: text outside fixed canvas: {t.get_text()!r}")
        if t.get_fontsize() < 8.2:
            raise ValueError(f"{name}: text below 8.2 pt")
    # All layout text and tick labels have separate physical rectangles.
    for i, t1 in enumerate(texts):
        bb = t1.get_window_extent(renderer)
        for t2 in texts[i+1:]:
            if bb.overlaps(t2.get_window_extent(renderer)):
                raise ValueError(f"{name}: text collision: {t1.get_text()!r} / {t2.get_text()!r}")
    min_font = min(t.get_fontsize() for t in texts)
    evidence.update({
        "canvas_inches": [float(v) for v in fig.get_size_inches()],
        "minimum_nominal_font_pt": min_font,
        "minimum_font_pt_at_0.98_ieeetran_textwidth": min_font * (.98*43*12/72.27) / RF_WIDTH,
        "all_text_inside_canvas": True,
        "text_bounding_boxes_do_not_overlap": True,
        "no_resampling_or_new_intervals": True,
        "source_sha256": {str(Path(p)): _rf_sha(p) for p in sources},
    })
    for ext in ("pdf", "svg", "png"):
        if ext == "pdf":
            dt = datetime(2026, 11, 2, tzinfo=timezone.utc)
            metadata = {"Creator": "IntentSat-2S original vector result figures",
                        "CreationDate": dt, "ModDate": dt}
        elif ext == "svg":
            metadata = {"Creator": "IntentSat-2S original vector result figures",
                        "Date": "2026-11-02T00:00:00Z"}
        else:
            metadata = {"Software": "IntentSat-2S original vector result figures"}
        fig.savefig(out/f"{name}.{ext}", dpi=300, metadata=metadata)
    fig.savefig(out/f"{name}_144dpi.png", dpi=144)
    (out/f"{name}_checks.json").write_text(json.dumps(evidence, indent=2)+"\n")
    plt.close(fig)
    return evidence

def quality_plot(data, out):
    data = Path(data)
    sources = [data/"run_full_pools_250/paired_analysis.json",
               data/"run_supervised_baseline_1000/summary.json",
               data/"run_lotus_baseline/summary.json"]
    report = _rf_read(sources[0])
    supervised = _rf_read(sources[1])["qwen_common_207"]["methods"]["supervised_lexical"]
    lotus = _rf_read(sources[2])["qwen_common_207"]["methods"]["lotus_sem_filter"]
    assert report["common_complete_pools"] == 207
    rows = [
        ("pointwise", "Pointwise", report["methods"]["pointwise"], False),
        ("batch_json", "Batch / JSON", report["methods"]["batch_json"], False),
        ("two_stage_json", "Two-stage / JSON", report["methods"]["two_stage_json"], False),
        ("two_stage_compact_fallback", "Two-stage / compact + fallback", report["methods"]["two_stage_compact_fallback"], False),
        ("lotus_sem_filter", "LOTUS filter (post hoc)", lotus, True),
        ("supervised_lexical", "Supervised lexical (post hoc)", supervised, True),
    ]
    evidence = {"common_complete_pools": report["common_complete_pools"],
                "scope": report["description"], "points": [], "baselines": []}
    with plt.rc_context(RF_RC):
        h = 2.55
        fig, text_ax = _rf_canvas(h)
        _rf_text(text_ax, .07, 2.43, "ESCI: Qwen, 207 common-complete pools", size=9.2, weight="bold")
        # Baselines retain the original values and do not acquire new whiskers.
        text_ax.plot([4.18, 4.49], [2.43, 2.43], color=RF_MUTED, linestyle="--", lw=.9)
        _rf_text(text_ax, 4.56, 2.43, "Always Y", size=8.2)
        text_ax.plot([5.34, 5.65], [2.43, 2.43], color=RF_TEAL, linestyle=":", lw=1)
        _rf_text(text_ax, 5.72, 2.43, "All query words", size=8.2)
        bottom, plot_h, ylim = .40, 1.67, (-.5, 5.5)
        panels = [
            (2.05, 4.40, "f1", "A  Candidate F1  ↑", (0, 1), [0, .25, .5, .75, 1]),
            (4.65, 7.02, "mae", "B  Pool-ratio MAE  ↓", (0, .8), [0, .2, .4, .6, .8]),
        ]
        sep = _rf_to_inch_y(1.5, bottom, plot_h, ylim)
        _rf_rule(text_ax, .07, 7.02, sep)
        for i, (_, name, _, _) in enumerate(rows):
            _rf_text(text_ax, .08, _rf_to_inch_y(5-i, bottom, plot_h, ylim), name, size=8.3)
        for x, value_x, metric, title, limits, ticks in panels:
            ax = _rf_axis(fig, h, x, bottom, 2.00, plot_h, limits, ticks, ylim)
            _rf_text(text_ax, x, 2.22, title, size=9.0, weight="bold")
            _rf_text(text_ax, value_x, 2.22, "Est.", size=8.2, color=RF_MUTED, ha="right")
            for baseline, color, style in (("always_yes", RF_MUTED, "--"),
                                            ("lexical_all_terms", RF_TEAL, ":")):
                value = report["methods"][baseline][metric]
                artist = ax.axvline(value, color=color, linestyle=style, lw=.9, zorder=1)
                assert all(float(a) == value for a in artist.get_xdata())
                evidence["baselines"].append({"metric": metric, "method": baseline,
                                               "estimate": value, "artist_equals_source": True})
            for i, (key, label, source, posthoc) in enumerate(rows):
                y = 5-i
                field = "ratio_mae" if posthoc and metric == "mae" else metric
                value, interval = source[field], source[field+"_ci"]
                rec = _rf_dot(ax, value, interval, y, RF_RUST if posthoc else RF_TEAL,
                              marker="s" if posthoc else "o")
                annotation = f"{value:.3f}"
                _rf_text(text_ax, value_x, _rf_to_inch_y(y, bottom, plot_h, ylim),
                          annotation, size=8.3, ha="right")
                evidence["points"].append({"method": key, "metric": metric,
                                            "display_estimate": annotation, **rec})
        _rf_text(text_ax, .08, .075, "Points: estimates; whiskers: 95% query-bootstrap intervals", size=8.2, color=RF_MUTED)
        return _rf_save(fig, out, "quality", evidence, sources)

def model_plot(data, out):
    path = Path(data)/"POSTHOC_7B_FIGURE_DATA.json"
    source = _rf_read(path)
    d = source["values"]
    datasets = ("ESCI", "WANDS", "TREC-COVID")
    models = (("Qwen3-4B", RF_TEAL, "o"),
              ("Qwen2.5-7B", RF_RUST, "s"),
              ("SmolLM2-1.7B", RF_SLATE, "^"))
    if source.get("fixture") is not False or set(d) != set(datasets):
        raise ValueError("Expected recorded model sensitivity data")
    evidence = {"contrast": "selective_mae - keyed_mae", "points": []}
    with plt.rc_context(RF_RC):
        h = 2.45
        fig, text_ax = _rf_canvas(h)
        _rf_text(text_ax, .07, 2.33, "Model sensitivity (post hoc)", size=9.4, weight="bold")
        _rf_text(text_ax, 7.02, 2.33, "Annotations: keyed → selective MAE", size=8.2, color=RF_MUTED, ha="right")
        bottom, plot_h, ylim = .47, 1.40, (-.65, 2.45)
        x_positions = (1.23, 3.23, 5.23)
        for i, (name, _, _) in enumerate(models):
            _rf_text(text_ax, .07, _rf_to_inch_y(2-i, bottom, plot_h, ylim), name, size=8.3)
        for x, dataset in zip(x_positions, datasets):
            assert set(d[dataset]) == {x[0] for x in models}
            counts = {d[dataset][name]["pools"] for name, _, _ in models}
            assert len(counts) == 1
            n = counts.pop()
            _rf_text(text_ax, x+.89, 2.05, f"{dataset} (n={n})", size=9.0, weight="bold", ha="center")
            ax = _rf_axis(fig, h, x, bottom, 1.78, plot_h,
                          (-.07, .045), [-.06, -.03, 0, .03], ylim, zero=True)
            for i, (name, color, marker) in enumerate(models):
                row = d[dataset][name]
                if not math.isclose(row["selective_mae"]-row["keyed_mae"], row["difference"], abs_tol=1e-10):
                    raise ValueError("Inconsistent recorded model contrast")
                y = 2-i
                rec = _rf_dot(ax, row["difference"], row["interval"], y, color, marker)
                label = f"{row['keyed_mae']:.3f} → {row['selective_mae']:.3f}"
                _rf_text(text_ax, x+.89, _rf_to_inch_y(y-.36, bottom, plot_h, ylim),
                          label, size=8.3, color=RF_MUTED, ha="center",
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
                evidence["points"].append({"dataset": dataset, "model": name, "pools": n,
                    "keyed_mae": row["keyed_mae"], "selective_mae": row["selective_mae"],
                    "display_absolute_maes": label, **rec})
        _rf_text(text_ax, 4.12, .10, "Selective − keyed ratio MAE  (lower is better)", size=8.5, ha="center")
        return _rf_save(fig, out, "model_sensitivity", evidence, [path])

def routing_plot(result, out):
    result = Path(result)
    report = _rf_read(result)
    settings = list(report["settings"].items())
    assert [k for k, _ in settings] == ["WANDS/Qwen", "WANDS/SmolLM2", "ESCI/Qwen", "ESCI/SmolLM2"]
    assert [row["pools"] for _, row in settings] == [52, 52, 250, 250]
    evidence = {"contrast": "routed - expected uniform", "points": [],
                "scope": report["scope"], "random_reference": report["random_reference"]}
    with plt.rc_context(RF_RC):
        h = 2.30
        fig, text_ax = _rf_canvas(h)
        _rf_text(text_ax, .07, 2.18, "Matched-budget routing (post hoc)", size=9.3, weight="bold")
        _rf_text(text_ax, 7.02, 2.18, "Equal per-pool pointwise-label budgets", size=8.2, color=RF_MUTED, ha="right")
        bottom, plot_h, ylim = .44, 1.29, (-.55, 3.55)
        sep = _rf_to_inch_y(1.5, bottom, plot_h, ylim)
        _rf_rule(text_ax, .07, 7.02, sep)
        for dataset, n, y in (("WANDS", 52, 2.5), ("ESCI", 250, .5)):
            center = _rf_to_inch_y(y, bottom, plot_h, ylim)
            _rf_text(text_ax, .07, center+.085, dataset, size=8.7, weight="bold")
            _rf_text(text_ax, .07, center-.09, f"n={n}", size=8.2, color=RF_MUTED)
        for i, (name, row) in enumerate(settings):
            _rf_text(text_ax, .83, _rf_to_inch_y(3-i, bottom, plot_h, ylim), name.split("/")[1], size=8.3)
        for x, val_x, metric, title in ((1.90, 4.39, "ratio_mae", "A  Ratio MAE"),
                                        (4.64, 7.02, "candidate_error", "B  Candidate error")):
            width = 2.05 if metric == "ratio_mae" else 2.00
            ax = _rf_axis(fig, h, x, bottom, width, plot_h,
                          (-.028, .016), [-.02, -.01, 0, .01], ylim, zero=True)
            _rf_text(text_ax, x, 1.89, title, size=9.0, weight="bold")
            _rf_text(text_ax, val_x, 1.89, "Δ", size=8.5, color=RF_MUTED, ha="right")
            for i, (name, row) in enumerate(settings):
                r = row["routed_minus_expected_random"][metric]
                diff, interval = r["difference"], r["unadjusted_query_interval"]
                field = "mae" if metric == "ratio_mae" else "candidate_error"
                assert math.isclose(row["means"]["routed_"+field]-row["means"]["expected_random_"+field], diff, abs_tol=1e-12)
                qwen = name.endswith("/Qwen")
                rec = _rf_dot(ax, diff, interval, 3-i, RF_TEAL if qwen else RF_SLATE,
                              "o" if qwen else "^")
                label = f"{diff:+.3f}".replace("-", "−")
                _rf_text(text_ax, val_x, _rf_to_inch_y(3-i, bottom, plot_h, ylim), label,
                          size=8.3, ha="right")
                evidence["points"].append({"setting": name, "metric": metric, "pools": row["pools"],
                    "pointwise_label_budget": row["pointwise_label_budget"], "display_difference": label, **rec})
        _rf_text(text_ax, 4.44, .10, "Routed − expected uniform error  (lower is better)", size=8.5, ha="center")
        return _rf_save(fig, out, "routing_budget", evidence, [result])

def save(fig,out,name):
    for suffix in ('pdf','svg','png'):
        if suffix=='pdf':metadata={'Creator':'IntentSat-2S figure generator','CreationDate':datetime(2026,10,31,tzinfo=timezone.utc),'ModDate':datetime(2026,10,31,tzinfo=timezone.utc)}
        elif suffix=='svg':metadata={'Creator':'IntentSat-2S figure generator','Date':'2026-10-31T00:00:00Z'}
        else:metadata={'Software':'IntentSat-2S figure generator'}
        fig.savefig(out/(name+'.'+suffix),dpi=300 if suffix=='png' else None,bbox_inches='tight',pad_inches=.035,metadata=metadata)
    plt.close(fig)

def style_axis(ax):
    for s in ax.spines.values():s.set_color('#555');s.set_linewidth(.55);s.set_visible(True)
    ax.set_axisbelow(True);ax.grid(axis='x',color='#d8d8d4',linewidth=.45)
    ax.tick_params(axis='both',labelsize=7.0,length=2.1,pad=2,width=.55,colors=INK)

def horizontal_bar(ax,y,v,lo,hi,color,hatch):
    if not (math.isfinite(v) and lo<=v<=hi):raise ValueError('Invalid plotted estimate or interval')
    ax.barh(y,v,height=.57,color=color,edgecolor='#613c2d',linewidth=.45,hatch=hatch,zorder=2)
    ax.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=INK,markersize=2.4,elinewidth=.7,capsize=2,capthick=.7,zorder=4)

def order_plot(data,out):
    q=json.loads((data/'run_reverse_order_qwen_v1/paired_analysis.json').read_text())['datasets']
    s=json.loads((data/'run_reverse_order_smollm_v1/paired_analysis.json').read_text())['datasets']
    rows=[('Qwen / ESCI',q['esci'],WARM[2],'///'),('Qwen / WANDS',q['wands'],WARM[2],'///'),
          ('SmolLM2 / ESCI',s['esci'],WARM[0],'...'),('SmolLM2 / WANDS',s['wands'],WARM[0],'...')]
    y=[3,2,1,0];fig,axes=plt.subplots(1,2,figsize=(7.05,2.22),sharey=True)
    for ax,metric,interval,title in ((axes[0],'candidate_label_change_fraction','candidate_label_change_fraction_ci','A  Candidate labels changed (%)'),
                                     (axes[1],'mean_absolute_pool_ratio_change','mean_absolute_pool_ratio_change_ci','B  Mean absolute ratio change (pp)')):
        for yi,(_,row,color,hatch) in zip(y,rows):
            v=100*row[metric];lo,hi=(100*z for z in row['paired_bootstrap'][interval]);horizontal_bar(ax,yi,v,lo,hi,color,hatch)
        upper=max(row['paired_bootstrap'][interval][1] for _,row,_,_ in rows)*100
        ax.set_xlim(0,min(100,max(10,(int(upper/10)+1)*10)));ax.set_ylim(-.56,3.56)
        ax.set_title(title,fontsize=8.0,fontweight='bold',pad=5);style_axis(ax)
    axes[0].set_yticks(y,[row[0] for row in rows],fontsize=7.1);axes[1].tick_params(labelleft=False)
    fig.subplots_adjust(left=.23,right=.99,top=.83,bottom=.15,wspace=.19)
    save(fig,out,'order_stability')

def wands_plot(data,out):
    report=json.loads((data/'run_wands_160/paired_analysis.json').read_text());keys=['pointwise','batch_json','two_stage_json','two_stage_compact_fallback']
    names=['Pointwise','Batch / JSON','Two-stage / JSON','Two-stage / compact + fallback'];y=[3,2,1,0]
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.30),sharey=True)
    for ax,metric,title in ((axes[0],'f1','A  Candidate F1  ↑'),(axes[1],'mae','B  Pool-ratio MAE  ↓')):
        for i,(yi,key) in enumerate(zip(y,keys)):
            row=report['methods'][key];v=row[metric];lo,hi=row[metric+'_ci'];horizontal_bar(ax,yi,v,lo,hi,WARM[i],HATCH[i])
        ax.axvline(report['methods']['always_no'][metric],color='#484848',linestyle='--',lw=.8,zorder=3)
        ax.axvline(report['methods']['lexical_all_terms'][metric],color='#356c66',linestyle=':',lw=1.,zorder=3)
        ax.set_title(title,fontsize=8.0,fontweight='bold',pad=4);ax.set_ylim(-.55,3.55);style_axis(ax)
    axes[0].set_yticks(y,names,fontsize=7.0);axes[1].tick_params(labelleft=False);axes[0].set_xlim(0,.8);axes[1].set_xlim(0,.45)
    legend=[Line2D([0],[0],color='#484848',linestyle='--',lw=.8,label='Always N'),Line2D([0],[0],color='#356c66',linestyle=':',lw=1.,label='All query words')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.65,1.01),ncol=2,fontsize=6.6,frameon=False,handlelength=2,columnspacing=1.4)
    fig.subplots_adjust(left=.32,right=.99,top=.77,bottom=.14,wspace=.19);save(fig,out,'wands_quality')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--routing-result',type=Path,help='Optional post-hoc matched-label-budget result from the two-order supplement.');a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh figure output directory')
    a.out.mkdir(parents=True)
    main_figure(a.out);quality_plot(a.data_dir,a.out);order_plot(a.data_dir,a.out);model_plot(a.data_dir,a.out);wands_plot(a.data_dir,a.out)
    if a.routing_result:routing_plot(a.routing_result,a.out)
    sources=['run_full_pools_250/paired_analysis.json','run_supervised_baseline_1000/summary.json','run_lotus_baseline/summary.json',
             'run_reverse_order_qwen_v1/paired_analysis.json','run_reverse_order_smollm_v1/paired_analysis.json','POSTHOC_7B_FIGURE_DATA.json','run_wands_160/paired_analysis.json']
    manifest={'generator_sha256':sha(Path(__file__)),'matplotlib':matplotlib.__version__,'figure_inputs_sha256':{x:sha(a.data_dir/x) for x in sources},
              'figure_files_sha256':{p.name:sha(p) for p in sorted(a.out.iterdir()) if p.is_file()}}
    if a.routing_result:manifest['routing_result_sha256']=sha(a.routing_result)
    (a.out/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated original vector figures, editable SVGs and PNG previews; matched-label routing panel included:',bool(a.routing_result))

if __name__ == '__main__':
    main()
