# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib==3.11.2", "pillow==13.0.0"]
# ///
"""Render measured release results, requiring a matching validation manifest.

Run: python -m scripts.plot_pii_social_release BUNDLE OUTPUT
The bundle contains summary.json, validation.json and native-speed.json.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

TEAL, INK, MUTED, GRAY = "#087F8C", "#172B42", "#43566B", "#657487"

NAMES = {
    "pii-tracer": "PII-Tracer",
    "bardsai-v3": "BardsAI-v3",
    "opf": "OPF",
    "openmed": "OpenMed",
    "gliner2-fastino": "GLiNER2-Fastino",
    "gliner-nvidia": "GLiNER NVIDIA",
    "gliner-knowledgator": "GLiNER Knowledgator",
    "presidio": "Presidio",
    "presidio-full": "Presidio full",
    "llm-guard": "LLM Guard",
    "piiranha": "Piiranha",
    "ai4privacy": "AI4Privacy",
    "datafog": "DataFog",
    "scrubadub": "scrubadub",
}


def extract(bundle):
    raw = (bundle / "summary.json").read_bytes()
    checked = json.loads((bundle / "validation.json").read_text())
    assert checked["passed"] is True
    assert checked["summary_sha256"] == hashlib.sha256(raw).hexdigest()
    assert checked["sct_version"] == "0.10.0"
    assert checked["quality_documents"] == 20817 and checked["quality_cells"] == 12
    assert checked["reused_cells_validated"] == 154
    data = json.loads(raw)
    assert data["paired_note"].startswith("a = sct-default, b = entry")
    rows = []
    for entry, views in sorted(data["pairwise"].items()):
        if entry.startswith("sct-"):
            continue
        row = {"entry": entry}
        for view in ("shared", "full"):
            result = views[view]["r2-privacy-synthetic"]["clean"]
            metrics = {}
            for name in ("masked_recall", "masked_precision", "doc_leak_rate"):
                m = result["metrics"][name]
                lo, hi = m["diff_ci"]
                assert -1 <= lo <= hi <= 1 and all(0 <= m[k] <= 1 for k in ("a", "b"))
                metrics[name] = {"sct": m["a"], "other": m["b"], "difference": m["a"] - m["b"], "ci95": [lo, hi]}
            assert result["datasets"] and result["groups"] > 0
            row[view] = {"datasets": result["datasets"], "groups": result["groups"], "metrics": metrics}
        rows.append(row)
    assert len(rows) == 14
    return {"sct_version": checked["sct_version"], "source_sha256": checked["summary_sha256"], "rows": rows}


def canvas(title, subtitle, *, speed=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 15), dpi=100, facecolor="white")
    scope = "CPU PIPELINE TIMING" if speed else "DEFAULT DETECTORS · SYNTHETIC DATA"
    fig.text(0.065, 0.945, f"SCT v0.10.0 · {scope}", fontsize=22, color=MUTED, weight="bold")
    fig.text(0.065, 0.88, title, va="top", fontsize=39, weight="bold", color=INK, linespacing=1.15)
    fig.text(0.065, 0.735, subtitle, va="top", fontsize=25, color=MUTED, linespacing=1.4)
    return fig


def footer(fig, lines, *, speed=False):
    if not speed:
        lines += "\nGold languages supplied; fixed evaluation samples."
    fig.text(0.065, 0.165, lines, va="top", fontsize=22, color=MUTED, linespacing=1.4)
    source = "FineWeb-2 · 1,000 documents · 10 languages" if speed else "Fixed samples · other-tool versions from earlier run"
    fig.text(0.065, 0.025, source, fontsize=20, color=MUTED)


def export(fig, output, name, alt):
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    alt = "SCT v0.10.0. " + alt
    assert len(alt) <= 1000
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    labels = []
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            bounds = text.get_window_extent(renderer)
            assert bounds.x0 >= 0 and bounds.y0 >= 0, text.get_text()
            assert bounds.x1 <= 1200 and bounds.y1 <= 1500, text.get_text()
            for other, label in labels:
                assert not bounds.overlaps(other), (text.get_text(), label)
            labels.append((bounds, text.get_text()))
    fig.savefig(output / f"{name}.png", dpi=100, facecolor="white")
    svg = output / f"{name}.svg"
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    with Image.open(output / f"{name}.png") as image:
        image.resize((390, 488), Image.Resampling.LANCZOS).save(output / f"{name}-phone.png")
        image.convert("L").resize((390, 488), Image.Resampling.LANCZOS).save(output / f"{name}-gray.png")
    return {"image": f"{name}.png", "alt": alt}


def render(bundle, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = extract(bundle)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
        }
    )
    cards = []
    advantages = sorted(
        (r for r in data["rows"] if r["shared"]["metrics"]["masked_recall"]["ci95"][1] < 0),
        key=lambda r: r["shared"]["metrics"]["masked_recall"]["difference"],
    )
    selected = (
        advantages[:4]
        or sorted(data["rows"], key=lambda r: abs(r["shared"]["metrics"]["masked_recall"]["difference"]), reverse=True)[:4]
    )
    title = (
        f"{len(advantages)} configurations have higher\nshared-label recall than SCT"
        if advantages
        else "Shared-label recall:\nlargest measured gaps"
    )
    fig = canvas(title, "EMAIL / PHONE / NUMBER labels only\nOther tool minus SCT, in percentage points")
    values = []
    for row in selected:
        m = row["shared"]["metrics"]["masked_recall"]
        values.append((-100 * m["difference"], -100 * m["ci95"][1], -100 * m["ci95"][0]))
    bound = max(5, math.ceil(max(abs(v) for triple in values for v in triple) / 5) * 5)
    ax = fig.add_axes((0.08, 0.24, 0.84, 0.40))
    ax.set(xlim=(0 if advantages else -bound, bound + 2), ylim=(-0.6, len(selected) - 0.35), yticks=[])
    ax.tick_params(axis="x", labelsize=23, length=0, pad=14, colors=MUTED)
    ax.spines["bottom"].set_color("#CBD2D8")
    ax.axvline(0, color="#CBD2D8", lw=1)
    for y, row, (value, lo, hi) in zip(reversed(range(len(selected))), selected, values, strict=True):
        ax.text(value, y + 0.31, NAMES.get(row["entry"], row["entry"]), ha="center", fontsize=27, weight="bold")
        ax.errorbar(
            (lo + hi) / 2,
            y,
            xerr=(hi - lo) / 2,
            fmt="none",
            color=GRAY,
            markersize=12,
            elinewidth=4,
            capsize=8,
            capthick=3,
        )
        # Percentile intervals need not contain their point estimate.
        ax.plot(value, y, "o", color=GRAY, markersize=12)
        ax.annotate(
            f"{value:+.1f}",
            (value, y),
            xytext=(0, -30),
            textcoords="offset points",
            ha="center",
            fontsize=24,
            color=GRAY,
            weight="bold",
        )
    selection = (
        f"Shown: {len(selected)} largest gaps of {len(advantages)} higher pairs; 14 total."
        if advantages
        else "Shown: four largest absolute gaps of 14 pairs."
    )
    footer(
        fig,
        selection
        + "\nPaired 95% intervals, unadjusted; pair subsets vary.\nRecall counts any overlap with a labelled mention.",
    )
    details = "; ".join(
        f"{NAMES.get(row['entry'], row['entry'])}: {value:+.2f} points, 95% interval {lo:+.2f} to {hi:+.2f}"
        for row, (value, lo, hi) in zip(selected, values, strict=True)
    )
    cards.append(
        export(
            fig,
            output,
            "01-shared-recall",
            selection
            + " "
            + details
            + ". Shared EMAIL, PHONE, NUMBER. Gold languages supplied. Synthetic dataset averages; pair subsets vary.",
        )
    )

    metrics = [r["full"]["metrics"]["masked_recall"] for r in data["rows"]]
    higher = sum(m["ci95"][0] > 0 for m in metrics)
    lower = sum(m["ci95"][1] < 0 for m in metrics)
    unclear = len(metrics) - higher - lower
    fig = canvas(
        f"Full-label recall:\nSCT higher in {higher} of 14 pairs",
        "Full scored label set · DATE excluded\nUnsupported types count as misses",
    )
    for i in range(14):
        colour = TEAL if i < higher else GRAY if i < higher + unclear else INK
        fig.add_artist(Rectangle((0.065 + i * 0.062, 0.53), 0.049, 0.10, transform=fig.transFigure, color=colour))
    for y, label, colour in [
        (0.46, f"{higher} higher", TEAL),
        (0.39, f"{unclear} unclear", GRAY),
        (0.32, f"{lower} lower", INK),
    ]:
        fig.text(0.065, y, label, fontsize=36, color=colour, weight="bold")
    fig.text(0.065, 0.21, "Broader label coverage affects this result.", fontsize=26, weight="bold")
    footer(
        fig,
        "One block per configuration pair; matched documents.\n"
        "Direction uses unadjusted paired 95% intervals.\n"
        "Any overlap counts; complete redaction is stricter.",
    )
    cards.append(
        export(
            fig,
            output,
            "02-full-recall",
            f"Full-label recall: SCT higher {higher}, unclear {unclear}, lower {lower}. "
            "Unsupported types count as misses. DATE excluded. Unadjusted paired 95% intervals. "
            "Gold languages supplied; synthetic dataset averages, with different matched subsets per pair.",
        )
    )

    pair = next(r for r in data["rows"] if r["entry"] == "pii-tracer")["full"]
    for filename, key, title, definition in [
        (
            "03-precision",
            "masked_precision",
            "Masking precision on\nthe same documents",
            "Predicted spans overlapping a labelled mention",
        ),
        (
            "04-exposed-text",
            "doc_leak_rate",
            "Documents with exposed\nlabelled text",
            "Documents with any labelled letter or digit visible",
        ),
    ]:
        m = pair["metrics"][key]
        lo, hi = [100 * v for v in m["ci95"]]
        if lo > 0 or hi < 0:
            higher_name = "SCT" if lo > 0 else "PII-Tracer"
            title = (
                f"{higher_name} has higher\nmasking precision"
                if key == "masked_precision"
                else f"{higher_name} has higher document\nexposure in scored masks"
            )
        fig = canvas(title, f"SCT default and PII-Tracer\nMeans: {len(pair['datasets'])} synthetic datasets · DATE excluded")
        ax = fig.add_axes((0.065, 0.36, 0.87, 0.29))
        ax.set(xlim=(0, 115), ylim=(-0.5, 1.8), yticks=[], xticks=[])
        ax.spines["bottom"].set_visible(False)
        for y, label, field, colour in [(1, "SCT v0.10.0", "sct", TEAL), (0, "PII-Tracer", "other", GRAY)]:
            value = 100 * m[field]
            ax.text(0, y + 0.35, label, fontsize=29, color=colour, weight="bold")
            ax.barh(y, value, height=0.22, color=colour)
            ax.text(value + 2, y, f"{value:.1f}%", va="center", fontsize=31, color=colour, weight="bold")
        fig.text(0.065, 0.30, f"SCT minus PII-Tracer: {100 * m['difference']:+.1f} points", fontsize=28, weight="bold")
        fig.text(0.065, 0.245, f"Unadjusted 95% interval: {lo:+.1f} to {hi:+.1f} points", fontsize=25, color=MUTED)
        note = (
            "Among documents with scored mentions.\n"
            "Any labelled letter or digit visible counts as exposed.\n"
            "Exposure uses reconstructed source-span masks."
            if key == "doc_leak_rate"
            else definition + ".\nFull scored label set; DATE excluded.\nPaired intervals unadjusted; one illustrative pair."
        )
        footer(fig, note)
        cards.append(
            export(
                fig,
                output,
                filename,
                f"{definition}. SCT {100 * m['sct']:.2f}%, PII-Tracer {100 * m['other']:.2f}%. "
                + (
                    "Denominator: documents with scored mentions. Reconstructed source-span masks. "
                    if key == "doc_leak_rate"
                    else ""
                )
                + f"Means across {len(pair['datasets'])} synthetic datasets. Gold languages supplied. DATE excluded.",
            )
        )

    speed_path = bundle / "native-speed.json"
    speed = json.loads(speed_path.read_text())
    checked = json.loads((bundle / "validation.json").read_text())
    assert checked["native_speed_sha256"] == hashlib.sha256(speed_path.read_bytes()).hexdigest()
    assert speed["repeats"] == 3 and speed["sample"]["documents"] == 1000
    for name, metric, title, units in [
        ("05-throughput", "common_batch_docs_per_second", "CPU batch throughput\nacross three repeats", "documents / second"),
        ("06-memory", "peak_rss_mb", "Peak process memory\nacross three repeats", "GiB of resident memory"),
    ]:
        fig = canvas(
            title, "SCT v0.8.2 and v0.10.0 · same inputs\nDirect pipeline API, automatic language detection", speed=True
        )
        measurements = [speed["entries"][v][metric] for v in ("0.8.2", "0.10.0")]
        if metric == "peak_rss_mb":
            measurements = [{key: value / 1024 for key, value in m.items()} for m in measurements]
        maximum = max(m["max"] for m in measurements)
        ax = fig.add_axes((0.065, 0.34, 0.87, 0.31))
        ax.set(xlim=(0, maximum * 1.18), ylim=(-0.55, 1.65), yticks=[], xticks=[])
        ax.spines["bottom"].set_visible(False)
        for y, version, m, colour in zip([1, 0], ["0.8.2", "0.10.0"], measurements, [GRAY, TEAL], strict=True):
            assert 0 < m["min"] <= m["median"] <= m["max"] and math.isfinite(m["max"])
            ax.text(0, y + 0.33, f"v{version}", fontsize=29, weight="bold", color=colour)
            ax.barh(y, m["median"], height=0.18, color=colour)
            ax.errorbar(
                m["median"],
                y,
                xerr=[[m["median"] - m["min"]], [m["max"] - m["median"]]],
                color=INK,
                elinewidth=3,
                capsize=7,
                fmt="none",
            )
            ax.text(m["max"] + maximum * 0.025, y, f"{m['median']:.1f}", va="center", fontsize=29, weight="bold", color=colour)
        fig.text(0.065, 0.28, units + " · median with observed range", fontsize=26, weight="bold")
        footer(
            fig,
            "Three alternating pairs on a shared host.\n"
            "CPU, 16 threads, 2 workers; throughput batch 8.\n"
            "Ranges describe repeats, not confidence intervals.",
            speed=True,
        )
        cards.append(export(fig, output, name, units + ". Versions 0.8.2 then 0.10.0: " + json.dumps(measurements)))
    manifest = {
        **data,
        "style_source": "https://github.com/skthewimp/karthik-data-visualization-skill",
        "cards": cards,
        "full_recall_counts": {"higher": higher, "unclear": unclear, "lower": lower},
        "speed": speed,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.bundle, args.output)
