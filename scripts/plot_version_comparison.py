# /// script
# requires-python = ">=3.12"
# dependencies = ["plotly==7.1.0", "kaleido==1.5.0"]
# ///
"""Render the recorded release comparison. Chrome is required for PNG/SVG export."""

import argparse
import json
import statistics
from pathlib import Path

VERSIONS = ("0.7.1", "0.8.3")
COLORS = ("#697586", "#087f8c")


def paired(root: Path, relative: str) -> list[dict]:
    records = [json.loads((root / relative.format(version=v)).read_text()) for v in VERSIONS]
    for field in ("sample_sha256", "documents", "languages", "seed", "n_per_language", "suite", "dataset"):
        if field not in records[0] or field not in records[1] or records[0][field] != records[1][field]:
            raise ValueError(f"Mismatched or missing {field}: {relative}")
    if not records[0]["sample_sha256"]:
        raise ValueError(f"Missing sample hash: {relative}")
    for version, record in zip(VERSIONS, records, strict=True):
        if record["run"]["sct_version"] != version:
            raise ValueError(f"Wrong installed version: {relative}")
    for field in ("scoring_overrides", "threads", "workers", "configs"):
        if records[0]["run"][field] != records[1]["run"][field]:
            raise ValueError(f"Mismatched {field}: {relative}")
    return records


def summarize(root: Path) -> dict:
    datasets = sorted(p.stem for p in (root / "quality" / VERSIONS[0]).glob("*.json") if p.stem not in ("run", "wili"))
    if len(datasets) != 12:
        raise ValueError("Expected all 12 PII/long-document datasets")
    quality = []
    for dataset in datasets:
        records = paired(root, f"quality/{{version}}/{dataset}.json")
        row = {"dataset": dataset, "documents": records[0]["documents"]}
        for version, record in zip(VERSIONS, records, strict=True):
            row[version] = {
                config: {
                    "strict_f1": values["strict"]["overall"]["f1"],
                    "overlap_masking_recall": values["partial"]["overall"]["masked_recall"],
                    "leak_heuristic": values["leak"]["overall"]["leak_rate"],
                }
                for config, values in record["configs"].items()
            }
        quality.append(row)
    speed = {version: {config: [] for config in ("default", "training")} for version in VERSIONS}
    sample_hashes = set()
    for repeat in (1, 2, 3):
        for version, record in zip(VERSIONS, paired(root, f"speed/{repeat}/{{version}}/fineweb2.json"), strict=True):
            sample_hashes.add(record["sample_sha256"])
            for config in speed[version]:
                speed[version][config].append(record["configs"][config]["total"]["docs_per_second"])
    if len(sample_hashes) != 1:
        raise ValueError("Speed repeats use different samples")
    lid = paired(root, "quality/{version}/wili.json")
    tab = paired(root, "quality/{version}/tab.json")
    limits = {}
    for index, version in enumerate(VERSIONS):
        language = lid[index]["configs"]["default"]
        masking = tab[index]["configs"]["default"]["masking_recall"]
        limits[version] = {
            "known_language_accuracy": language["accuracy"],
            "unsupported_language_false_answers": language["out_of_set"]["false_positive_rate"],
            "tab_direct_full_coverage": masking["direct"]["macro_recall"],
            "tab_quasi_full_coverage": masking["quasi"]["macro_recall"],
        }
    return {"quality": quality, "speed_docs_per_second": speed, "limits": limits}


def render(root: Path, summary: dict) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output = root / "charts"
    output.mkdir(exist_ok=True)

    def save(fig, name, title, note, height=650):
        fig.update_layout(
            title={"text": title, "x": 0.04}, template="plotly_white", width=1200, height=height,
            font={"family": "Arial, sans-serif", "size": 17, "color": "#182536"},
            margin={"l": 140, "r": 45, "t": 170, "b": 135},
            legend={"orientation": "h", "y": 1.10},
        )
        fig.add_annotation(text=note, x=0, y=-0.22, xref="paper", yref="paper", showarrow=False,
                           xanchor="left", align="left", font={"size": 14})
        for extension in ("png", "svg"):
            fig.write_image(output / f"{name}.{extension}", scale=2 if extension == "png" else 1)
        fig.write_html(output / f"{name}.html", include_plotlyjs="cdn", div_id=f"sct-{name}")

    fig = go.Figure()
    for version, color in zip(VERSIONS, COLORS, strict=True):
        samples = [summary["speed_docs_per_second"][version][config] for config in ("default", "training")]
        medians = [statistics.median(values) for values in samples]
        fig.add_bar(name=f"v{version}", x=["Default", "Training"], y=medians, marker_color=color,
                    text=[f"{value:.3f}" for value in medians], textposition="outside",
                    error_y={"type": "data", "symmetric": False,
                             "array": [max(values) - median for values, median in zip(samples, medians, strict=True)],
                             "arrayminus": [median - min(values) for values, median in zip(samples, medians, strict=True)]})
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Documents per second (higher is better)", range=[0, 3.2])
    save(fig, "speed", "SCT measured stacks: 2.27× default, 1.59× training throughput",
         "1,000 FineWeb2 documents; CPU; median and observed range of 3 runs (not confidence intervals).<br>"
         "ORT 1.23.2 vs 1.31.1. Legacy ORT is below SCT's declared minimum. No isolated code-speed claim.")

    rows = summary["quality"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Strict span F1", "Any-overlap masking recall"), shared_yaxes=True)
    for col, metric in enumerate(("strict_f1", "overlap_masking_recall"), 1):
        for version, color, symbol in zip(VERSIONS, COLORS, ("circle-open", "diamond"), strict=True):
            fig.add_scatter(x=[100 * row[version]["default"][metric] for row in rows],
                            y=[row["dataset"] for row in rows], mode="markers", name=f"v{version}",
                            legendgroup=version, showlegend=col == 1,
                            marker={"color": color, "symbol": symbol, "size": 12}, row=1, col=col)
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    save(fig, "quality", "All 12 datasets: span quality and masking overlap",
         "Default + scoring overrides; same samples and released scorer. Higher is better.<br>"
         "Any overlap can leave part of an identifier visible. It does not establish full redaction.", height=850)

    fig = go.Figure()
    for version, color, symbol in zip(VERSIONS, COLORS, ("circle-open", "diamond"), strict=True):
        fig.add_scatter(x=[100 * row[version]["default"]["leak_heuristic"] for row in rows],
                        y=[row["dataset"] for row in rows], mode="markers", name=f"v{version}",
                        marker={"color": color, "symbol": symbol, "size": 12})
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title="Heuristic leak rate (lower is better)")
    save(fig, "leaks", "Visible-value leak heuristic falls on all 12 datasets",
         "Default + scoring overrides. DATE excluded. Literal values / words of 4+ characters.<br>"
         "Repeated words and scripts without spaces can distort this heuristic; no privacy guarantee.", height=850)

    labels = ["Known-language accuracy ↑", "Unsupported false answers ↓",
              "TAB direct full coverage ↑", "TAB quasi full coverage ↑"]
    fig = go.Figure()
    for version, color in zip(VERSIONS, COLORS, strict=True):
        values = [100 * value for value in summary["limits"][version].values()]
        fig.add_bar(x=labels, y=values, name=f"v{version}", marker_color=color,
                    text=[f"{value:.2f}%" for value in values], textposition="outside")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 110], ticksuffix="%")
    save(fig, "limits", "Language scope expands; two checks worsen",
         "WiLI: 7,200 known / 16,300 unsupported documents; old detector offered 4 languages.<br>"
         "TAB: full alphanumeric coverage, macro over 12 annotators; direct 1,612 / quasi 19,197 mentions.", height=700)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    summary = summarize(args.bundle)
    (args.bundle / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(args.bundle, summary)


if __name__ == "__main__":
    main()
