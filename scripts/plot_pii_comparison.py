# /// script
# requires-python = ">=3.12"
# dependencies = ["plotly==7.1.0", "kaleido==1.5.0"]
# ///
"""Chart the archived SCT v0.8.2 comparison without changing its result bundle."""

import argparse
import hashlib
import json
from pathlib import Path


def extract(source: Path) -> dict:
    raw = source.read_bytes()
    data = json.loads(raw)
    if not data["paired_note"].startswith("a = sct-default, b = entry"):
        raise ValueError("Unknown direction of paired differences")
    rows = []
    for entry, views in sorted(data["pairwise"].items()):
        if entry.startswith("sct-"):
            continue
        row = {"entry": entry}
        for view in ("shared", "full"):
            result = views[view]["r2-privacy-synthetic"]["clean"]
            metrics = {}
            for name in ("masked_recall", "masked_precision", "doc_leak_rate"):
                metric = result["metrics"][name]
                lo, hi = metric["diff_ci"]
                if not (-1 <= lo <= hi <= 1 and 0 <= metric["a"] <= 1 and 0 <= metric["b"] <= 1):
                    raise ValueError(f"Invalid metric: {entry}/{view}/{name}")
                metrics[name] = {"sct": metric["a"], "other": metric["b"],
                                 "difference": metric["a"] - metric["b"], "ci95": [lo, hi]}
            if not result["datasets"] or result["groups"] <= 0:
                raise ValueError(f"Empty paired comparison: {entry}/{view}")
            row[view] = {"datasets": result["datasets"], "groups": result["groups"], "metrics": metrics}
        rows.append(row)
    if len(rows) != 14:
        raise ValueError("Expected all 14 non-SCT entries")
    return {"sct_version": "0.8.2", "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source": "compare-2026-11-04/default/summary.json", "rows": rows}


def render(data: dict, output: Path) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    output.mkdir(parents=True, exist_ok=True)
    (output / "plot-data.json").write_text(json.dumps(data, indent=2) + "\n")
    charts = (
        ("shared-recall", "Shared types: SCT v0.8.2 trails four tools on paired recall", "shared",
         [("masked_recall", "Masking recall: higher is better")]),
        ("full-recall", "Full label set: SCT v0.8.2 has stronger paired recall", "full",
         [("masked_recall", "Masking recall: higher is better")]),
        ("tradeoffs", "SCT v0.8.2: stronger recall comes with tradeoffs", "full",
         [("masked_precision", "Masking precision: higher is better"),
          ("doc_leak_rate", "Documents with exposed gold text: lower is better")]),
    )
    for filename, title, view, metrics in charts:
        rows = sorted(data["rows"], key=lambda r: r[view]["metrics"]["masked_recall"]["difference"])
        fig = make_subplots(rows=1, cols=len(metrics), shared_yaxes=True,
                            subplot_titles=[label for _, label in metrics])
        for col, (name, _) in enumerate(metrics, 1):
            values = [row[view]["metrics"][name] for row in rows]
            for row, metric in zip(rows, values, strict=True):
                x = 100 * metric["difference"]
                lo, hi = [100 * value for value in metric["ci95"]]
                fig.add_trace(go.Scatter(x=[x], y=[row["entry"]], mode="markers", showlegend=False,
                                        marker={"size": 11, "color": "#087f8c", "symbol": "diamond"},
                                        error_x={"type": "data", "symmetric": False,
                                                 "array": [max(0, hi - x)], "arrayminus": [max(0, x - lo)]},
                                        customdata=[[lo, hi, 100 * metric["sct"], 100 * metric["other"]]],
                                        hovertemplate="SCT v0.8.2: %{customdata[2]:.2f}%<br>%{y}: %{customdata[3]:.2f}%"
                                                      "<br>SCT minus other: %{x:.2f} pp<br>95% CI [%{customdata[0]:.2f}, "
                                                      "%{customdata[1]:.2f}]<extra></extra>"), row=1, col=col)
            fig.add_vline(x=0, line_dash="dash", line_color="#697586", row=1, col=col,
                          annotation_text="SCT reference<br>0 = equal scores",
                          annotation_position="top", annotation_font_size=13)
            lower = min(0, *(100 * metric["ci95"][0] for metric in values)) - 5
            upper = max(0, *(100 * metric["ci95"][1] for metric in values)) + 5
            direction = ("← SCT better | Other tool better →" if name == "doc_leak_rate"
                         else "← Other tool better | SCT better →")
            fig.update_xaxes(title_text=f"SCT minus other tool (percentage points)<br>{direction}", row=1, col=col,
                             range=[lower, upper])
        for annotation in fig.layout.annotations[:len(metrics)]:
            annotation.y = 1.08
        label_note = "Shared types: EMAIL, PHONE, NUMBER." if view == "shared" else "Full scored taxonomy; DATE excluded."
        fig.update_layout(template="plotly_white", width=1400, height=1050,
                          title={"text": title, "x": 0.03}, font={"family": "Arial, sans-serif", "size": 18},
                          margin={"l": 205, "r": 70, "t": 195, "b": 250})
        fig.add_annotation(x=0, y=-0.15, xref="paper", yref="paper", showarrow=False,
                           xanchor="left", yanchor="top", align="left",
                           font={"size": 15}, text=f"{label_note} Synthetic datasets only. Default track.<br>"
                           "SCT is the dashed reference line, not a separate tool row. Zero means equal scores.<br>"
                           "Each row: same documents, clean sets, languages both claim; rows use different subsets.<br>"
                           "Points: macro differences. Whiskers: paired bootstrap 95% intervals.<br>"
                           "Intervals are not adjusted for multiple tests.<br>"
                           "Tested SCT v0.8.2. Results do not measure v0.8.3. Source: round 2, 2026-11-04.")
        for extension in ("png", "svg"):
            fig.write_image(output / f"{filename}.{extension}", scale=2 if extension == "png" else 1)
        fig.write_html(output / f"{filename}.html", include_plotlyjs="cdn", div_id=f"sct-r2-{filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(extract(args.source), args.output)
