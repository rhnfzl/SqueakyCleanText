"""Build editable, phone-readable cards from the archived v0.8.2 paired results.

Export the generated draw.io sources with the desktop CLI documented in the card README.
"""

import argparse
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.plot_pii_comparison import extract

INK = "#172B42"
TEAL = "#087F8C"
ORANGE = "#A44711"
MUTED = "#43566B"
WIDTH, HEIGHT = 1200, 1350
NAMES = {"pii-tracer": "PII-Tracer", "bardsai-v3": "BardsAI-v3", "opf": "OPF", "openmed": "OpenMed"}


class Card:
    def __init__(self, title, subtitle, benchmark=False):
        self.model = ET.Element(
            "mxGraphModel",
            {
                "pageWidth": str(WIDTH),
                "pageHeight": str(HEIGHT),
                "page": "1",
                "background": "#FFFFFF",
                "adaptiveColors": "auto",
            },
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})
        self.box(0, 0, WIDTH, HEIGHT, "#FFFFFF", stroke="none", radius=False)
        self.text(
            "SCT v0.8.2  |  DEFAULT CONFIG" if benchmark else "SQUEAKYCLEANTEXT",
            70,
            40,
            1060,
            50,
            size=36,
            color=TEAL,
            bold=True,
        )
        self.text(title, 70, 110, 1060, 170, size=62, bold=True)
        self.text(subtitle, 70, 290, 1060, 110, size=40, color=MUTED)

    def cell(self, value, x, y, width, height, style):
        if not (0 <= x <= WIDTH - width and 0 <= y <= HEIGHT - height):
            raise ValueError(f"Cell outside the card: {value}")
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {"id": f"c{len(self.root)}", "value": value, "vertex": "1", "parent": "1", "style": style + ";html=1;"},
        )
        ET.SubElement(
            cell, "mxGeometry", {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "as": "geometry"}
        )
        return cell.get("id")

    def box(self, x, y, width, height, fill, stroke="none", radius=True):
        return self.cell(
            "", x, y, width, height, f"rounded={int(radius)};arcSize=12;fillColor={fill};strokeColor={stroke};strokeWidth=3"
        )

    def text(self, text, x, y, width, height, size=44, color=INK, bold=False, align="left"):
        return self.cell(
            html.escape(text).replace("\n", "<br>"),
            x,
            y,
            width,
            height,
            f"text;whiteSpace=wrap;overflow=hidden;fillColor=none;strokeColor=none;fontFamily=Arial;"
            f"fontSize={size};fontColor={color};fontStyle={int(bold)};align={align};verticalAlign=middle;spacing=0",
        )

    def arrow(self, source, target):
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": f"c{len(self.root)}",
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
                "style": f"edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor={TEAL};strokeWidth=4;endArrow=block;",
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    def footer(self, text):
        self.text(text, 70, 1130, 1060, 190, size=36, color=MUTED)

    def write(self, path):
        ET.indent(self.model)
        path.write_text(ET.tostring(self.model, encoding="unicode") + "\n")


def build(data: dict, output: Path) -> dict:
    if data["sct_version"] != "0.8.2" or len(data["rows"]) != 14:
        raise ValueError("Expected the complete v0.8.2 other-tools comparison")
    output.mkdir(parents=True, exist_ok=True)
    cards, alt = {}, {}
    card = Card("A text-processing\npipeline for datasets", "PII handling is one stage.\nChoose which stages run.")
    previous = None
    for i, label in enumerate(("Detect language", "Repair text", "Handle personal data", "Score & document")):
        y = 425 + 165 * i
        node = card.box(100, y, 1000, 120, "#E9F5F5" if i == 2 else "#F0F4F8", TEAL if i == 2 else "none")
        card.text(f"{i + 1}   {label}", 145, y + 15, 910, 90, size=48, bold=i == 2)
        if previous:
            card.arrow(previous, node)
        previous = node
    card.footer("Text in → cleaned text + signals.\nOptional scoring and data cards.")
    cards["01-pipeline"] = card
    alt["01-pipeline"] = (
        "SqueakyCleanText is a configurable text-processing pipeline. Four stages: detect language, "
        "repair text, handle personal data, then score and document. PII is one stage. "
        "Outputs include cleaned text and signals, with optional scoring and data cards. Stages can be switched off."
    )

    card = Card("Configurable\nPII routing", "SCT's own rules, plus\nsupported models and backends.")
    rules = card.box(70, 430, 480, 480, "#F0F4F8")
    card.text("SCT's rules", 105, 455, 410, 80, size=48, bold=True)
    card.text("Emails\nPhones\nIPs, IBANs, cards\nCustom patterns", 105, 560, 410, 300, size=42)
    models = card.box(650, 430, 480, 480, "#E9F5F5", TEAL)
    card.text("Choose a backend", 680, 455, 420, 80, size=44, bold=True)
    card.text("ONNX / PyTorch\nGLiNER / GLiNER2\nPresidio + GLiNER\n(beta)", 680, 550, 420, 300, size=40)
    result = card.box(200, 1000, 800, 110, "#E9F5F5", TEAL)
    card.text("One configurable PII stage", 225, 1010, 750, 90, size=44, bold=True, align="center")
    card.arrow(rules, result)
    card.arrow(models, result)
    card.footer("Language-based model selection.\nCustom models use supported interfaces.")
    cards["02-pii-routing"] = card
    alt["02-pii-routing"] = (
        "SCT's PII stage combines its own regex rules with a chosen model backend. "
        "Rules cover emails, phones, IPs, IBANs, cards and custom patterns. Supported backend choices "
        "include ONNX, PyTorch, GLiNER, GLiNER2 and beta Presidio plus GLiNER. Language-based "
        "model selection and custom models use documented interfaces. These are configurable "
        "choices, not simultaneous use of all backends or integration of every benchmarked tool."
    )

    rows = data["rows"]
    selected = sorted(
        (r for r in rows if r["shared"]["metrics"]["masked_recall"]["ci95"][1] < 0),
        key=lambda r: r["shared"]["metrics"]["masked_recall"]["difference"],
    )
    if len(selected) != 4:
        raise ValueError("The shared-type claim must be reviewed: expected four higher-recall tools")
    card = Card(
        "Four tools have\nhigher masking recall",
        "Synthetic data: emails, phones, numbers.\nPercentage points above SCT default.",
        True,
    )
    details = []
    for i, row in enumerate(selected):
        metric = row["shared"]["metrics"]["masked_recall"]
        difference = -100 * metric["difference"]
        lo, hi = [-100 * x for x in reversed(metric["ci95"])]
        y = 425 + i * 170
        card.text(NAMES[row["entry"]], 70, y, 680, 65, size=46, bold=True)
        card.text(f"+{difference:.1f} points", 760, y, 370, 65, size=46, color=ORANGE, bold=True, align="right")
        card.box(70, y + 80, 1060, 15, "#EDF0F3", radius=False)
        card.box(70, y + 80, round(1060 * difference / 18), 15, ORANGE, radius=False)
        card.text(f"95% interval: +{lo:.1f} to +{hi:.1f} points", 70, y + 105, 1060, 50, size=36, color=MUTED)
        details.append(
            f"{NAMES[row['entry']]}: {difference:.2f} percentage points above SCT; 95% interval {lo:.2f} to {hi:.2f}."
        )
    card.footer(
        "Selection: 95% advantage interval above zero.\n"
        "Matched docs per pair; subsets vary.\nAny overlap counts, not complete redaction.\n"
        "Unadjusted intervals. Full results: report."
    )
    cards["03-shared-recall"] = card
    alt["03-shared-recall"] = (
        "SCT v0.8.2 default, synthetic datasets, shared EMAIL/PHONE/NUMBER types. "
        + " ".join(details)
        + " The four entries are selected because the paired 95% interval "
        "for their recall advantage is entirely above zero. Each pair uses matched documents "
        "and languages both claim; subsets differ. Bootstrap intervals are unadjusted. "
        "Masking recall counts any overlap, not complete redaction."
    )

    metrics = [r["full"]["metrics"]["masked_recall"] for r in rows]
    higher = sum(m["ci95"][0] > 0 for m in metrics)
    lower = sum(m["ci95"][1] < 0 for m in metrics)
    unclear = len(metrics) - higher - lower
    card = Card(
        "Broader label coverage\nchanges the result",
        "Full scored taxonomy • synthetic data.\nSCT default against 14 other entries.",
        True,
    )
    for i, (count, label) in enumerate(
        ((higher, "Higher\nrecall"), (unclear, "Unclear\ndifference"), (lower, "Lower\nrecall"))
    ):
        x = 70 + i * 365
        card.box(x, 455, 330, 340, "#E9F5F5" if i == 0 else "#F0F4F8")
        card.text(str(count), x + 15, 470, 300, 175, size=140, bold=True, color=TEAL, align="center")
        card.text(label, x + 15, 650, 300, 120, size=42, bold=True, align="center")
    card.text("Counts use paired 95% intervals.\nGLiNER2-Fastino: difference unclear.", 70, 845, 1060, 130, size=42)
    card.text(
        "Unsupported types count as misses.\nThis also measures entity coverage.", 70, 995, 1060, 120, size=42, bold=True
    )
    card.footer(
        "DATE excluded. Matched docs within each pair.\nPair subsets vary. Intervals are unadjusted.\n"
        "Higher recall does not prove complete redaction."
    )
    cards["04-full-recall"] = card
    alt["04-full-recall"] = (
        f"SCT v0.8.2 default, full-taxonomy synthetic comparisons: {higher} paired 95% intervals "
        f"show higher SCT masking recall; {unclear} crosses zero; {lower} shows lower SCT recall. "
        "GLiNER2-Fastino is the unclear comparison. These are unadjusted bootstrap intervals, "
        "not equivalence tests or a universal ranking. Unsupported types count as misses and DATE "
        "is excluded. Each pair matches documents; pair subsets differ. Any overlap counts as recall."
    )

    example = next(r for r in rows if r["entry"] == "pii-tracer")["full"]
    for filename, metric_name, title, subtitle in (
        (
            "05-precision",
            "masked_precision",
            "PII-Tracer has higher\nmasking precision",
            "One paired example • full taxonomy.\nHigher is better.",
        ),
        (
            "06-exposed-text",
            "doc_leak_rate",
            "More recall can still\nleave more text exposed",
            "One paired example • full taxonomy.\nDocuments with exposed text: lower is better.",
        ),
    ):
        metric = example["metrics"][metric_name]
        card = Card(title, subtitle, True)
        for i, (label, key, color) in enumerate((("SCT default", "sct", TEAL), ("PII-Tracer", "other", "#586B7E"))):
            y = 450 + 245 * i
            card.text(label, 70, y, 650, 75, size=48, bold=True)
            card.text(f"{100 * metric[key]:.1f}%", 760, y, 370, 75, size=68, color=color, bold=True, align="right")
            card.box(70, y + 110, 1060, 65, "#EDF0F3", radius=False)
            card.box(70, y + 110, round(1060 * metric[key]), 65, color, radius=False)
        lo, hi = metric["ci95"]
        if metric_name == "masked_precision":
            lo, hi = -hi, -lo
            result = f"PII-Tracer advantage: {-100 * metric['difference']:.1f} points"
            definition = "Precision counts predicted mask spans\nthat overlap labelled mentions."
        else:
            result = f"SCT exposure is {100 * metric['difference']:.1f} points higher"
            definition = "Exposure: at least one labelled letter\nor digit stays visible in a document."
        card.text(result, 70, 955, 1060, 70, size=44, bold=True)
        card.text(f"95% difference interval: {100 * lo:.1f} to {100 * hi:.1f} points", 70, 1030, 1060, 65, size=36)
        card.footer(definition + "\nDataset means; same docs; DATE excluded.\nIntervals have no multiple-test correction.")
        cards[filename] = card
        alt[filename] = (
            f"SCT v0.8.2 default versus PII-Tracer, dataset averages on the same synthetic documents in eight datasets, "
            f"full taxonomy excluding DATE. {metric_name}: SCT {100 * metric['sct']:.2f}%, "
            f"PII-Tracer {100 * metric['other']:.2f}%. {result}. Paired bootstrap 95% difference "
            f"interval {100 * lo:.2f} to {100 * hi:.2f} percentage points, unadjusted. "
            + definition.replace("\n", " ")
            + " This is one illustrative pair, not a ranking."
        )
    for name, card in cards.items():
        card.write(output / f"{name}.drawio")
    if any(len(text) > 1000 for text in alt.values()):
        raise ValueError("An image description exceeds 1,000 characters")
    metadata = {
        "source_sha256": data["source_sha256"],
        "sct_version": data["sct_version"],
        "width": WIDTH,
        "height": HEIGHT,
        "full_counts": [higher, unclear, lower],
        "alt_text": alt,
    }
    (output / "cards.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(extract(args.source), args.output)
