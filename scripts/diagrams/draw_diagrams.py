"""Build the README diagrams (.excalidraw + .svg) from one layout spec each.

Usage: python scripts/diagrams/draw_diagrams.py [out_dir]   (default: resources/)
Then render the PNGs with scripts/diagrams/render.sh. Stdlib only.

Every box must be a real SCT feature; check sct/sct.py and sct/config.py when the pipeline changes.
"""
import json
import sys
import zlib
from html import escape
from pathlib import Path

W = 1000  # README shows images at 900 px, so 15 px text stays >= 13.5 px there
FONT = "Helvetica, Arial, sans-serif"
MONO = "Menlo, Consolas, monospace"
INK, MUTED = "#1e1e1e", "#495057"
# (stroke, fill) per family, one meaning per hue in every diagram: blue = repair, teal = language,
# purple = personal data, orange = score, green = output and scale. Light fills, dark text,
# readable on light and dark page themes.
C = {
    "grey": ("#495057", "#f1f3f5"),
    "panel": ("#343a40", "#f8f9fa"),
    "lang": ("#0b7285", "#e3fafc"),
    "fix": ("#1864ab", "#e7f5ff"),
    "pii": ("#6741d9", "#f3f0ff"),
    "score": ("#a65b00", "#fff4e6"),
    "out": ("#2b8a3e", "#ebfbee"),
}


class Diagram:
    def __init__(self, name, height, title):
        self.name, self.h, self.title = name, height, title
        self.rects, self.texts, self.arrows = [], [], []

    def rect(self, x, y, w, h, fam, dashed=False, fill=None):
        stroke, f = C[fam]
        self.rects.append((f"r{len(self.rects)}", x, y, w, h, stroke, fill or f, dashed))

    def text(self, x, y, s, size=16, bold=False, color=INK, anchor="start", mono=False):
        self.texts.append((f"t{len(self.texts)}", x, y, s, size, bold, color, anchor, mono))

    def arrow(self, *pts, dashed=False, head=True):
        self.arrows.append((f"a{len(self.arrows)}", list(pts), dashed, head))

    # A box with a coloured title and optional body lines.
    def card(self, x, y, w, h, fam, title, lines=(), size=16, dashed=False, step=23):
        self.rect(x, y, w, h, fam, dashed)
        self.text(x + 14, y + 30, title, 18, True, C[fam][0])
        for n, line in enumerate(lines):
            self.text(x + 14, y + 60 + n * step, line, size)

    # A one-line step inside a stage: solid = on by default, dashed = off by default.
    def chip(self, x, y, w, label, fam, off=False, h=30, suffix="  (off)"):
        self.rect(x, y, w, h, fam, dashed=off, fill="#ffffff")
        self.text(x + w / 2, y + h / 2 + 5, label + (suffix if off else ""), 15,
                  color=MUTED if off else INK, anchor="middle")

    # A box with a title line and a smaller muted line under it.
    def pair(self, x, y, w, h, fam, title, sub, bold=True, mono=False):
        self.rect(x, y, w, h, fam, fill="#ffffff")
        self.text(x + 12, y + 24, title, 17 if bold else 16, bold, C[fam][0] if bold else INK)
        self.text(x + 12, y + 46, sub, 15, color=MUTED, mono=mono)

    # ---------- checks ----------
    def _box_around(self, x, y):
        """The smallest box holding (x, y), or the canvas."""
        inside = [r for r in self.rects if r[1] <= x <= r[1] + r[3] and r[2] <= y <= r[2] + r[4]]
        return min(inside, key=lambda r: r[3] * r[4])[1:5] if inside else (0, 0, W, self.h)

    def check(self):
        errors = []
        for _, x, y, s, size, bold, _, anchor, mono in self.texts:
            if size * 900 / W < 13:
                errors.append(f"too small: {s!r}")
            # conservative Helvetica / monospace estimate
            w = len(s) * size * (0.6 if mono else 0.59 if bold else 0.54)
            x0 = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
            rx, ry, rw, rh = self._box_around(x, y)
            if not (rx + 4 <= x0 and x0 + w <= rx + rw - 4 and ry <= y - size and y + 4 <= ry + rh):
                errors.append(f"overflows its box by {x0 + w - (rx + rw - 4):.0f} px: {s!r}")
        errors += [f"off canvas: {r}" for r in self.rects
                   if r[1] < 0 or r[2] < 0 or r[1] + r[3] > W or r[2] + r[4] > self.h]
        if errors:
            raise ValueError(f"{self.name}:\n  " + "\n  ".join(errors))

    # ---------- SVG ----------
    def svg(self):
        out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" '
               f'viewBox="0 0 {W} {self.h}" font-family="{FONT}">',
               f'<title>{escape(self.title)}</title>',
               '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
               f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/></marker></defs>',
               f'<rect width="{W}" height="{self.h}" fill="#ffffff"/>']
        for _, x, y, w, h, stroke, fill, dashed in self.rects:
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
                       f'stroke="{stroke}" stroke-width="2"{dash}/>')
        for _, pts, dashed, head in self.arrows:
            d = " ".join(f"{px},{py}" for px, py in pts)
            dash = ' stroke-dasharray="6 4"' if dashed else ""
            end = ' marker-end="url(#ah)"' if head else ""
            out.append(f'<polyline points="{d}" fill="none" stroke="{INK}" stroke-width="2"{dash}{end}/>')
        for _, x, y, s, size, bold, color, anchor, mono in self.texts:
            fam = f' font-family="{MONO}"' if mono else ""
            out.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}"{fam} '
                       f'font-weight="{"bold" if bold else "normal"}" fill="{color}" xml:space="preserve">'
                       f'{escape(s)}</text>')
        out.append("</svg>")
        return "\n".join(out)

    # ---------- Excalidraw ----------
    def excalidraw(self):
        def base(i, typ, x, y, w, h, stroke, fill="transparent", dashed=False):
            return {"id": f"{self.name}-{i}", "type": typ, "x": x, "y": y, "width": w, "height": h,
                    "angle": 0, "strokeColor": stroke, "backgroundColor": fill, "fillStyle": "solid",
                    "strokeWidth": 2, "strokeStyle": "dashed" if dashed else "solid", "roughness": 0,
                    "opacity": 100, "groupIds": [], "frameId": None, "roundness": None,
                    "seed": zlib.crc32(f"{self.name}-{i}".encode()) % 2**31, "version": 1,
                    "versionNonce": 1, "isDeleted": False, "boundElements": None, "updated": 1,
                    "link": None, "locked": False}

        els = []
        for i, x, y, w, h, stroke, fill, dashed in self.rects:
            r = base(i, "rectangle", x, y, w, h, stroke, fill, dashed)
            r["roundness"] = {"type": 3}
            els.append(r)
        for i, pts, dashed, head in self.arrows:
            x0, y0 = pts[0]
            rel = [[px - x0, py - y0] for px, py in pts]
            xs, ys = [p[0] for p in rel], [p[1] for p in rel]
            a = base(i, "arrow", x0, y0, max(xs) - min(xs), max(ys) - min(ys), INK, dashed=dashed)
            a.update({"points": rel, "startBinding": None, "endBinding": None, "startArrowhead": None,
                      "endArrowhead": "arrow" if head else None, "elbowed": False})
            els.append(a)
        for i, x, y, s, size, _, color, anchor, mono in self.texts:
            w = round(len(s) * size * (0.6 if mono else 0.55))
            x = x - w / 2 if anchor == "middle" else x - w if anchor == "end" else x
            e = base(i, "text", x, y - size, w, round(size * 1.25), color)
            e.update({"text": s, "originalText": s, "fontSize": size, "fontFamily": 3 if mono else 2,
                      "textAlign": "left", "verticalAlign": "top", "containerId": None,
                      "lineHeight": 1.25, "autoResize": True})
            els.append(e)
        return {"type": "excalidraw", "version": 2, "source": "https://excalidraw.com", "elements": els,
                "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None}, "files": {}}

    def write(self, out_dir):
        self.check()  # self-check: text fits its box, min size, nothing off canvas
        (out_dir / f"{self.name}.svg").write_text(self.svg() + "\n")
        (out_dir / f"{self.name}.excalidraw").write_text(json.dumps(self.excalidraw(), indent=1) + "\n")


def overview():
    d = Diagram("sct_overview", 600, "What SqueakyCleanText offers")
    top, row_h, gap = 20, 58, 10
    body_h = 52 + 5 * (row_h + gap) + 4
    def row(n):  # one grid for all three columns
        return top + 52 + n * (row_h + gap)

    d.rect(20, top, 220, body_h, "grey")
    d.text(130, top + 32, "In", 16, True, MUTED, "middle")
    for n, (t, s) in enumerate([("Raw text", "a string or a list"), ("JSONL, JSON", "records, nested fields"),
                                ("DataFrame", "a text column"), ("RAG documents", "chunks before indexing")]):
        d.pair(32, row(n), 196, row_h, "grey", t, s, bold=False)

    d.rect(280, top, 400, body_h, "panel")
    d.text(480, top + 32, "SqueakyCleanText", 20, True, INK, "middle")
    for n, (fam, t, s) in enumerate([
            ("lang", "Detect language", "75 languages, with a confidence score"),
            ("fix", "Repair", "mojibake, HTML, Unicode, whitespace"),
            ("pii", "Remove personal data", "regex rules + multilingual NER models"),
            ("score", "Score and document", "sct_* columns, quality score, MinHash")]):
        d.pair(294, row(n), 372, row_h, fam, t, s)
    d.rect(294, row(4), 372, row_h, "out")
    d.text(480, row(4) + 35, "Batched with process_batch, on CPU or GPU", 16, color=C["out"][0], anchor="middle")

    d.rect(720, top, 260, body_h, "grey")
    d.text(850, top + 32, "Out", 16, True, MUTED, "middle")
    for n, (t, s, mono) in enumerate([("Clean text", "lm_text, stat_text", True),
                                      ("Per-record signals", "signals, sct_* columns", True),
                                      ("Quality score, MinHash", "optional, for filtering", False),
                                      ("Data card", "sct-report: JSON + Markdown", False),
                                      ("Reversible map", "optional, store it apart", False)]):
        d.pair(732, row(n), 236, row_h, "grey", t, s, bold=False, mono=mono)

    mid = top + body_h / 2
    d.arrow((240, mid), (280, mid))
    d.arrow((680, mid), (720, mid))

    y = top + body_h + 24
    d.rect(20, y, 960, 140, "grey")
    d.text(36, y + 28, "Call SCT from", 17, True, MUTED)
    for r, labels in enumerate([["process / process_batch", "sct-clean (CLI)", "sct-report (CLI)", "sct-mcp (MCP)"],
                                ["Hugging Face datasets", "DataTrove", "LangChain", "Haystack"]]):
        for n, label in enumerate(labels):
            d.chip(36 + n * 236, y + 44 + r * 46, 220, label, "grey", h=36)
    return d


def flow():
    d = Diagram("sct_flow", 766, "SqueakyCleanText pipeline, in the order the code runs it")
    d.rect(20, 20, 960, 52, "grey")
    d.text(36, 52, "Any entry point (process(), process_batch(), sct-clean, sct-mcp, the adapters) "
                   "runs this pipeline", 16)
    d.arrow((145, 72), (145, 100))

    top, h = 100, 268
    d.card(20, top, 250, h, "lang", "1  Detect language",
           ["Lingua, 75 languages", "on the raw input", "language= pins or limits", "low confidence warns",
            "picks the NER models", "and the Faker locale"], step=36)
    d.arrow((270, top + h / 2), (300, top + h / 2))

    d.card(300, top, 270, h, "fix", "2  Fix the text")
    for n, (label, off) in enumerate([("Unicode repair (ftfy)", False), ("fold styled characters", False),
                                      ("ASCII transliteration", True), ("emoji removal", True),
                                      ("HTML entities and tags", False)]):
        d.chip(314, top + 46 + n * 38, 242, label, "fix", off)
    d.arrow((570, top + h / 2), (600, top + h / 2))

    # Chips read down each column in run order (README "The diagram as text" lists it).
    d.card(600, top, 380, h, "pii", "3  Regex PII and cleanup")
    steps = [("URLs", False), ("emails", False), ("IPs", False), ("dates", False), ("fuzzy dates", True),
             ("IBAN, cards", False), ("ID codes", True), ("phones", False), ("years", False),
             ("numbers", False), ("currency symbols", False), ("symbols, spaces", False)]
    for n, (label, off) in enumerate(steps):
        d.chip(614 + (n // 6) * 182, top + 46 + (n % 6) * 36, 170, label, "pii", off, h=30)

    top2 = top + h + 60
    d.arrow((790, top + h), (790, top + h + 30), (160, top + h + 30), (160, top2))
    d.card(20, top2, 300, h, "pii", "4  NER")
    for n, (label, off) in enumerate([("ONNX models per language", False), ("vote: best model score", False),
                                      ("long text: chunk windows", False), ("propagate masked names", False),
                                      ("snap to word edges", False), ('GLiNER, ner_mode="pii"', True)]):
        d.chip(34, top2 + 46 + n * 36, 272, label, "pii", off)
    d.arrow((320, top2 + h / 2), (350, top2 + h / 2))

    # Placeholder tags are emitted inside steps 3 and 4; only synthetic / reversible swap after NER.
    d.card(350, top2, 310, h, "pii", "5  Replacement mode")
    for n, (label, off) in enumerate([("<PERSON> tags", False), ("synthetic fakes", True),
                                      ("reversible <PERSON_0>", True)]):
        d.chip(364, top2 + 46 + n * 38, 282, label, "pii", off)
    for n, line in enumerate(["Placeholder tags are written in", "steps 3 and 4. Synthetic and",
                              "reversible swap the values in", "after NER."]):
        d.text(364, top2 + 180 + n * 22, line, 15, color=MUTED)
    d.arrow((660, top2 + h / 2), (690, top2 + h / 2))

    d.card(690, top2, 290, h, "out", "6  Output")
    for n, (field, desc) in enumerate([("lm_text", "the cleaned text"), ("stat_text", "lower case, no stopwords"),
                                       ("language", "the detected language"),
                                       ("signals", "counts, warnings, no values"),
                                       ("metadata", "adds regex_counts")]):
        d.text(704, top2 + 60 + n * 42, field, 16, mono=True)
        d.text(704, top2 + 80 + n * 42, desc, 15, color=MUTED)

    y = top2 + h + 20
    d.chip(20, y, 200, "on by default", "grey")
    d.chip(236, y, 200, "off by default", "grey", off=True, suffix="")
    return d


def pii_modes():
    # Real SCT 0.8.1 output: preset_config("training", replacement_mode=...).process(SENTENCE)
    sentence = "Maria Chen lives in Utrecht. Mail her at maria.chen@example.com."
    d = Diagram("sct_pii_modes", 636, "One sentence, three PII modes")
    d.rect(20, 20, 960, 92, "grey")
    d.text(36, 48, "Input", 18, True, C["grey"][0])
    d.text(36, 86, sentence, 17, mono=True)
    d.text(60, 137, "replacement_mode=", 15, mono=True, color=C["pii"][0])

    rows = [("placeholder (default)", "for training and RAG data",
             "<PERSON> lives in <LOCATION>. Mail her at <EMAIL>."),
            ("reversible", "to send text to an LLM and restore the answer",
             "<PERSON_0> lives in <LOCATION_0>. Mail her at <EMAIL_0>."),
            ("synthetic", "fake values, the same fakes for the same document; needs the synthetic extra",
             "Patty Hunter lives in East Bobbyfurt. Mail her at lindabradley@example.net.")]
    y = 150
    for title, note, example in rows:
        rev = title == "reversible"
        h = 218 if rev else 108
        d.rect(80, y, 900, h, "pii")
        d.text(96, y + 30, title, 18, True, C["pii"][0])
        d.text(96, y + 54, note, 15, color="#343a40")
        d.text(96, y + 88, example, 16, mono=True)
        if rev:
            d.rect(96, y + 102, 520, 100, "pii", fill="#ffffff")
            for n, (k, v) in enumerate([("<PERSON_0>", "Maria Chen"), ("<LOCATION_0>", "Utrecht"),
                                        ("<EMAIL_0>", "maria.chen@example.com")]):
                d.text(110, y + 128 + n * 26, f"{k:<13}-> {v}", 16, mono=True)
            d.text(632, y + 128, "map.deanonymize(answer)", 15, mono=True, color=C["pii"][0])
            d.text(632, y + 154, "restores the values; the map", 15)
            d.text(632, y + 180, "is personal data, store it apart.", 15)
        d.arrow((48, y + 38), (80, y + 38))
        last = y + 38
        y += h + 16
    d.arrow((48, 112), (48, last), head=False)  # trunk from the input to the last row
    return d


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[2] / "resources")
    for diagram in (overview(), flow(), pii_modes()):
        diagram.write(out)
        print(out / f"{diagram.name}.svg")
