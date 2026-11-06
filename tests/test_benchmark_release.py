from scripts.benchmark_release import find_leaks, predicted_spans
from sct.evaluation import EntitySpan, EvaluationExample


def test_predicted_spans_maps_placeholders_in_order():
    source = "Call 555 1234, then 555 1234 again; Ann knows."
    output = "Call <NUMBER_0>, then <NUMBER_0> again; <PERSON_0> knows. <EMAIL_0>"
    originals = {"<NUMBER_0>": "555 1234", "<PERSON_0>": "Ann", "<EMAIL_0>": "gone@x.io"}

    assert predicted_spans(source, output, originals) == [
        EntitySpan("NUMBER", 5, 13),
        EntitySpan("NUMBER", 20, 28),
        EntitySpan("PERSON", 36, 39),
        EntitySpan("EMAIL", -1, -1),
    ]


def test_find_leaks_reports_full_and_partial_pii_only():
    text = "Bram de Vries paid 2023 from 8.8.8.8"
    example = EvaluationExample(
        id="doc",
        text=text,
        language="nl",
        domain="test",
        expected=(
            EntitySpan("PERSON", 0, 13),
            EntitySpan("YEAR", 19, 23),
            EntitySpan("IP", 29, 36),
        ),
    )

    leaks, partial = find_leaks(example, "<PERSON> Vries paid 2023 from 8.8.8.8")

    assert leaks == [{"id": "doc", "label": "IP", "value": "8.8.8.8"}]
    assert partial == [{"id": "doc", "label": "PERSON", "value": "Bram de Vries"}]


def test_predicted_spans_skip_untouched_earlier_copy():
    assert predicted_spans("Ann met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 8, 11),
    ]


def test_predicted_spans_map_legacy_type_names_to_dataset_label():
    assert predicted_spans("at 8.8.8.8", "at <IP_ADDRESS_0>", {"<IP_ADDRESS_0>": "8.8.8.8"}) == [
        EntitySpan("IP", 3, 10),
    ]
    assert predicted_spans("t 112", "t <PHONE_NUMBER_0>", {"<PHONE_NUMBER_0>": "112"}) == [
        EntitySpan("PHONE", 2, 5),
    ]


def test_find_leaks_catches_split_values():
    text = "Call 9876543210 from 192.168.1.10"
    example = EvaluationExample(
        id="doc", text=text, language="en", domain="test",
        expected=(EntitySpan("PHONE", 5, 15), EntitySpan("IP", 21, 33)),
    )

    leaks, partial = find_leaks(example, "Call 98765<NUMBER> from <NUMBER_0>.1.10")

    assert leaks == []
    assert [item["label"] for item in partial] == ["PHONE", "IP"]


def test_predicted_spans_context_changed_by_cleanup():
    # Isolated-letter removal drops "x", so the output context is not in the source.
    assert predicted_spans("Ann x met Ann", "Ann met <PERSON_0>", {"<PERSON_0>": "Ann"}) == [
        EntitySpan("PERSON", 10, 13),
    ]


def test_run_batch_reports_throughput_keys(tmp_path):
    from scripts.benchmark_release import ROOT, run_batch
    from tests.conftest import TEST_NER_MODELS

    lines = (ROOT / "evaluation" / "release_benchmark.jsonl").read_text(encoding="utf-8").splitlines()
    dataset = tmp_path / "two.jsonl"
    dataset.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    result = run_batch(
        dataset, documents=4, batch_size=2, ner_batch_size=8, intra_op_threads=1,
        repeats=1, long=False, ner_models=TEST_NER_MODELS,
    )

    assert {"settings", "seconds", "docs_per_second", "peak_rss_mb", "output_sha256"} <= set(result)
    assert result["settings"]["documents"] == 4 and len(result["seconds"]) == 1


def test_predicted_spans_map_unicode_folded_value_to_source_offsets():
    # Codex bench round 1 #1: SCT folds full-width text before replacing it.
    source = "Mail Ａｎｎａ＠ｅｘａｍｐｌｅ．ｃｏｍ ok"
    assert predicted_spans(source, "Mail <EMAIL_0> ok", {"<EMAIL_0>": "Anna@example.com"}) == [
        EntitySpan("EMAIL", 5, 21)]
    # A combining accent stays with its letter; a ligature maps to its one source char.
    assert predicted_spans("René met ﬁ", "<PERSON_0> met <X_0>", {"<PERSON_0>": "René", "<X_0>": "fi"}) == [
        EntitySpan("PERSON", 0, 5), EntitySpan("X", 10, 11)]


def test_predicted_spans_place_values_whose_whitespace_sct_collapsed():
    # SCT's normalize_whitespace turns every run (tabs, newlines, NBSP, ideographic space) into one space.
    for source in ("Dear Jan  Kowalski, hi", "Dear Jan\nKowalski, hi", "Dear Jan\t Kowalski, hi",
                   "Dear Jan　Kowalski, hi", "Dear\r\n  Jan Kowalski, hi"):
        got = predicted_spans(source, "Dear <PERSON_0>, hi", {"<PERSON_0>": "Jan Kowalski"})
        assert [source[s.start:s.end].split() for s in got] == [["Jan", "Kowalski"]]


def test_predicted_spans_skip_tokens_sct_dropped_inside_a_value():
    # Isolated letters/symbols and bracket content are removed before NER, so the value has a gap.
    for source, gap in (("Dr Anna B Schulz in Bonn", "B"), ("Ask * Tom Hardy * now", None),
                        ("Pan Jan [przyp. red.] Kowalski tu", "[przyp. red.]"), ("Hans {x} Meier, ok", "{x}")):
        value = "Anna Schulz" if "Anna" in source else "Tom Hardy" if "Tom" in source else (
            "Jan Kowalski" if "Jan" in source else "Hans Meier")
        (span,) = predicted_spans(source, "x <PERSON_0> y", {"<PERSON_0>": value})
        placed = source[span.start:span.end]
        assert placed.split()[0] == value.split()[0] and placed.split()[-1] == value.split()[-1]
        assert gap is None or gap in placed
    # The exact copy after the cursor still wins over a gapped one, and a real word is never skipped.
    assert predicted_spans("Ann B Lee met Ann Lee", "<PERSON_0> met <PERSON_1>",
                           {"<PERSON_0>": "Ann Lee", "<PERSON_1>": "Ann Lee"}) == [
        EntitySpan("PERSON", 0, 9), EntitySpan("PERSON", 14, 21)]
    assert predicted_spans("Ann Bob Lee", "<PERSON_0>", {"<PERSON_0>": "Ann Lee"}) == [EntitySpan("PERSON", -1, -1)]


def test_predicted_spans_mirror_sct_cleanup_exactly():
    from sct.config import TextCleanerConfig

    def placed(source, value, output="x <PERSON_0> y", cfg=None):
        (span,) = predicted_spans(source, output, {"<PERSON_0>": value}, cfg=cfg)
        return source[span.start:span.end] if span.start >= 0 else None

    # Only what SCT removes is skipped: it keeps "A", non-ASCII letters, CJK and "&".
    for source, value in (("Ann A Lee", "Ann Lee"), ("Ann ł Lee", "Ann Lee"),
                          ("李 明 华", "李 华"), ("Smith & Jones", "Smith Jones")):
        assert placed(source, value) is None
    # Glued bracket content, and a quote dropped inside a token after the symbol step.
    assert placed("Yesterday John[1] Smith came", "John Smith") == "John[1] Smith"
    assert placed("Yesterday Anna (*) Smith came", "Anna () Smith") == "Anna (*) Smith"
    # The steps follow the cleaner's config.
    assert placed("Dr Ann B Lee", "Ann Lee", cfg=TextCleanerConfig(check_remove_isolated_letters=False)) is None
    # A regex value taken before the cleanup is still whole in the uncleaned view.
    assert predicted_spans("Call 06 - 1234 now", "Call <PHONE_0> now", {"<PHONE_0>": "06 - 1234"}) == [
        EntitySpan("PHONE", 5, 14)]
    # Edge whitespace is not part of a value, and a blank value is never placed.
    assert placed("Jan  B , ok", "Jan ", "<PERSON_0>, ok") == "Jan"
    assert placed("Hi there Bob", " ", "Hi there <PERSON_0>") is None


def test_predicted_spans_never_credit_text_the_output_kept():
    # SCT masked the later, differently written mention: the unmasked "Ann B Lee" is a leak, not a hit.
    assert predicted_spans("Ann B Lee wrote. Later ANN LEE came", "Ann Lee wrote. Later <PERSON_0> came",
                           {"<PERSON_0>": "Ann Lee"}) == [EntitySpan("PERSON", -1, -1)]


def test_predicted_spans_expand_a_marker_a_wider_ner_span_took_in():
    # A NER span that cut "<PHONE>" took it whole; its own placeholder is gone from the output.
    source = "Write to Jan Kowalski at 0612345678 today."
    got = predicted_spans(source, "Write to <PERSON_0> today.",
                          {"<PHONE_0>": "0612345678", "<PERSON_0>": "Jan Kowalski at <PHONE>"})
    assert [source[s.start:s.end] for s in got] == ["Jan Kowalski at 0612345678"]


def _in_order(source, values):
    """Each value's span, found in *source* after the previous one: where a correct placement puts it."""
    spans, at = [], 0
    for value in values:
        at = source.index(value, at)
        spans.append((at, at + len(value)))
        at += len(value)
    return spans


# Round 2 RedactionBench: on long docs one span credited to a later copy pushed the cursor ahead, and
# every later span was lost. Each test below is a pattern seen there, in hand-written text.
ROWS = "".join(f" row{i} <td>cell{i}</td>" for i in range(6))


def test_predicted_spans_literal_upper_case_tag_is_not_a_marker():
    # A literal "<HTML>" tag is a value, not a regex step's marker, even when an HTML entry was swallowed.
    source = "<HTML> <HEAD> <b>Title</b> </HEAD> <BODY> intro <i>note</i>" + ROWS
    originals = {"<HTML_0>": "<HTML>", "<HTML_1>": "<HEAD>", "<HTML_2>": "<b>", "<HTML_3>": "</b>",
                 "<HTML_4>": "</HEAD>", "<HTML_5>": "<BODY>", "<HTML_9>": "<i>"}
    output = "<HTML_0> <HTML_1> <HTML_2>Title<HTML_3> <HTML_4> <HTML_5> intro <i>note</i>" + ROWS
    got = predicted_spans(source, output, originals)
    assert [(s.start, s.end) for s in got] == _in_order(source, [originals[f"<HTML_{i}>"] for i in range(6)])


def test_predicted_spans_context_may_start_on_the_run_holding_the_cursor():
    # The value ends in symbols the cleanup drops, so the cursor sits inside a whitespace run the
    # cleaned view keeps as an earlier space; the one-space context must still match that run.
    source = 'A <img src="a.png" style="w:50%" />\n   </td> </tr> x </td> </tr>' + ROWS
    values = ['<img src="a.png" style="w:50%" />', "</td>", "</tr>", "</td>", "</tr>"]
    output = "A <HTML_0> <HTML_1> <HTML_2> x <HTML_3> <HTML_4>" + ROWS
    got = predicted_spans(source, output, {f"<HTML_{i}>": v for i, v in enumerate(values)})
    assert [(s.start, s.end) for s in got] == _in_order(source, values)


def test_predicted_spans_context_kept_only_in_the_uncleaned_view():
    # SCT kept bracket content the cleanup mirror drops: the context is only in the uncleaned view.
    source = 'if [ -f "$HOME/a.cfg" ]; then\n  echo "$USER" > log\nfi\n' + "".join(
        f'run step{i} "$ARG{i}"\n' for i in range(6))
    output = 'if [ -f "<CUR_0>HOME/a.cfg" ]; then echo "<CUR_1>USER" > log fi ' + "".join(
        f'run step{i} "<CUR_{i + 2}>ARG{i}" ' for i in range(6))
    got = predicted_spans(source, output, {f"<CUR_{i}>": "$" for i in range(8)})
    assert [(s.start, s.end) for s in got] == _in_order(source, ["$"] * 8)


def test_predicted_spans_reanchor_after_a_span_credited_to_a_later_copy():
    # A value SCT rewrote ("K1" -> "k1") is credited to a later copy and moves the cursor past the next
    # values: they are found again right after their whole context.
    source = "id 77 key K1. tel 555 1234 tel 555 9999 id 88 key k1. url"
    output = "id 77 key <KEY_0>. tel <PHONE_0> tel <PHONE_1> id 88 key <KEY_1>. url"
    got = predicted_spans(source, output, {"<KEY_0>": "k1", "<PHONE_0>": "555 1234", "<PHONE_1>": "555 9999",
                                           "<KEY_1>": "k1"})
    far = source.index("k1")
    assert [(s.start, s.end) for s in got] == [(far, far + 2), *_in_order(source, ["555 1234", "555 9999"]),
                                               (far, far + 2)]
    # Only a context with a letter or digit re-anchors: " " before an unmasked copy is not evidence.
    assert predicted_spans("say Bob. Al Ann", "say Bob. <P_0> <P_1>", {"<P_0>": "Al", "<P_1>": "Bob"}) == [
        EntitySpan("P", 9, 11), EntitySpan("P", -1, -1)]


def test_predicted_spans_marker_reading_wins_over_a_later_literal_copy():
    # "<EMAIL>" in a value is a swallowed marker at the true place; a later template line keeps it literally.
    source = "Write Ann at ann@x.io now. Call 555 1234 today. Template: Write Ann at <EMAIL> now."
    output = "Write <PERSON_0> now. Call <PHONE_0> today. Template: Write Ann at <EMAIL> now."
    got = predicted_spans(source, output, {"<EMAIL_0>": "ann@x.io", "<PERSON_0>": "Ann at <EMAIL>",
                                           "<PHONE_0>": "555 1234"})
    assert [(s.start, s.end) for s in got] == _in_order(source, ["Ann at ann@x.io", "555 1234"])


def test_predicted_spans_reanchor_never_takes_a_copy_the_output_shows():
    # The value is not literal at its true place; the earlier copy with the same context stays unmasked.
    assert predicted_spans("Bob wrote to Ann. Cy wrote to ANN.", "Bob wrote to Ann. <P_0> wrote to <P_1>.",
                           {"<P_0>": "Cy", "<P_1>": "Ann"})[1] == EntitySpan("P", -1, -1)


def test_predicted_spans_fallback_window_ends_where_the_whole_context_starts():
    # Isolated ";" break the cleaned-view context match; the uncleaned view holds all of it, so earlier
    # unmasked rows are outside the window.
    rows = "".join(f"row {i} owner Ann Lee ; " for i in range(5))
    source = rows + "row 5 owner ANN LEE ; row 6 owner Bob Ray ; "
    got = predicted_spans(source, rows + "row 5 owner <P_0> ; row 6 owner <P_1> ; ",
                          {"<P_0>": "Ann Lee", "<P_1>": "Bob Ray"})
    assert [(s.start, s.end) for s in got] == [(-1, -1), _in_order(source, ["Bob Ray"])[0]]


def test_predicted_spans_uncleaned_context_keeps_an_earlier_cleaned_view_hit():
    # The context is whole only in the uncleaned view; the value is whole only in the cleaned one there.
    source = "id=[k]Ann B Lee wrote. later xy]Ann Lee left"
    got = predicted_spans(source, "id=[k]<P_0> wrote. later xy]Ann Lee left", {"<P_0>": "Ann Lee"})
    assert [(s.start, s.end) for s in got] == [_in_order(source, ["Ann B Lee"])[0]]
