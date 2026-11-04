"""sct.utils.ner_spans called directly: plain functions over text and span dicts, no model."""

import regex

from sct.utils.ner_spans import (
    _letter_scripts,
    drop_allowed,
    merge_location_gaps,
    merge_overlaps,
    propagate,
    snap_word_edges,
)


def span(start, end, group="PER", score=0.9, text=""):
    return {"entity_group": group, "score": score, "start": start, "end": end,
            "word": text[start:end], "key": f"{start}:{end}"}


def test_letter_scripts_skip_digits_marks_and_modifier_letters():
    assert _letter_scripts("aあ1-ー") == ["LATIN", "HIRAGANA"]
    assert _letter_scripts("ｱ") == ["KATAKANA"]  # read after NFKC


def test_snap_moves_edges_out_to_the_word():
    assert snap_word_edges("Salute now", 0, 5, None, False) == (0, 6)
    assert snap_word_edges("a Salute", 4, 8, None, False) == (2, 8)


def test_snap_leaves_an_edge_next_to_a_non_letter_or_an_empty_span():
    assert snap_word_edges("Ann, 12", 0, 3, None, False) == (0, 3)
    assert snap_word_edges("Ann", 2, 2, None, False) == (2, 2)
    assert snap_word_edges("12ab", 0, 2, None, False) == (0, 2)  # a digit edge does not grow


def test_snap_never_joins_wide_characters():
    assert snap_word_edges("王小明先生", 0, 3, None, False) == (0, 3)


def test_snap_same_script_stops_at_a_script_change():
    text = "Annประเทศ"
    assert snap_word_edges(text, 0, 3, None, True) == (0, 3)
    assert snap_word_edges(text, 0, 3, None, False) == (0, 9)


def test_snap_ratio_caps_how_much_may_be_added():
    assert snap_word_edges("Annabc", 0, 3, 0.5, False) == (0, 3)
    assert snap_word_edges("Annabc", 0, 3, 1.0, False) == (0, 6)


def test_merge_overlaps_takes_the_union_and_the_best_label():
    text = "Maria Chen arrived"
    merged = merge_overlaps(text, [span(6, 10, "LOC", 0.7, text), span(0, 7, "PER", 0.9, text)])
    assert len(merged) == 1
    assert merged[0]["entity_group"] == "PER" and merged[0]["score"] == 0.9
    assert (merged[0]["start"], merged[0]["end"], merged[0]["word"], merged[0]["key"]) == (0, 10, "Maria Chen", "0:10")


def test_merge_overlaps_keeps_touching_spans_and_a_tie_keeps_the_first():
    text = "AnnBob"
    assert [s["end"] for s in merge_overlaps(text, [span(3, 6, text=text), span(0, 3, text=text)])] == [3, 6]
    tied = merge_overlaps(text, [span(0, 4, "PER", 0.5, text), span(2, 6, "ORG", 0.5, text)])
    assert tied[0]["entity_group"] == "PER" and tied[0]["end"] == 6


def test_location_gap_joins_within_the_limit_only():
    text = "Paris, France"
    spans = [span(0, 5, "LOC", 0.8, text), span(7, 13, "LOC", 0.9, text)]
    merged = merge_location_gaps(text, spans, (), 2)
    assert [(s["start"], s["end"], s["word"], s["score"]) for s in merged] == [(0, 13, text, 0.9)]
    assert len(merge_location_gaps(text, spans, (), 1)) == 2


def test_location_gap_does_not_count_placeholder_characters():
    text = "IL <NUMBER>, USA"
    spans = [span(0, 2, "LOC", text=text), span(13, 16, "LOC", text=text)]
    assert len(merge_location_gaps(text, spans, [(3, 11)], 3)) == 1
    assert len(merge_location_gaps(text, spans, (), 3)) == 2


def test_location_gap_stops_at_a_line_break_or_sentence_end():
    for text in ("Paris\nFrance", "Paris. Lyon!", "Paris。Lyon"):
        spans = [span(0, 5, "LOC", text=text), span(len(text) - 4, len(text), "LOC", text=text)]
        assert len(merge_location_gaps(text, spans, (), 20)) == 2, text


def test_location_gap_needs_two_locations():
    text = "Paris Chen"
    assert len(merge_location_gaps(text, [span(0, 5, "LOC", text=text), span(6, 10, "PER", text=text)], (), 20)) == 2


def test_propagate_adds_whole_word_case_sensitive_copies():
    text = "Maria met Maria, Marian and maria."
    result = propagate(text, [span(0, 5, text=text)], (), None)
    assert [(s["start"], s["end"], s["word"], s["key"]) for s in result] == [
        (0, 5, "Maria", "0:5"), (10, 15, "Maria", "10:15")]


def test_propagate_treats_a_combining_mark_as_part_of_the_word():
    text = "राम राम और रामा"
    assert [s["start"] for s in propagate(text, [span(0, 3, text=text)], (), None)] == [0, 4]


def test_propagate_skips_weak_short_letterless_and_non_name_values():
    assert propagate("Bob Bob", [span(0, 3, score=0.4, text="Bob Bob")], (), 0.5)[1:] == []
    assert propagate("A A", [span(0, 1, text="A A")], (), None)[1:] == []
    assert propagate("12 12", [span(0, 2, text="12 12")], (), None)[1:] == []
    assert propagate("Bob Bob", [span(0, 3, "MISC", text="Bob Bob")], (), None)[1:] == []


def test_propagate_leaves_no_overlap_with_spans_or_markers():
    text = "Ann Lee met Ann Lee and <Ann>"
    longest_first = propagate(text, [span(0, 3, text=text), span(0, 7, text=text)], [(24, 29)], None)
    assert [(s["start"], s["end"]) for s in longest_first] == [(0, 3), (0, 7), (12, 19)]


def test_propagate_copies_the_best_score_of_a_value():
    text = "Bob Bob Bob"
    result = propagate(text, [span(0, 3, score=0.6, text=text), span(4, 7, score=0.9, text=text)], (), None)
    assert [s["score"] for s in result] == [0.6, 0.9, 0.9]


def test_drop_allowed_removes_only_spans_inside_an_allowed_term():
    text = "Acme Corp and Bob"
    allow = regex.compile("Acme Corp")
    items = [(0, 4), (0, 9), (5, 12), (14, 17)]
    assert drop_allowed(allow, text, items, lambda item: item) == [(5, 12), (14, 17)]
    assert drop_allowed(None, text, items, lambda item: item) is items
    assert drop_allowed(allow, text, [], lambda item: item) == []
