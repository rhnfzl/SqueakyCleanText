"""A GeneralNER built without loading models."""

from sct.utils.ner import GeneralNER


def bare_ner():
    """GeneralNER skipping __init__, with the settings __init__ always assigns at their defaults."""
    ner = object.__new__(GeneralNER)
    ner._ner_batch_size = 8
    ner._ner_chunk_overlap = 0
    ner._ner_ensemble_vote = 'max'
    ner._ner_propagate_masked = True
    ner._ner_propagate_min_score = None
    ner._ner_merge_location_gap = 0
    ner._ner_snap_word_edges = True
    ner._ner_subword_widen = 1.0
    ner._ner_subword_widen_same_script = True
    ner._allow_regex = None
    ner._synthetic_replacer = None
    ner._regex_markers = ()
    ner._engine = None
    ner._tokenizers = {}
    return ner
