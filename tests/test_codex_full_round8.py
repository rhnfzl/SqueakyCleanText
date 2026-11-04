"""Codex 0.8.0 full review, round 8: ids 1, 2."""

import pytest

from sct import TextCleaner
from sct.presets import preset_config
from tests.helpers import FakeRunDetector, word_ner


# --- 1: URLs with private, loopback or IPv6 hosts are replaced whole ---

@pytest.mark.parametrize("host", ["192.168.1.10", "10.0.0.5", "127.0.0.1", "172.16.0.1",
                                  "169.254.1.1", "[2001:db8::1]", "[::1]:8080"])
def test_url_with_ip_host_hides_query_values(host):
    cleaner = TextCleaner(cfg=preset_config("training", check_ner_process=False))
    text = f"see http://{host}/?email=alice%40example.com now"
    assert cleaner.process(text).lm_text == "see <URL> now"


# --- 2: one model's two labels for a widened span are one vote, its best ---

class _TwoLabelDetector(FakeRunDetector):
    """Like a widened "Do" + "##e": PER 0.99 and MISC 0.60 over the same word."""

    def __call__(self, texts, batch_size=1):
        return [[{**span, "entity_group": group, "score": score}
                 for span in found for group, score in (("PER", 0.99), ("MISC", 0.60))]
                for found in super().__call__(texts, batch_size)]


def test_same_model_two_labels_do_not_average_below_threshold():
    ner = word_ner(0, _TwoLabelDetector())
    result = ner.ner_process_detailed("w0 w1 Doe w3", positional_tags=("PER", "MISC"),
                                      ner_confidence_threshold=0.85)
    assert result.text == "w0 w1 <PERSON> w3"
