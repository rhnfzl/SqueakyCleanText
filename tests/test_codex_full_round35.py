"""Codex 0.8.0 full review, round 35: ids 1 and 2."""

import ast
import os
import subprocess
import sys

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import WindowedStreamingRedactor

# --- 1: a fully qualified host's final dot does not end the URL before its path or query ---


@pytest.mark.parametrize("url", [
    "https://portal.example.com./?email=alice%40example.com",
    "www.example.com./path?u=bob%40x.org",
    "http://10.0.0.1./?e=alice%40b.com",
    "http://localhost./?e=alice%40b.com",
    "https://example.com.:8080/a?e=alice",
    "https://example.com.#frag=alice",
])
def test_url_with_final_host_dot_is_whole(url):
    cleaner = TextCleaner(TextCleanerConfig(check_ner_process=False))
    text = f"Open {url} now."
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=32, overlap_chars=16)
    outputs = [cleaner.process(text).lm_text, cleaner.process_batch([text])[0].lm_text,
               stream.feed(text) + stream.finish()]
    assert outputs == ["Open <URL> now."] * 3


@pytest.mark.parametrize("text", ["Visit https://example.com. Next one.", "Is it https://example.com.? Yes."])
def test_sentence_period_after_url_stays(text):
    out = TextCleaner(TextCleanerConfig(check_ner_process=False)).process(text).lm_text
    assert out == text.replace("https://example.com", "<URL>")


# --- 2: birth-date fakes do not depend on the worker's timezone ---

_DOB = """
from sct.utils.synthetic import SyntheticReplacer
r = SyntheticReplacer(seed=42, locale="en_US", reference_date="2026-01-01")
out = []
for i in range(8):
    r.reset_cache(f"Patient {i} was born on 1990-01-01.")
    out.append(r.get_replacement("DATE_OF_BIRTH", "1990-01-01"))
print(out)
"""


def test_date_of_birth_fakes_same_in_any_timezone():
    def run(tz):
        env = {**os.environ, "TZ": tz}
        return subprocess.run([sys.executable, "-c", _DOB], env=env, capture_output=True, text=True,  # noqa: S603
                              check=True).stdout
    outputs = {run(tz) for tz in ("UTC", "America/Los_Angeles", "Asia/Tokyo")}
    assert len(outputs) == 1
    assert all(len(value) == 10 for value in ast.literal_eval(outputs.pop()))  # dates, not datetimes
