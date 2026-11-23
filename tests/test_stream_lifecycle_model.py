"""Explore short retry traces through the streaming public APIs."""

from itertools import product

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor

EMAIL = "anna@example.com"
PREFIX = f"Mail {EMAIL} now "
BLOCKED = PREFIX + "x" * 80
SPLITTABLE = BLOCKED.replace("x" * 80, ("y" * 19 + " ") * 4 + "zeta")
ACTIONS = ("retry", "append", "finish", "failed_finish")


class Interrupted(Exception):
    pass


@pytest.mark.parametrize("redactor,first", [
    (BufferedStreamingRedactor, "split_email"),
    (WindowedStreamingRedactor, "split_email"),
    (WindowedStreamingRedactor, "blocked"),
    (WindowedStreamingRedactor, "failed_feed"),
    (WindowedStreamingRedactor, "nested_feed"),
    (WindowedStreamingRedactor, "nested_finish"),
])
def test_stream_lifecycle_model(redactor, first):
    for continuation in product(ACTIONS, repeat=2):
        trace = (first, *continuation, "finish", "retry", "finish")
        try:
            _check_trace(redactor, trace)
        except (Exception, pytest.fail.Exception) as error:
            error.add_note(f"{redactor.__name__}: {trace!r}")
            raise


def _check_trace(redactor, trace):
    fault = ""

    def inject_fault(_metrics):
        nonlocal fault
        current, fault = fault, ""
        if current in ("failed_feed", "failed_finish"):
            raise Interrupted()
        if current == "nested_feed":
            stream.feed("unexpected input")
        elif current == "nested_finish":
            stream.finish()

    config = TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        check_normalize_whitespace=False,
    )
    cleaner = TextCleaner(config, metrics_callback=inject_fault)
    stream = (
        redactor(cleaner, max_buffer_chars=40, overlap_chars=20)
        if redactor is WindowedStreamingRedactor else redactor(cleaner)
    )

    accepted = ""
    delivered = ""
    finished = False
    for action in trace:
        finish = action in ("finish", "failed_finish")
        for chunk in {
            "split_email": ("Mail anna@", "example.com now"),
            "blocked": (BLOCKED,),
            "failed_feed": (SPLITTABLE,),
            "nested_feed": (SPLITTABLE,),
            "nested_finish": (SPLITTABLE,),
            "retry": ("",),
            "append": (" more",),
            "finish": ("",),
            "failed_finish": ("",),
        }[action]:
            fault = action if action.startswith(("failed_", "nested_")) else ""
            if finished:
                with pytest.raises(RuntimeError, match="finished"):
                    stream.finish() if finish else stream.feed(chunk)
                continue

            expected_error = {
                "blocked": BufferError,
                "failed_feed": Interrupted,
                "failed_finish": Interrupted,
                "nested_feed": RuntimeError,
                "nested_finish": RuntimeError,
            }.get(action)
            accepted += chunk
            if expected_error is not None:
                with pytest.raises(expected_error):
                    stream.finish() if finish else stream.feed(chunk)
                continue

            try:
                output = stream.finish() if finish else stream.feed(chunk)
            except BufferError:
                assert not finish and BLOCKED in accepted
                continue

            delivered += output
            expected = accepted.replace(EMAIL, "<EMAIL>")
            assert expected.startswith(delivered)
            if finish:
                assert delivered == expected
                finished = True
