from functools import partial

import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor, _masking_steps, _sub_fuzzy_dates
from sct.utils import constants, contact


def _cleaner_failing_on_call(fail_at, normalize_whitespace=True):
    attempts = 0

    def fail_once(text):
        nonlocal attempts
        attempts += 1
        if attempts == fail_at:
            raise ValueError("transient processing failure")
        return text

    return TextCleaner(TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
        check_normalize_whitespace=normalize_whitespace,
        custom_pipeline_steps=(fail_once,),
    ))


@pytest.mark.parametrize("redactor", [BufferedStreamingRedactor, WindowedStreamingRedactor])
@pytest.mark.parametrize("append_after_failure", [False, True])
def test_finish_can_retry_after_processing_fails(redactor, append_after_failure):
    stream = redactor(_cleaner_failing_on_call(1))
    stream.feed("Mail anna@")
    stream.feed("example.com now")

    with pytest.raises(ValueError, match="transient processing failure"):
        stream.finish()

    if append_after_failure:
        assert stream.feed(" please") == ""
    assert stream.finish() == "Mail <EMAIL> now" + (" please" if append_after_failure else "")
    with pytest.raises(RuntimeError, match="finished"):
        stream.finish()
    with pytest.raises(RuntimeError, match="finished"):
        stream.feed("another chunk")


@pytest.mark.parametrize("redactor", [BufferedStreamingRedactor, WindowedStreamingRedactor])
@pytest.mark.parametrize("operation", ["feed", "finish"])
def test_finish_rejects_reentrant_stream_calls(redactor, operation):
    reenter = True

    def callback(_metrics):
        nonlocal reenter
        if reenter:
            reenter = False
            if operation == "feed":
                stream.feed(" extra")
            else:
                stream.finish()

    cleaner = TextCleaner(TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ), metrics_callback=callback)
    stream = redactor(cleaner)
    stream.feed("Mail anna@example.com now")

    with pytest.raises(RuntimeError, match="finished"):
        stream.finish()

    assert stream.finish() == "Mail <EMAIL> now"


@pytest.mark.parametrize("fail_at", [1, 2])
@pytest.mark.parametrize("normalize_whitespace", [False, True])
@pytest.mark.parametrize("retry_feed", [False, True])
def test_failed_window_keeps_text_for_retry(fail_at, normalize_whitespace, retry_feed):
    cleaner = _cleaner_failing_on_call(fail_at, normalize_whitespace=normalize_whitespace)
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)
    text = "alpha  beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron anna@example.com end"
    prefix = "Earlier words that were emitted before the failure. " if fail_at == 2 else ""
    out = stream.feed(prefix)

    with pytest.raises(ValueError, match="transient processing failure"):
        stream.feed(text)

    if retry_feed:
        out += stream.feed("")
    assert out + stream.finish() == prefix + ("alpha beta " if normalize_whitespace else "alpha  beta ") + (
        "gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron <EMAIL> end")


@pytest.mark.parametrize("normalize_whitespace", [False, True])
def test_finish_retry_keeps_output_queued_before_a_failed_window(normalize_whitespace):
    cleaner = _cleaner_failing_on_call(2, normalize_whitespace=normalize_whitespace)
    stream = WindowedStreamingRedactor(cleaner, max_buffer_chars=40, overlap_chars=16)

    with pytest.raises(BufferError):
        stream.feed("Mail anna@example.com before " + "z" * 200)
    with pytest.raises(ValueError, match="transient processing failure"):
        stream.finish()

    assert stream.finish() == "Mail <EMAIL> before " + "z" * 200


def test_streaming_redactors_refuse_reversible_and_keep_placeholder():
    def cleaner(mode):
        return TextCleaner(cfg=TextCleanerConfig(check_ner_process=False, replacement_mode=mode))

    for redactor in (BufferedStreamingRedactor, WindowedStreamingRedactor):
        with pytest.raises(ValueError, match="retain its token map"):
            redactor(cleaner("reversible"))
        stream = redactor(cleaner("placeholder"))
        assert stream.feed("mail a@b.com now") + stream.finish() == "mail <EMAIL> now"


def test_buffered_streaming_redactor_handles_entities_split_across_chunks():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))
    redactor = BufferedStreamingRedactor(cleaner)

    assert redactor.feed("Email maria@exam") == ""
    assert redactor.feed("ple.com now") == ""
    assert redactor.finish() == "Email <EMAIL> now"


def test_windowed_streaming_redactor_retains_split_token():
    cleaner = TextCleaner(cfg=TextCleanerConfig(
        check_ner_process=False,
        check_detect_language=False,
        check_statistical_model_processing=False,
    ))
    redactor = WindowedStreamingRedactor(
        cleaner,
        max_buffer_chars=24,
        overlap_chars=12,
    )

    first = redactor.feed("Prefix words maria@exam")
    second = redactor.feed("ple.com suffix words")
    final = redactor.finish()

    assert first + second + final == "Prefix words <EMAIL> suffix words"


def test_stream_guard_follows_the_masking_step_order():
    # sct.streaming._masking_steps restates these steps: changing them here means changing it too.
    cfg = TextCleanerConfig(check_ner_process=False, check_fuzzy_replace_dates=True, check_replace_identifiers=True,
                            custom_deny_terms={"PROJECT": ("acme",)})
    cleaner = TextCleaner(cfg)
    names = [step.__name__ for step in cleaner._pipeline]
    masking = names[names.index("_replace_user_terms"):names.index("_replace_phone_numbers") + 1]
    assert [name for name in masking if name != "_restore_brackets"] == [
        "_replace_user_terms", "_replace_html", "_replace_urls", "_replace_emails", "_replace_ips", "_replace_dates",
        "_replace_fuzzy_dates", "_replace_iban_cards", "_replace_identifiers", "_replace_phone_numbers",
    ]
    # One guard step for each, and the IBAN/card step reads two patterns.
    steps = _masking_steps(cleaner, cfg, "text")
    owners = [s.func if isinstance(s, partial) else s.__self__ for s in steps]
    assert len(owners) == 11
    assert owners[0] == cleaner._sub_user_rule
    for owner, expected in zip(owners[1:6], (cleaner._html_regex, cleaner._url_regex, constants.EMAIL_REGEX,
                                             contact.IP_VIEW, cleaner._date_view), strict=True):
        assert owner is expected
    assert owners[6] is _sub_fuzzy_dates
    for owner, expected in zip(owners[7:10], (contact.IBAN_VIEW, contact.CARD_VIEW, constants.IDENTIFIER_REGEX),
                               strict=True):
        assert owner is expected
    assert isinstance(owners[10], contact.RejectView)
    assert owners[10].view is cleaner._phone_folded_view
