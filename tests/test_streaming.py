import pytest

from sct import TextCleaner, TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor


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
