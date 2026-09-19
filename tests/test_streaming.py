from sct import TextCleaner, TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor


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
