from sct import TextCleaner, TextCleanerConfig
from sct.streaming import BufferedStreamingRedactor, WindowedStreamingRedactor, _masking_steps


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
    assert len(_masking_steps(cleaner, cfg, "text")) == 11
