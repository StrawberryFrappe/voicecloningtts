from vctts.tts.text_chunker import SentenceChunker, clean_for_speech, split_text


def feed_all(text: str, step: int = 3, **kw) -> list[str]:
    c = SentenceChunker(**kw)
    out = []
    for i in range(0, len(text), step):
        out += c.feed(text[i:i + step])
    return out + c.flush()


def test_first_sentence_is_emitted_early():
    c = SentenceChunker()
    out = c.feed("Sure thing, here we go. And then ")
    assert out == ["Sure thing, here we go."]


def test_streaming_matches_whole_text():
    text = "Hello there, my friend. How are you doing today? I am fine! The end."
    assert " ".join(feed_all(text)) == " ".join(split_text(text))
    assert " ".join(feed_all(text)).replace("  ", " ") == text


def test_short_sentences_are_merged_after_first():
    chunks = split_text("Okay, here it comes. Yes. No. Maybe. This is a longer closing sentence for the test.")
    assert chunks[0] == "Okay, here it comes."
    assert "Yes. No. Maybe." in chunks[1]


def test_decimals_and_abbreviations_do_not_split():
    chunks = feed_all("Dr. Smith measured 3.14 meters exactly today. Then e.g. he left the lab for lunch.", step=1)
    assert chunks[0].startswith("Dr. Smith measured 3.14 meters")


def test_long_run_without_punctuation_is_split():
    text = ("word " * 200).strip()
    chunks = feed_all(text, step=7)
    assert all(len(c) <= 280 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == 200


def test_markdown_is_cleaned():
    assert clean_for_speech("**Bold** and `code` see [docs](http://x.y) 😀") == "Bold and code see docs"
    assert clean_for_speech("# Title\n- item one\n- item two") == "Title item one item two"


def test_newlines_are_boundaries():
    chunks = split_text("First line without period\nSecond line here")
    assert chunks == ["First line without period", "Second line here"]


def test_symbol_only_chunks_dropped():
    assert split_text("***") == []
