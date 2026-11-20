"""Sampler boundaries and explicit legacy-language cache handling."""

from docmind.language import detection_sample


def test_danda_and_double_danda_deduplicate_complete_sentences():
    first = "\u092a\u0939\u0932\u093e \u0935\u093e\u0915\u094d\u092f\u0964"
    second = "\u0926\u0942\u0938\u0930\u093e \u0935\u093e\u0915\u094d\u092f\u0965"
    assert detection_sample((first + " " + second + " ") * 4) == first + " " + second


def test_possessive_apostrophes_are_not_sentence_delimiters():
    text = "The operators\u2019 shared notes. The inspectors\u2019 shared notes."
    assert detection_sample(text) == text


def test_cached_v3_metadata_requires_explicit_reprocessing_to_refresh(
    make_file, monkeypatch
):
    from docmind import language
    from docmind.ingestion.pipeline import ingest
    from docmind.store import CorpusStore

    monkeypatch.setattr(
        language,
        "language_info",
        lambda text: {"language": "en", "version": language.VERSION},
    )
    monkeypatch.setattr(language, "VERSION", "language-v3")
    store = CorpusStore()
    path = make_file(
        "cached-language.txt",
        "The document describes an English safety procedure. Operators inspect the pressure valve before startup.",
    )
    first = ingest(path, store=store)
    file_id = first.file_ids[0]
    old_chunks, _ = store.load_corpus([file_id])
    monkeypatch.setattr(language, "VERSION", "language-v4")
    assert ingest(path, store=store).files[0].status == "cached"
    assert (
        store.get_document(file_id)["lineage"]["preprocessing"]["language"]
        == "language-v3"
    )
    cached_chunks, _ = store.load_corpus([file_id])
    assert [c.chunk_id for c in cached_chunks] == [c.chunk_id for c in old_chunks]
    ingest(path, store=store, rebuild_index=True)
    assert (
        store.get_document(file_id)["lineage"]["preprocessing"]["language"]
        == "language-v4"
    )
    new_chunks, _ = store.load_corpus([file_id])
    assert [c.content for c in new_chunks] == [c.content for c in old_chunks]
    assert not ({c.chunk_id for c in new_chunks} & {c.chunk_id for c in old_chunks})
