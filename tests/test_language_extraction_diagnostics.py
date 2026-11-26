from docmind.language import detection_sample, language_info
from docmind.ingestion.quality import extraction_quality


def test_language_sample_spans_tail_and_deduplicates_repetition():
    text = 'A first English statement. ' * 300 + 'A distinct middle statement. ' + 'Un texte français distinct. ' * 300
    sample = detection_sample(text)
    assert 'first English' in sample and 'texte français' in sample
    assert sample.count('A first English statement.') == 1
    long = 'a'*5000 + 'A middle section.' + 'z'*5000
    assert 'z'*100 in detection_sample(long)


def test_legacy_font_patterns_need_no_caller_hint_and_are_not_font_conversion():
    quality = extraction_quality('Hkkjr esa foKku vkSj izkS|ksfxdh dk fodkl gqvk gS')
    assert quality['status'] == 'suspected_legacy_font' and quality['quarantined']
    assert extraction_quality('Ordinary English document prose remains readable.')['status'] == 'usable_unicode'
    assert extraction_quality('Unreadable source prose', font_names=['Preeti'])['quarantined']


def test_hindi_repetition_never_grants_english_permission():
    result = language_info('कंप्यूटर प्रणाली में जानकारी सुरक्षित रखी जाती है। '*20)
    assert result['language'] != 'en'


def test_detached_devanagari_signs_are_quarantined():
    examples = (
        ("\u0917\u093e\u0901\u0935", "\u0901"),
        ("\u092a\u0939\u0932\u0941\u0913\u0902", "\u0902"),
        ("\u0926\u0941\u0903\u0916", "\u0903"),
    )
    for word, mark in examples:
        assert not extraction_quality(word, "hi")["quarantined"]
        detached = word.replace(mark, "") + "\n" + mark
        quality = extraction_quality(detached, "hi")
        assert quality["status"] == "uncertain_or_mixed"
        assert "suspicious_combining_sequence" in quality["flags"]
        assert quality["quarantined"]
