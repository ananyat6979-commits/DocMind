"""Source matching, not factual entailment or human annotation."""
import json
import pytest
from docmind.evidence import match_quote, untrusted_json, validate_claims
from docmind.models import Chunk, SearchResult


def test_quotes_preserve_original_offsets_across_nfc_and_wraps():
    text = 'A café\n  closes in two seconds.'
    result = match_quote(text, 'cafe\u0301 closes in two seconds.')
    assert result.status == 'normalized'
    assert text[result.start:result.end] == 'café\n  closes in two seconds.'
    assert match_quote('tempera-\nture', 'temperature').status == 'missing'
    assert match_quote('tempera\u00adture', 'temperature').status == 'normalized'
    assert match_quote('repeat repeat', 'repeat').status == 'ambiguous'
    assert match_quote('Seconds', 'seconds').status == 'missing'


def test_unicode_json_preserves_hindi_but_escapes_direction_and_line_controls():
    text = 'कंप्यूटर प्रणाली \u200d' + ''.join(chr(x) for x in (0x85,0x2028,0x2029,0x202e,0x2066)) + '\n"quote"'
    encoded = untrusted_json(text)
    assert 'कंप्यूटर' in encoded and '\u200d' in encoded
    assert all(chr(x) not in encoded for x in (0x85,0x2028,0x2029,0x202e,0x2066))
    assert json.loads(encoded) == text


def test_all_answer_text_must_have_linked_quotes_from_observed_chunks():
    source = SearchResult(Chunk('The valve closes in two seconds.','doc','a; b.txt',0,file_id='f'),1,'dense')
    claim = {'text':'Two seconds.','evidence':[{'chunk_id':source.chunk.chunk_id,'quote':'two seconds'}]}
    result = validate_claims({'claims':[claim]},[source])
    assert result['quotes_matched'] and result['claim_support']=='not_independently_verified'
    assert result['claims'][0]['filename']=='a; b.txt'
    with pytest.raises(ValueError,match='outside'):validate_claims({'answer':'Also safe.','claims':[claim]},[source])
    claim['evidence'][0]['chunk_id']='unseen'
    assert validate_claims({'claims':[claim]},[source])['errors']==['unknown_citation']


def test_invisible_controls_escape_without_changing_json_value():
    value = "Hindi क़ कंप्यूटर " + "\u200b\u2060" + chr(0xE0061)
    encoded = untrusted_json(value)
    assert json.loads(encoded) == value
    assert not any(character in encoded for character in ("\u200b", "\u2060", chr(0xE0061)))


def test_offset_mapping_randomized_combining_text():
    import random
    from docmind.evidence import canonical_text
    rng = random.Random(7)
    alphabet = ['क', 'क़', 'क़', 'ि', 'ं', 'é', 'e\u0301', 'क्ष', '\u200d', '१२', ' ', '\n']
    for _ in range(250):
        text = ''.join(rng.choice(alphabet) for _ in range(30))
        canonical, positions = canonical_text(text)
        assert len(canonical) == len(positions)
        assert all(0 <= start < end <= len(text) for start, end in positions)
        assert all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(positions, positions[1:]))
        if canonical:
            located = match_quote(text, canonical)
            if located.status in ('literal', 'normalized'):
                assert located.text == text[located.start:located.end]
                assert canonical_text(located.text)[0] == canonical
