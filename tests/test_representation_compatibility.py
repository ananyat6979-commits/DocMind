import json
import pytest
from docmind.config import CONFIG
from docmind.db import connect
from docmind.embeddings import compatible_embedding_identity, embedding_identity
from docmind.ingestion.pipeline import ingest
from docmind.retrieval.hybrid import HybridRetriever
from docmind.store import CorpusStore
from tests.conftest import ALPHA


def test_versions_warn_but_representation_changes_fail(caplog):
    old = embedding_identity(); changed = json.loads(old)
    changed['versions']['transformers'] = 'different'
    assert compatible_embedding_identity(old, json.dumps(changed))
    assert 'versions changed' in caplog.text
    for key in ('model', 'revision', 'normalized', 'prefix_policy', 'instruction'):
        modified = dict(changed); modified[key] = 'incompatible'
        assert not compatible_embedding_identity(old, json.dumps(modified))
    assert not compatible_embedding_identity('', old)
    assert not compatible_embedding_identity('unknown', old)


def test_existing_corpus_keeps_recipe_and_ids_after_default_changes(make_file, monkeypatch):
    monkeypatch.setattr(CONFIG.chunker, 'strategy', 'fixed')
    path = make_file('manual.txt', ALPHA); store = CorpusStore()
    assert ingest(path, store=store).chunks_added
    original = store.load_corpus()[0]
    with connect(store.db_path) as conn:
        signature = json.loads(store._get_meta(conn, 'embedding_signature'))
        signature['versions']['transformers'] = 'old-version'
        store._set_meta(conn, 'embedding_signature', json.dumps(signature))
    monkeypatch.setattr(CONFIG.chunker, 'strategy', 'recursive')
    assert ingest(path, store=store).n_cached == 1
    assert [c.chunk_id for c in store.load_corpus()[0]] == [c.chunk_id for c in original]
    assert HybridRetriever(store).search('coolant temperature')
    next_path = make_file('new.txt', ALPHA + ' Additional operating guidance.')
    assert ingest(next_path, store=store).chunks_added
    assert store.get_document(ingest(next_path, store=store).file_ids[0])['lineage']['chunker']['strategy'] == 'fixed'


def test_word_bullets_are_not_a_general_private_use_bypass():
    from docmind.ingestion.quality import extraction_quality
    assert not extraction_quality('\uf0b7 An ordinary list item.\n\uf0a7 Another ordinary item.')['quarantined']
    assert extraction_quality('Broken text ' + '\uf0b7' * 20)['quarantined']
