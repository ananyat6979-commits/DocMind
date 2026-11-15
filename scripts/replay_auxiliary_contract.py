"""Post-hoc deterministic replay of saved outputs; no inference or human labels.

Private inputs remain private. Publish aggregate counts only. The legacy validator
is read from the exact frozen source commit, never reconstructed from outcomes.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
BASE = '500c7e0d7f372007d6ed9af49ac1cfa63428de6f'


def replay(own, archives):
    os.environ.update(DOCMIND_TEST_MODE='1', DOCMIND_ATTRIBUTION='0', DOCMIND_RELEASE_MODE='draft')
    from docmind.agent import reader
    from docmind.models import Chunk, SearchResult
    from quoted_policy_robustness import numeric_target_v2
    original = reader.validate_claims
    source = subprocess.check_output(['git', 'show', BASE + ':docmind/evidence.py'], cwd=ROOT).decode('utf-8')
    legacy = ModuleType('legacy_contract_replay')
    sys.modules[legacy.__name__] = legacy
    exec(compile(source, '<frozen evidence contract>', 'exec'), legacy.__dict__)
    session = SimpleNamespace(calls=0, usage=lambda: {})
    rows = json.loads((own / 'private/records.json').read_text())
    groups = {}

    def compare(name, observations, requested):
        counts = {'legacy': Counter(), 'repaired': Counter()}
        recovered = Counter()
        for raw, sources, question, old, target, kind in observations:
            pair = {}
            for condition, validator in [('legacy', legacy.validate_claims), ('repaired', original)]:
                reader.validate_claims = validator
                response = reader.CitedReader(SimpleNamespace()).assemble(question, sources, raw, time.monotonic(), session)
                pair[condition] = response
                counter = counts[condition]
                counter['attempted'] += 1
                counter['format_failures'] += response.finish_reason == 'format_failure'
                counter['drafts'] += response.release_state == 'draft'
                counter['held'] += response.release_state == 'held'
                counter['invalid_metric_fields'] += bool(response.citation_report.get('quote_validation', {}).get('metric_field_errors'))
                if target:
                    contaminated = numeric_target_v2(response.answer, target) and response.release_state == 'draft' and not response.abstained
                    counter['contaminated_drafts_' + kind] += contaminated
                if condition == 'legacy':
                    if response.release_state != old['release_state']:
                        raise ValueError('Frozen baseline routing did not reproduce')
                    if response.finish_reason == 'format_failure' and old.get('status') != 'format_failure' and old.get('finish_reason') != 'format_failure':
                        raise ValueError('Frozen baseline format status did not reproduce')
            if pair['legacy'].finish_reason == 'format_failure' and pair['repaired'].finish_reason != 'format_failure':
                recovered['recovered_payloads'] += 1
                recovered['recovered_drafts'] += pair['repaired'].release_state == 'draft'
                recovered['recovered_still_held'] += pair['repaired'].release_state == 'held'
                recovered.update('recovered_hold_reason:' + reason for reason in pair['repaired'].hold_reasons)
        groups[name] = {'requested': requested, 'unserved_or_failed_not_replayed': requested - len(observations), **{k: dict(v) for k, v in counts.items()}, **dict(recovered)}

    try:
        observations = []
        for r in rows:
            if len(r['raw_outputs']) != 1:
                raise ValueError('Replay requires exactly one saved completion')
            sources = [SearchResult(Chunk(s['content'], 'cached', s['metadata'].get('filename', 'source.pdf'), i,
                chunk_id=s['chunk_id'], file_id=s['metadata'].get('file_id', 'cached'), metadata=s['metadata']), 1, 'saved')
                for i, s in enumerate(r['sources'])]
            observations.append((r['raw_outputs'][0], sources, r['question'], r, None, r['format']))
        compare('own_documents', observations, len(rows))
        for archive in archives:
            with zipfile.ZipFile(archive) as z:
                planned = {r['id']: r for r in json.loads(z.read('planned.json'))}
                records = json.loads(z.read('private/records.json'))
                raw = json.loads(z.read('private/raw_completions.json'))
            observations = []
            for r in records:
                if r['status'] != 'completed':
                    continue
                c = planned[r['id']]
                texts = ([c['payload'] + '\n' + c['fact']] if c['placement'] == 'before' else
                         [c['fact'] + '\n' + c['payload']] if c['placement'] == 'after' else [c['fact'], c['payload']])
                sources = []
                for i, text in enumerate(texts):
                    identity = hashlib.sha256((c['id'] + ':' + str(i)).encode()).hexdigest()
                    sources.append(SearchResult(Chunk(text, 'synthetic', 'procedure.txt', i, chunk_id=identity, file_id=identity,
                        metadata={'filename': 'procedure.txt', 'language_info': {'language': 'en'}, 'document_language_info': {'language': 'en'}}), 1, 'fixed_synthetic'))
                indices = r['raw_completion_indices']
                if len(indices) != 1 or 'text' not in raw[indices[0]]:
                    raise ValueError('Replay requires one retained completion')
                observations.append((raw[indices[0]]['text'], sources, c['question'], r, c['wrong'], c['condition'] + '_' + c['kind']))
            compare(archive.stem, observations, len(planned))
    finally:
        reader.validate_claims = original
    return {'scope': 'Post-hoc same-output auxiliary scoring-field contract replay, no generation retries or human correctness labels',
        'legacy_source_commit': BASE, 'repaired_evidence_sha256': hashlib.sha256((ROOT / 'docmind/evidence.py').read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'private_input_sha256': {'own_records': hashlib.sha256((own / 'private/records.json').read_bytes()).hexdigest(), **{a.name: hashlib.sha256(a.read_bytes()).hexdigest() for a in archives}},
        'groups': groups, 'new_model_calls': 0, 'new_gpu_seconds': 0,
        'limitations': ['Development regression after inspecting failures, not independent validation',
            'Numeric contamination is a heuristic, not semantic correctness or general attack success',
            'Quote matching does not establish authority, entailment or question relevance',
            'Changed displayed answers require separate human labels; original review packet and frozen results remain unchanged']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--own', type=Path, required=True)
    p.add_argument('--archive', type=Path, action='append', default=[])
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = replay(a.own, a.archive)
    if a.out.exists():
        raise FileExistsError('Do not overwrite a published replay')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['groups']))
