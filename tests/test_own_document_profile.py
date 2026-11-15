import hashlib
import json
import pytest
from scripts.own_document_profile import prepare, summarize


def fixture(tmp_path):
    root=tmp_path/'root';root.mkdir();(root/'corpus').mkdir()
    source=root/'corpus/nist-sp800-63b-4.pdf';source.write_bytes(b'trusted-fixture')
    original=tmp_path/'original.json';original.write_text(json.dumps({'records':[]}))
    packet={'sha256':'packet','original_export_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'items':[{'id':'n1','group':'nist','source_page':1,'source_passage':None,'origin':'prefilled_synthetic_proposal','answerable':True}]}
    (root/'confirmation-packet.private.json').write_text(json.dumps(packet))
    units=[{'content':'The limit is 12.','source':'fixture','metadata':{},'doc_id':'d'}]
    cache={'documents':[{'filename':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'units':units,'units_sha256':hashlib.sha256(json.dumps(units,sort_keys=True,ensure_ascii=True).encode()).hexdigest()}]}
    (root/'parsed-cache.private.json').write_text(json.dumps(cache))
    record={'id':'n1','finished':True,'human_adjudicated':True,'question_validity':'valid','answer_correctness':'correct','evidence_support':'supported','question':'What is the limit?','answer':'12','aliases':'twelve','support_quote':'The limit is 12.'}
    labels={'packet_sha256':'packet','records':[record],'elapsed_ms':1000}
    confirmation=tmp_path/'confirmation.json';confirmation.write_text(json.dumps(labels))
    return root,original,confirmation,labels


def test_only_explicit_complete_human_decisions_enter_frozen_gold(tmp_path):
    root,original,confirmation,labels=fixture(tmp_path)
    labels['records'][0]['human_adjudicated']=False;confirmation.write_text(json.dumps(labels))
    with pytest.raises(ValueError,match='No eligible'):prepare(root,original,confirmation,tmp_path/'out')
    labels['records'][0]['human_adjudicated']=True;confirmation.write_text(json.dumps(labels))
    result=prepare(root,original,confirmation,tmp_path/'out')
    assert result['pdf_questions']==1 and result['selection_uses_model_outcomes'] is False
    with pytest.raises(FileExistsError):prepare(root,original,confirmation,tmp_path/'out')


def test_changed_export_and_source_are_rejected_before_generation(tmp_path):
    root,original,confirmation,labels=fixture(tmp_path)
    original.write_text('{}')
    with pytest.raises(ValueError,match='Original export changed'):prepare(root,original,confirmation,tmp_path/'out')


def test_requested_denominator_retains_unserved_own_document_questions():
    design={'pdf_questions':3,'requested_tasks':4,'scope':'smoke','cohort_sha256':'h'}
    result=summarize(design,[{'task_id':'one','format':'pdf','status':'completed','exact_match':1,'release_state':'held'}])
    assert result['groups']['pdf']['requested_candidate_em']==1/3
    assert result['groups']['pdf']['unserved']==2
    assert result['groups']['docx']['unserved']==1
    assert result['groups']['pdf']['false_release_risk']=='insufficient evidence'


def test_published_own_document_records_reject_changed_token_accounting(tmp_path):
    import shutil
    from pathlib import Path
    from scripts.verify_own_document_profile import verify
    source = Path(__file__).resolve().parents[1] / 'reports/releases/r4/own_document_t4'
    target = tmp_path / 'published'
    shutil.copytree(source, target)
    assert verify(target)['requested_tasks'] == 19
    records = json.loads((target / 'numeric_records.json').read_text())
    records[0]['tokens'] += 1
    (target / 'numeric_records.json').write_text(json.dumps(records))
    with pytest.raises(AssertionError):
        verify(target)
