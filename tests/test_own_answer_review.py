import json
from scripts.prepare_own_answer_review import prepare


def test_packet_contains_real_answers_but_hides_routing_and_automatic_scores(tmp_path):
    root=tmp_path/'review';root.mkdir();profile=tmp_path/'profile';(profile/'private').mkdir(parents=True)
    gold=[{'task_id':'q:pdf','question':'How long?','source_page':16,'group':'nist','source_filename':'nist.pdf','answerable':True}]
    records=[{'task_id':'q:pdf','format':'pdf','status':'completed','answer':'Ten minutes.','release_state':'held','hold_reasons':['invalid_citation'],'nli':{'score':.01},'evidence':[{'claim':'Ten minutes.','text':'It expires in ten minutes.','status':'literal'}]}]
    (profile/'private/gold.json').write_text(json.dumps(gold));(profile/'private/records.json').write_text(json.dumps(records))
    result=prepare(root,profile)
    assert result['answers']==1 and result['claims']==1 and result['human_labels_created']==0
    packet=(root/'actual-answer-packet.private.json').read_text()
    assert 'Ten minutes.' in packet and 'It expires in ten minutes.' in packet
    assert 'release_state' not in packet and 'invalid_citation' not in packet and 'score' not in packet
    linkage=json.loads((root/'actual-answer-linkage.private.json').read_text())
    assert linkage[0]['release_state']=='held'


def test_timed_partial_review_never_invents_labels_or_population_risk():
    from scripts.analyze_own_answer_review import summarize
    import pytest
    packet={'sha256':'p','cap_seconds':1800,'items':[{'id':'a','claims':[{'id':'c'}]},{'id':'b','claims':[]}]}
    linkage=[{'id':'a','release_state':'draft'},{'id':'b','release_state':'held'}]
    labels={'packet_sha256':'p','elapsed_ms':1000,'records':[{'id':'a','finished':True,'human_adjudicated':True,'answer_correctness':'incorrect','claims':{'c':'unsupported'}}]}
    result=summarize(packet,linkage,labels)
    assert result['unfinished_or_missing']==1 and result['incorrect_reviewed_drafts']==1
    assert result['false_release_risk']=='insufficient evidence' and result['reviewed_subset_false_release_fraction'] is None
    labels['records'][0]['claims']={}
    with pytest.raises(ValueError,match='claim label'):summarize(packet,linkage,labels)
