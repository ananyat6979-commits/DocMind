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
