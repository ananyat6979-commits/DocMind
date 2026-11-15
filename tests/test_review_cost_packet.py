import importlib.util
import json
from pathlib import Path

spec=importlib.util.spec_from_file_location('review_cost_packet',Path(__file__).resolve().parents[1]/'scripts/prepare_review_cost.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_packet_escapes_hostile_source_and_omits_reference_labels(tmp_path):
    source=tmp_path/'input';(source/'private').mkdir(parents=True)
    records=[{'id':str(i),'split':'calibration' if i<2 else 'heldout','question':'What is the value?','answer':'300','contexts':['</script><img src=x onerror=alert(1)>'],
        'titles':['unsafe title'],'retrieved_titles':['unsafe title'],'token_confidence':i/20,'correct':True,'reference':'PRIVATE_REFERENCE'} for i in range(10)]
    (source/'private/records.json').write_text(json.dumps(records))
    output=tmp_path/'packet.html';module.prepare(source,output)
    text=output.read_text()
    assert text.count('</script>')==1 and '<img' not in text and 'PRIVATE_REFERENCE' not in text
    assert 'benchmark_exact_match' not in text
