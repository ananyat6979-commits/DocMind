import importlib.util
import json
import pytest
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

    with pytest.raises(FileExistsError):module.prepare(source,output)
    assert 'main" hidden' in text and 'Math.min(now-last,20*60*1000-state.visible_ms)' in text
    assert 'protocol_sha256' in text


def test_future_packet_caps_time_and_blocks_untimed_labels(tmp_path):
    import re
    import shutil
    import subprocess
    node=shutil.which('node')
    if node is None:pytest.skip('JavaScript runtime unavailable')
    item={'review_id':'test','practice':False,'layout':'A','question':'Test?','answer':'300','docs':[],'retrieved_titles':[]}
    script=re.search(r'<script>(.*)</script>',module.HTML,re.S)[1].replace('__DATA__',json.dumps({'sha256':'synthetic','items':[item]}))
    harness=r'''const vm=require('vm');let clock=0,interval,download;
class Element {constructor(){this.children=[];this.hidden=false;this.disabled=false}append(x){this.children.push(x)}replaceChildren(){this.children=[]}click(){}}
let elements={main:new Element(),start:new Element(),export:new Element(),timer:new Element()};elements.main.hidden=true;
let stored={};let context={document:{visibilityState:'visible',getElementById:id=>elements[id],createElement:()=>new Element(),createTextNode:value=>({textContent:value})},
localStorage:{getItem:key=>stored[key]||null,setItem:(key,value)=>stored[key]=value},performance:{now:()=>clock},setInterval:fn=>interval=fn,setTimeout:()=>{},
Date,alert:()=>{},Blob:class{constructor(parts){download=parts.join('')}},URL:{createObjectURL:()=>'',revokeObjectURL:()=>{}}};
vm.runInNewContext(__SCRIPT__,context);elements.start.onclick();if(elements.main.hidden)throw Error('Review did not start');
clock=1200500;interval();if(!elements.main.hidden||!elements.start.disabled)throw Error('Cap did not stop the review interface');
elements.export.onclick();let result=JSON.parse(download);if(result.visible_ms!==1200000||result.records[0].finished)throw Error('Capped time or unfinished record corrupted');
'''
    target=tmp_path/'timer.js';target.write_text(harness.replace('__SCRIPT__',json.dumps(script)))
    result=subprocess.run([node,str(target)],capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
