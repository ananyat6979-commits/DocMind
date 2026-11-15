"""Prepare a private blinded timing packet. This never produces human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import random

HTML=r'''<!doctype html><meta charset="utf-8"><title>Private answer review</title><style>
body{font:18px system-ui;max-width:1100px;margin:32px auto;padding:16px;line-height:1.5;background:#f6f7fa;color:#182338}button,select{font:inherit;padding:10px;margin:8px}article{background:white;padding:20px;margin:14px 0;border:1px solid #ccd5e1}pre{white-space:pre-wrap;font:inherit}mark{background:#fff0ac}#timer{position:sticky;top:0;background:#14263b;color:white;padding:12px}label{display:block}</style>
<h1>Private answer review</h1><p>Review for at most 20 minutes. The first two items are practice. Decide whether the question is clear and whether the proposed answer is correct using the supplied sources. Use unclear when you cannot decide. No reference answer is shown. You may stop early and export unfinished work. This is an English public-document timing study, not a Hindi claim-label task.</p>
<div id="timer"></div><button id="start">Start / resume</button><button id="export">Export private labels</button><main id="main" hidden></main><script>
const packet=__DATA__, key='review-cost-'+packet.sha256;let saved=JSON.parse(localStorage.getItem(key)||'null');let state=saved||{schema:2,packet_sha256:packet.sha256,index:0,visible_ms:0,started_at:null,records:{}};let running=false,last=performance.now();
function persist(){localStorage.setItem(key,JSON.stringify(state))}function row(){let item=packet.items[state.index];return state.records[item.review_id]||(state.records[item.review_id]={review_id:item.review_id,visible_ms:0,expansions:0,question_validity:'',answer_correctness:'',finished:false})}
function text(tag,value,parent){let n=document.createElement(tag);n.textContent=value;parent.append(n);return n}
function sourceText(value,highlight,parent){let n=document.createElement('pre');if(!highlight)n.textContent=value;else{let terms=new Set((packet.items[state.index].question+' '+packet.items[state.index].answer).toLowerCase().match(/[a-z]{3,}/g)||[]);terms=new Set([...terms].filter(t=>!['the','and','what','which','with','from','that','this','was','were','does','how'].includes(t)));for(let word of value.split(/(\b[a-zA-Z]{3,}\b)/)){if(terms.has(word.toLowerCase()))text('mark',word,n);else n.append(document.createTextNode(word))}}parent.append(n)}
function render(){let main=document.getElementById('main');main.replaceChildren();let item=packet.items[state.index],r=row();text('h2',(item.practice?'Practice: ':'Review: ')+(state.index+1)+' / '+packet.items.length,main);text('h3',item.question,main);text('p','Proposed answer: '+item.answer,main);
for(let [field,title] of [['question_validity','Is the question clear?'],['answer_correctness','Is the proposed answer correct?']]){let label=text('label',title,main),select=document.createElement('select');for(let value of ['',...(field==='question_validity'?['valid','invalid','unclear']:['correct','incorrect','unclear'])]){let option=text('option',value||'Choose a label',select);option.value=value}select.value=r[field];select.onchange=()=>{r[field]=select.value;r.finished=false;persist()};label.append(select)}
let done=text('button','Save and next',main);done.onclick=()=>{if(!running)return;if(!r.question_validity||!r.answer_correctness){alert('Choose both labels or leave this item unfinished and export.');return}r.finished=true;r.finished_at=new Date().toISOString();if(state.index<packet.items.length-1)state.index++;else running=false;persist();render()};let skip=text('button','Skip for now',main);skip.onclick=()=>{if(state.index<packet.items.length-1)state.index++;persist();render()};let back=text('button','Previous',main);back.onclick=()=>{state.index=Math.max(0,state.index-1);persist();render()};
let docs=item.docs.map((d,i)=>({...d,original:i}));if(item.layout==='B')docs.sort((a,b)=>(item.retrieved_titles.includes(b.title)?1:0)-(item.retrieved_titles.includes(a.title)?1:0)||a.original-b.original);
for(let [i,d] of docs.entries()){let article=document.createElement('article');if(item.layout==='B'&&i>=5){let details=document.createElement('details');text('summary',d.title,details);details.ontoggle=()=>{if(details.open){r.expansions++;persist()}};sourceText(d.text,true,details);article.append(details)}else{text('h3',d.title,article);sourceText(d.text,item.layout==='B',article)}main.append(article)}}
setInterval(()=>{let now=performance.now();if(running&&document.visibilityState==='visible'){let elapsed=Math.min(now-last,20*60*1000-state.visible_ms);state.visible_ms+=elapsed;row().visible_ms+=elapsed;if(state.visible_ms>=20*60*1000){running=false;state.cap_reached_at=new Date().toISOString();document.getElementById('main').hidden=true;document.getElementById('start').disabled=true}persist()}last=now;document.getElementById('timer').textContent='Visible review time: '+Math.floor(state.visible_ms/60000)+'m '+Math.floor(state.visible_ms/1000%60)+'s'+(running?' (running)':' (paused)')},500);
document.getElementById('start').onclick=()=>{if(state.visible_ms>=20*60*1000){alert('The 20-minute cap is reached. Export the current labels.');return}if(!state.started_at)state.started_at=new Date().toISOString();running=true;document.getElementById('main').hidden=false;last=performance.now();persist()};document.getElementById('export').onclick=()=>{running=false;document.getElementById('main').hidden=true;persist();let result={...state,exported_at:new Date().toISOString(),records:Object.values(state.records)};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));a.download='docmind-review-cost-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();
</script>'''


def prepare(directory,output):
    if output.exists() or output.with_suffix('.private-key.json').exists():raise FileExistsError('Existing review packets are immutable; use a new output path')
    records=json.loads((directory/'private/records.json').read_text(encoding='utf-8'))
    held=[r for r in records if r['split']=='heldout'];rng=random.Random(17)
    available=sorted([r for r in held if r.get('token_confidence') is not None],key=lambda r:(r['token_confidence'],r['id']))
    strata=[available[len(available)*i//4:len(available)*(i+1)//4] for i in range(4)]+[[r for r in held if r.get('token_confidence') is None]]
    selected=[];key=[]
    for index,group in enumerate(strata):
        sample=rng.sample(group,min(6,len(group)))
        sample.sort(key=lambda r:sum(map(len,r['contexts'])))
        for pair in range(0,len(sample),2):
            assignment=['A','B'];rng.shuffle(assignment)
            for item,layout in zip(sample[pair:pair+2],assignment):selected.append((item,layout,False,index,len(group),len(sample)))
    rng.shuffle(selected)
    practice=rng.sample([r for r in records if r['split']=='calibration'],2)
    selected=[(r,layout,True,None,None,None) for r,layout in zip(practice,['A','B'])]+selected
    items=[]
    for i,(r,layout,practice,stratum,population,sampled) in enumerate(selected):
        identifier=hashlib.sha256(('private-review:'+r['id']).encode()).hexdigest()[:16]
        items.append({'review_id':identifier,'practice':practice,'layout':layout,'question':r['question'],'answer':r['answer'],
            'retrieved_titles':r['retrieved_titles'],'docs':[{'title':t,'text':s} for t,s in zip(r['titles'],r['contexts'])]})
        key.append({'review_id':identifier,'id':r['id'],'practice':practice,'layout':layout,'stratum':stratum,'population':population,'sampled':sampled,'benchmark_exact_match':r['correct']})
    protocol_sha=hashlib.sha256(HTML.encode()).hexdigest()
    encoded=json.dumps({'items':items,'protocol_sha256':protocol_sha},ensure_ascii=True,separators=(',',':'));sha=hashlib.sha256(encoded.encode()).hexdigest()
    packet={'sha256':sha,'protocol_sha256':protocol_sha,'items':items}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(HTML.replace('__DATA__',json.dumps(packet,ensure_ascii=True).replace('<',chr(92)+'u003c')),encoding='utf-8')
    output.with_suffix('.private-key.json').write_text(json.dumps({'packet_sha256':sha,'protocol_sha256':protocol_sha,'selection_seed':17,'sampling':'Equal six-per-score-quartile sample plus all missing-score items; layouts assigned within evidence-length pairs, practice excluded','records':key},indent=2),encoding='utf-8')
    return {'packet_sha256':sha,'items':len(items),'practice':2,'cap_minutes':20}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();print(json.dumps(prepare(args.input,args.output)))
