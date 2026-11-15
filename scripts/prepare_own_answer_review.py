"""Blinded actual-answer and randomly selected claim review; raw labels stay private."""
import argparse
import hashlib
import json
from pathlib import Path
import random

HTML=r'''<!doctype html><meta charset="utf-8"><title>Private actual-answer review</title>
<style>body{font:17px system-ui;max-width:1100px;margin:24px auto;padding:16px;line-height:1.5;background:#f4f6fa;color:#17253a}article{padding:20px;background:white;margin:16px 0}pre{white-space:pre-wrap;font:inherit}button,select,textarea{font:inherit;padding:9px;margin:8px}img{max-width:100%}#clock{position:sticky;top:0;background:#162b43;color:white;padding:12px}label{display:block;margin:12px 0}</style>
<h1>Review actual document answers</h1><p>Use at most 30 recorded running minutes. Decisions are private. Routing, NLI and experimental conditions are hidden. Judge answer correctness against the full source, separately from whether each displayed quote supports its claim. A located quote does not prove support. No choice is preselected. Pause for genuine breaks, not source checking. Complete fewer items if needed.</p><p>Answer correctness includes completeness and whether an absence answer is justified. Claim support concerns only the displayed claim and its actual evidence. No missing claim is silently counted as supported. This single-reviewer subset cannot establish general safety or attribution recall.</p>
<div id="clock"></div><button id="start">Start or resume</button><button id="pause">Pause for a break</button><button id="export">Export private labels</button><main hidden id="main"></main>
<script>const packet=__DATA__,cap=1800000,key='actual-review-'+packet.sha256;let s=JSON.parse(localStorage.getItem(key)||'null')||{schema:'own-actual-review-v1',packet_sha256:packet.sha256,elapsed_ms:0,index:0,records:{},pauses:[]};let run=false,last=Date.now();
function persist(){localStorage.setItem(key,JSON.stringify(s))}function rec(){let p=packet.items[s.index];return s.records[p.id]||(s.records[p.id]={id:p.id,answer_correctness:'',claims:{},note:'',finished:false,duration_ms:0})}function sync(){let now=Date.now();if(run){let dt=Math.min(Math.max(0,now-last),cap-s.elapsed_ms);s.elapsed_ms+=dt;rec().duration_ms+=dt;if(s.elapsed_ms>=cap){run=false;s.cap_reached_at=new Date().toISOString();document.getElementById('main').hidden=true}}last=now;persist();document.getElementById('clock').textContent='Recorded running time '+Math.floor(s.elapsed_ms/60000)+'m '+Math.floor(s.elapsed_ms/1000%60)+'s / 30m'+(run?' (running)':' (paused or stopped)');return run&&s.elapsed_ms<cap}
function el(tag,text,parent){let e=document.createElement(tag);e.textContent=text;parent.append(e);return e}function choose(parent,label,values,current,save){let l=el('label',label,parent),d=el('select','',l);for(let v of ['',...values]){let o=el('option',v||'Choose',d);o.value=v}d.value=current;d.onchange=()=>{if(!sync())return;save(d.value);persist()}}
function render(){let m=document.getElementById('main');m.replaceChildren();let p=packet.items[s.index],r=rec();el('h2','Answer '+(s.index+1)+' of '+packet.items.length,m);el('h3',p.question,m);let a=el('article','',m);el('h3','Actual candidate answer',a);el('pre',p.answer,a);choose(a,'Correctness against the full source',['correct','incorrect','unclear'],r.answer_correctness,v=>{r.answer_correctness=v;r.finished=false});let source=el('article','',m);let link=el('a','Open complete source PDF',source);link.href=p.href;link.target='_blank';let image=el('img','',source);image.src=p.image;image.alt='Source page for correctness checking';el('h3','Actual claim and evidence units',m);
if(!p.claims.length)el('p','No structured claim/evidence unit was returned. No support label is inferred.',m);
for(let c of p.claims){let box=el('article','',m);el('h4','Claim',box);el('pre',c.claim,box);el('h4','Evidence actually supplied',box);el('pre',c.quote||'No usable evidence was supplied.',box);choose(box,'Does this evidence support the complete claim?',['supported','unsupported','unclear'],r.claims[c.id]||'',v=>{r.claims[c.id]=v;r.finished=false})}
let note=el('textarea','',m);note.placeholder='Optional note';note.value=r.note;note.oninput=()=>{if(sync()){r.note=note.value;r.finished=false;persist()}};let save=el('button','Save my decisions and next',m);save.onclick=()=>{if(!sync())return;if(!r.answer_correctness||p.claims.some(c=>!r.claims[c.id])){alert('Choose an answer label and every displayed claim label, or skip unfinished.');return}r.finished=true;r.human_adjudicated=true;r.finished_at=new Date().toISOString();if(s.index<packet.items.length-1)s.index++;else run=false;persist();render()};let skip=el('button','Leave unfinished and next',m);skip.onclick=()=>{if(!sync())return;s.index=Math.min(packet.items.length-1,s.index+1);persist();render()};let back=el('button','Previous',m);back.onclick=()=>{if(!sync())return;s.index=Math.max(0,s.index-1);persist();render()}}
document.getElementById('start').onclick=()=>{sync();if(s.elapsed_ms>=cap)return;if(!s.started_at)s.started_at=new Date().toISOString();run=true;last=Date.now();document.getElementById('main').hidden=false;persist();render()};document.getElementById('pause').onclick=()=>{sync();run=false;s.pauses.push(new Date().toISOString());document.getElementById('main').hidden=true;persist()};document.getElementById('export').onclick=()=>{sync();run=false;document.getElementById('main').hidden=true;persist();let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({...s,exported_at:new Date().toISOString(),records:Object.values(s.records)},null,2)],{type:'application/json'}));a.download='docmind-own-actual-answer-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};setInterval(sync,250);render();sync();</script>'''


def prepare(root,profile):
    records=json.loads((profile/'private/records.json').read_text(encoding='utf-8'))
    gold=json.loads((profile/'private/gold.json').read_text(encoding='utf-8'));lookup={r['task_id']:r for r in gold}
    if len({r['task_id'] for r in records})!=len(records):raise ValueError('Duplicate attempts')
    # Include every attempted primary PDF answer, across routing strata. Time-based
    # completion remains a subset description, never a population estimate.
    selected=[r for r in records if r['format']=='pdf'];random.Random(7).shuffle(selected)
    items=[];private=[]
    for index,r in enumerate(selected):
        g=lookup[r['task_id']];identity='answer-'+str(index+1);claims=[]
        grouped={}
        for e in r.get('evidence',[]):grouped.setdefault(e.get('claim',''),[]).append(e.get('text',e.get('quote','')))
        for i,(claim,quotes) in enumerate(grouped.items()):claims.append({'id':'claim-'+str(i+1),'claim':claim,'quote':'\n\n'.join(quotes)})
        page=g['source_page'];project=g['group']=='project'
        items.append({'id':identity,'question':g['question'],'answer':r.get('answer','No candidate was returned.'),'claims':claims,'href':'corpus/'+g['source_filename']+'#page='+str(page),'image':('pdf-qa/final-'+str(page)+'.png' if project else 'nist-qa/page-'+str(page)+'.png')})
        private.append({'id':identity,'task_id':r['task_id'],'release_state':r.get('release_state'),'status':r['status'],'abstained':r.get('abstained',False),'answerable':g['answerable'],'input_sha256':hashlib.sha256(json.dumps({'answer':r.get('answer'),'claims':claims},sort_keys=True).encode()).hexdigest()})
    if not items:raise ValueError('No actual PDF attempts to review')
    content={'items':items,'cap_seconds':1800,'sampling':'All attempted primary PDF answers in seeded random order; no population extrapolation from timed completion','human_labels_created':0}
    packet={**content,'sha256':hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=True).encode()).hexdigest()}
    output=root/'actual-answer-review.html'
    if output.exists():raise FileExistsError('Private packets are immutable')
    output.write_text(HTML.replace('__DATA__',json.dumps(packet,ensure_ascii=True).replace('<',r'\u003c')),encoding='utf-8')
    (root/'actual-answer-packet.private.json').write_text(json.dumps(packet,indent=2,ensure_ascii=True),encoding='utf-8')
    (root/'actual-answer-linkage.private.json').write_text(json.dumps(private,indent=2),encoding='utf-8')
    return {'answers':len(items),'claims':sum(len(i['claims']) for i in items),'packet_sha256':packet['sha256'],'human_labels_created':0}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('profile',type=Path);a=p.parse_args();print(json.dumps(prepare(a.root,a.profile)))
