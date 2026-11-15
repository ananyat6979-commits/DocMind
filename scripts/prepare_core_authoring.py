"""Private timed human authoring, before model outputs. This script never creates gold labels."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

HTML=r'''<!doctype html><meta charset="utf-8"><title>Private document question authoring</title>
<style>body{font:17px system-ui;max-width:1150px;margin:24px auto;padding:16px;line-height:1.5;color:#17253a;background:#f4f6fa}button,input,textarea,select{font:inherit;padding:8px;box-sizing:border-box}button{margin:8px 8px 8px 0}textarea,input{width:100%;margin:5px 0 12px}label{display:block;font-weight:600}article{padding:22px;background:white;border:1px solid #cbd3df}pre{white-space:pre-wrap;font:inherit}#timer{position:sticky;top:0;background:#162b43;color:white;padding:12px}img{max-width:100%;border:1px solid #ccd}fieldset{border:0;padding:0}aside{background:#eaf0f6;padding:12px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:24px}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<h1>Private document question authoring</h1><p>Write clear questions and their correct short answers using the source shown. No model output exists in this packet. You have at most 19 minutes, may stop early, and are not expected to finish all 30 opportunities. Aim for useful definitions, named values, table entries and some questions needing two sentences. Use an unanswerable question only for the complete five-page project guide, with an absence note; do not infer absence from a NIST excerpt.</p>
<aside>Keep answers concise and include accepted aliases when needed. Copy a supporting passage, then check the page image and source context. Exact text matching checks location only. Mark uncertain or invalid items accordingly. Only your saved decisions count as human review. Unsaved and unfinished opportunities stay unreviewed. Exported labels remain private.</aside>
<div id="timer"></div><button id="start">Start or resume</button><button id="export">Export private questions</button><main hidden id="main"></main>
<script>const packet=__DATA__,key='core-authoring-'+packet.sha256,cap=19*60*1000;let state=JSON.parse(localStorage.getItem(key)||'null')||{schema:'core-human-authoring-v1',packet_sha256:packet.sha256,protocol_sha256:packet.protocol_sha256,index:0,started_at:null,elapsed_ms:0,records:{}};let running=false,last=performance.now();
function persist(){localStorage.setItem(key,JSON.stringify(state))}function record(){let item=packet.items[state.index];return state.records[item.id]||(state.records[item.id]={id:item.id,finished:false,question:'',answer:'',aliases:'',support_quote:'',absence_note:'',question_validity:'',answerable:'',duration_ms:0})}
function el(tag,value,parent){let node=document.createElement(tag);node.textContent=value;parent.append(node);return node}
function render(){let main=document.getElementById('main');main.replaceChildren();let item=packet.items[state.index],r=record();el('h2','Opportunity '+(state.index+1)+' of '+packet.items.length,main);el('p',item.source_label,main);let columns=el('div','',main);columns.className='columns';let source=el('section','',columns);el('h3','Source passage',source);el('pre',item.excerpt,source);let link=el('a','Open full source PDF',source);link.href=item.href;link.target='_blank';let image=el('img','',source);image.src=item.image;image.alt='Source page for visual comparison';let form=el('fieldset','',columns);form.disabled=!running;
for(let [field,title] of [['question','Your question'],['answer','Correct short answer'],['aliases','Other accepted answers separated with |'],['support_quote','Copy the supporting passage'],['absence_note','Absence note for an unanswerable project-guide question']]){let label=el('label',title,form),input=el(field==='aliases'?'input':'textarea','',label);input.value=r[field];input.rows=field==='support_quote'?4:2;input.oninput=()=>{r[field]=input.value;r.finished=false;persist()}}
for(let [field,title,values] of [['question_validity','Is the question clear and well defined?',['','valid','invalid','unclear']],['answerable','Can it be answered from this selected source?',['','yes','no','unclear']]]){let label=el('label',title,form),select=el('select','',label);for(let value of values){let option=el('option',value||'Choose',select);option.value=value}select.value=r[field];select.onchange=()=>{r[field]=select.value;r.finished=false;persist()}}
let save=el('button','Save decision and next',form);save.onclick=()=>{if(!running)return;if(!r.question.trim()||!r.question_validity||!r.answerable){alert('Write a question and choose both labels, or skip unfinished.');return}if(r.answerable==='yes'&&(!r.answer.trim()||!r.support_quote.trim())){alert('An answerable item needs your answer and supporting text.');return}if(r.answerable==='no'&&(item.group!=='project'||!r.absence_note.trim())){alert('Unanswerable labels need a full project-guide check and an absence note.');return}r.finished=true;r.human_adjudicated=true;r.finished_at=new Date().toISOString();if(state.index<packet.items.length-1)state.index++;else running=false;persist();render()};let skip=el('button','Leave unfinished and next',main);skip.disabled=!running;skip.onclick=()=>{state.index=Math.min(packet.items.length-1,state.index+1);persist();render()};let back=el('button','Previous',main);back.disabled=!running;back.onclick=()=>{state.index=Math.max(0,state.index-1);persist();render()}}
function tick(){let now=performance.now();if(running){let elapsed=Math.min(now-last,cap-state.elapsed_ms);state.elapsed_ms+=Math.max(0,elapsed);record().duration_ms+=Math.max(0,elapsed);if(state.elapsed_ms>=cap){running=false;state.cap_reached_at=new Date().toISOString();document.getElementById('main').hidden=true;document.getElementById('start').disabled=true}persist()}last=now;document.getElementById('timer').textContent='Elapsed authoring time '+Math.floor(state.elapsed_ms/60000)+'m '+Math.floor(state.elapsed_ms/1000%60)+'s of 19m'+(running?' (running, including hidden-tab time)':' (stopped)')}
setInterval(tick,250);document.getElementById('start').onclick=()=>{if(state.elapsed_ms>=cap)return;if(!state.started_at)state.started_at=new Date().toISOString();running=true;last=performance.now();document.getElementById('main').hidden=false;persist();render()};document.getElementById('export').onclick=()=>{tick();running=false;document.getElementById('main').hidden=true;persist();let output={...state,exported_at:new Date().toISOString(),records:Object.values(state.records)};let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(output,null,2)],{type:'application/json'}));a.download='docmind-core-human-questions.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};render();tick();</script>'''


def prepare(root):
    output=root/'question-authoring.html'
    if output.exists():raise FileExistsError('Private packets are immutable; use a new directory')
    excerpts=json.loads((root/'guide-excerpts.private.json').read_text(encoding='utf-8'))
    nist=json.loads((root/'nist-pages.private.json').read_text(encoding='utf-8'))
    items=[]
    for index,entry in enumerate(excerpts):
        page=index//3+1
        items.append({'id':'project-'+str(index+1),'group':'project','source_label':'DocMind guide, source passage '+str(index+1),
            'excerpt':entry['text'],'source_provenance':entry['source'],'href':'corpus/docmind-operations-guide.pdf#page='+str(page),
            'image':'pdf-qa/final-'+str(page)+'.png','source_page':page})
    selected=[15,19,23,25,26,28,32,38,43,49,55,61,69,82,94]
    for index in selected:
        text=nist[index]
        # Pre-locate a bounded passage; these are source opportunities, not answers.
        lines=text.splitlines();body='\n'.join(lines[3:])
        excerpt=body[:1300]
        items.append({'id':'nist-'+str(index+1),'group':'nist','source_label':'NIST SP 800-63B-4, PDF page '+str(index+1),
            'excerpt':excerpt,'href':'corpus/nist-sp800-63b-4.pdf#page='+str(index+1),
            'image':'nist-qa/page-'+str(index+1)+'.png','source_page':index+1})
    random.Random(7).shuffle(items)
    protocol_sha=hashlib.sha256(HTML.encode()).hexdigest()
    inputs={'items':items,'protocol_sha256':protocol_sha,'authoring_cap_seconds':1140}
    encoded=json.dumps(inputs,sort_keys=True,ensure_ascii=True)
    packet={**inputs,'sha256':hashlib.sha256(encoded.encode()).hexdigest()}
    output.write_text(HTML.replace('__DATA__',json.dumps(packet,ensure_ascii=True).replace('<',chr(92)+'u003c')),encoding='utf-8')
    (root/'packet.private.json').write_text(json.dumps(packet,indent=2,ensure_ascii=True),encoding='utf-8')
    return {'opportunities':len(items),'packet_sha256':packet['sha256'],'human_labels_created':0,'authoring_cap_seconds':1140}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args()
    print(json.dumps(prepare(args.root)))
