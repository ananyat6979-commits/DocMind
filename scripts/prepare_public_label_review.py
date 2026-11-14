"""Create a private blinded Hindi public-label review packet, never human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from docmind.evaluation.release import digest

REVISION="b8dd5d7af51114dbda02c0e3f6133f332186418e"


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,default=ROOT/"reports/release_review");args=parser.parse_args()
    import pyarrow.parquet as pq
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.out/"xnli-hi-test.parquet"
    if not source.exists():urllib.request.urlretrieve("https://huggingface.co/datasets/facebook/xnli/resolve/"+REVISION+"/hi/test-00000-of-00001.parquet",source)
    rows=pq.read_table(source).to_pylist();order=list(range(len(rows)));random.Random(7).shuffle(order)
    selected=order[:200];random.Random(71).shuffle(selected);selected=selected[:15]
    tasks=[];key=[]
    for index in selected:
        row=rows[index];review_id=digest(["public-label-hi",REVISION,index])[:20]
        tasks.append({"review_id":review_id,"premise":row["premise"],"hypothesis":row["hypothesis"],"input_sha256":digest([row["premise"],row["hypothesis"]])})
        key.append({"review_id":review_id,"source_index":index,"original_label":row["label"],"premise_sha256":hashlib.sha256(row["premise"].encode()).hexdigest()})
    (args.out/"public-label-private-key.json").write_text(json.dumps({"revision":REVISION,"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"sample_size":15,"tasks":key},indent=2),encoding="utf-8")
    encoded=json.dumps(tasks,ensure_ascii=True).replace("<","\\u003c")
    html=HTML.replace("TASK_DATA",encoded)
    (args.out/"hindi-public-label-review.html").write_text(html,encoding="utf-8")
    print("Private blinded review packet:",args.out/"hindi-public-label-review.html")


HTML="""<!doctype html><meta charset="utf-8"><title>Hindi public-label review</title>
<style>body{font:18px/1.7 system-ui,sans-serif;max-width:850px;margin:32px auto;padding:0 20px;color:#17212d}section{border:1px solid #ccd4df;padding:22px;margin:24px 0;border-radius:8px}label{display:block}select,input,button{font:inherit;padding:8px;margin:6px 0}p{white-space:pre-wrap}small{color:#536174}button{cursor:pointer}</style>
<h1>Hindi public-label review</h1><p>Review these public NLI sentence pairs yourself. Decide whether the hypothesis follows from the premise, contradicts it, or is neither established nor contradicted. Mark ambiguous or faulty items unclear. Original labels and model scores are hidden. This is a component-label audit, not PDF attribution validation.</p>
<p>Main session: at most 25 minutes. A separate optional consistency session may use the remaining five minutes later. Unfinished items remain unlabeled.</p>
<button id="start">Start review timer</button><button id="pause">Pause timer</button><strong id="clock"></strong>
<div id="tasks"></div><label><input type="checkbox" id="human">I completed these labels myself.</label><button id="export">Save review labels</button><p id="message"></p>
<script>
const tasks=TASK_DATA;
let used=0,started=null,last=performance.now(),active=null;
const labels=Object.fromEntries(tasks.map(t=>[t.review_id,{review_id:t.review_id,input_sha256:t.input_sha256,label:null,duration_seconds:0,finished:false,note:''}]));
function tick(){const now=performance.now();if(started!==null){let delta=(now-last)/1000;const remaining=Math.max(0,1500-used);delta=Math.min(delta,remaining);used+=delta;if(active)labels[active].duration_seconds+=delta;if(used>=1500)started=null;}last=now;document.getElementById('clock').textContent=' Active seconds: '+Math.floor(used)+' / 1500';}
setInterval(tick,500);
document.getElementById('start').onclick=()=>{tick();if(used<1500)started=performance.now();};
document.getElementById('pause').onclick=()=>{tick();started=null;};
for(const [i,task] of tasks.entries()){
 const section=document.createElement('section');const heading=document.createElement('h2');heading.textContent='Pair '+(i+1);section.append(heading);
 for(const field of ['premise','hypothesis']){const label=document.createElement('strong');label.textContent=field==='premise'?'Premise':'Hypothesis';section.append(label);const p=document.createElement('p');p.textContent=task[field];section.append(p);}
 const select=document.createElement('select');for(const [value,text] of [['','Unreviewed'],['0','Follows from the premise'],['1','Neither established nor contradicted'],['2','Contradicts the premise'],['unclear','Unclear or faulty item']]){const option=document.createElement('option');option.value=value;option.textContent=text;select.append(option);}
 const note=document.createElement('input');note.placeholder='Optional note';
 section.onfocusin=()=>{tick();active=task.review_id;};
 select.onchange=()=>{tick();if(started===null){select.value=labels[task.review_id].label===null?'':String(labels[task.review_id].label);document.getElementById('message').textContent='Start the timer before labeling. The session stops at the time cap.';return;}labels[task.review_id].label=select.value===''?null:(select.value==='unclear'?'unclear':Number(select.value));labels[task.review_id].finished=select.value!=='';};
 note.onchange=()=>labels[task.review_id].note=note.value;section.append(select,note);document.getElementById('tasks').append(section);
}
document.getElementById('export').onclick=()=>{tick();if(!document.getElementById('human').checked){document.getElementById('message').textContent='Confirm personal review before saving human labels.';return;}const payload={schema:'public-label-review-v1',dataset:'facebook/xnli',language:'hi',independent_review:true,active_seconds:used,main_budget_seconds:1500,optional_consistency_budget_seconds:300,consistency:'unmeasured',records:Object.values(labels)};const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});const anchor=document.createElement('a');anchor.href=URL.createObjectURL(blob);anchor.download='docmind-hindi-label-review.json';anchor.click();URL.revokeObjectURL(anchor.href);};
</script>"""

if __name__=="__main__":main()
