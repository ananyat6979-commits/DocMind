"""Prepare private prefilled proposals; never create human labels automatically."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HTML=r'''<!doctype html><meta charset="utf-8"><title>Private source confirmation</title>
<style>body{font:17px system-ui;max-width:1150px;margin:24px auto;padding:16px;background:#f4f6fa;color:#17253a;line-height:1.5}button,input,textarea,select{font:inherit;padding:8px}textarea,input{width:96%;display:block;margin:7px 0 15px}label{display:block;font-weight:600}pre{white-space:pre-wrap;font:inherit}img{max-width:100%}article,aside{background:white;padding:20px;margin:16px 0}#clock{position:sticky;top:0;background:#162b43;color:white;padding:12px}.cols{display:grid;grid-template-columns:1fr 1fr;gap:20px}@media(max-width:750px){.cols{display:block}}</style>
<h1>Confirm source-based proposals</h1><p>Spend at most 20 recorded running minutes. These questions and answers are prefilled proposals, not human gold. First check four repairs to your export, then as many new source-based synthetic drafts as time permits. Check the source and page image, edit any error, and make your own decisions. Completing fewer is expected. No model answers or routing decisions appear here.</p><aside>For answerable questions, judge question clarity, the proposed answer and the quoted evidence separately. A quote location check cannot establish correctness. For unanswerable guide questions, read the complete five-page guide and confirm the absence note. Pause only for breaks, not while checking the source. Exported labels stay private. The recorded running time is not independently observed active work.</aside>
<div id="clock"></div><button id="start">Start or resume</button><button id="pause">Pause for a break</button><button id="export">Export private decisions</button><main id="main" hidden></main>
<script>const packet=__DATA__,cap=1200000,key='core-confirmation-'+packet.sha256;let s=JSON.parse(localStorage.getItem(key)||'null')||{schema:'core-source-confirmation-v1',packet_sha256:packet.sha256,elapsed_ms:0,index:0,started_at:null,records:{},pauses:[]};let running=false,last=Date.now();
function persist(){localStorage.setItem(key,JSON.stringify(s))}function rec(){let p=packet.items[s.index];return s.records[p.id]||(s.records[p.id]={id:p.id,question:p.question,answer:p.answer,aliases:p.aliases||'',support_quote:p.support_quote,absence_note:p.absence_note||'',question_validity:'',answer_correctness:'',evidence_support:'',finished:false,duration_ms:0})}
function sync(){let now=Date.now();if(running){let dt=Math.min(Math.max(0,now-last),cap-s.elapsed_ms);s.elapsed_ms+=dt;rec().duration_ms+=dt;if(s.elapsed_ms>=cap){running=false;s.cap_reached_at=new Date().toISOString();document.getElementById('main').hidden=true}}last=now;persist();document.getElementById('clock').textContent='Recorded running time '+Math.floor(s.elapsed_ms/60000)+'m '+Math.floor(s.elapsed_ms/1000%60)+'s / 20m'+(running?' (running, including source checks and hidden tabs)':' (paused or stopped)');return running&&s.elapsed_ms<cap}
function el(tag,text,parent){let e=document.createElement(tag);e.textContent=text;parent.append(e);return e}
function render(){let m=document.getElementById('main');m.replaceChildren();let p=packet.items[s.index],r=rec();el('h2',(s.index+1)+' / '+packet.items.length+' - '+p.source_label,m);el('p',p.note||'Synthetic source-based proposal. Your review is still required.',m);let c=el('div','',m);c.className='cols';let src=el('article','',c);el('pre',p.excerpt,src);let a=el('a','Open full source PDF',src);a.href=p.href;a.target='_blank';let img=el('img','',src);img.src=p.image;img.alt='Actual source page';let f=el('article','',c);
for(let [field,label] of [['question','Question'],['answer','Proposed short answer'],['aliases','Accepted aliases, separated by |'],['support_quote','Supporting source text'],['absence_note','Absence note']]){let l=el('label',label,f),t=el('textarea','',l);t.rows=field==='support_quote'?6:2;t.value=r[field];t.oninput=()=>{if(!sync()){render();return}r[field]=t.value;r.finished=false;r.human_adjudicated=false;persist()}}
for(let [field,label,values] of [['question_validity','Is this question clear and well defined?',['valid','invalid','unclear']],['answer_correctness','Is the proposed answer correct for the selected source?',['correct','incorrect','unclear']],['evidence_support','Does the quoted source support the complete answer (or does the full guide establish absence)?',['supported','unsupported','unclear']]]){let l=el('label',label,f),d=el('select','',l);for(let v of ['',...values]){let o=el('option',v||'Choose',d);o.value=v}d.value=r[field];d.onchange=()=>{if(!sync()){render();return}r[field]=d.value;r.finished=false;r.human_adjudicated=false;persist()}}
let save=el('button','Save my decisions and next',m);save.onclick=()=>{if(!sync())return;if(!r.question_validity||!r.answer_correctness||!r.evidence_support){alert('Choose all three labels, or skip unfinished.');return}r.finished=true;r.human_adjudicated=true;r.finished_at=new Date().toISOString();if(s.index<packet.items.length-1)s.index++;else running=false;persist();render()};let skip=el('button','Skip unfinished',m);skip.onclick=()=>{if(!sync())return;s.index=Math.min(packet.items.length-1,s.index+1);persist();render()};let back=el('button','Previous',m);back.onclick=()=>{if(!sync())return;s.index=Math.max(0,s.index-1);persist();render()}}
document.getElementById('start').onclick=()=>{sync();if(s.elapsed_ms>=cap)return;if(!s.started_at)s.started_at=new Date().toISOString();running=true;last=Date.now();document.getElementById('main').hidden=false;persist();render()};document.getElementById('pause').onclick=()=>{sync();running=false;s.pauses.push(new Date().toISOString());document.getElementById('main').hidden=true;persist()};document.getElementById('export').onclick=()=>{sync();running=false;document.getElementById('main').hidden=true;persist();let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify({...s,exported_at:new Date().toISOString(),records:Object.values(s.records)},null,2)],{type:'application/json'}));a.download='docmind-core-source-confirmation.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};setInterval(sync,250);render();sync();</script>'''


def locate(text,first,last):
    # Match only whitespace differences to locate the literal original span.
    def pattern(value):return r"\s+".join(re.escape(w) for w in value.split())
    start=re.search(pattern(first),text);assert start,first
    end=re.search(pattern(last),text[start.start():]);assert end,last
    return text[start.start():start.start()+end.end()]


def prepare(root,export):
    output=root/'source-confirmation.html'
    if output.exists():raise FileExistsError('Use a fresh immutable packet')
    guide=json.loads((root/'guide-excerpts.private.json').read_text(encoding='utf-8'))
    nist=json.loads((root/'nist-pages.private.json').read_text(encoding='utf-8'))
    items=[]
    def add(identifier,group,page,q,a,first=None,last=None,passage=None,note='',aliases='',absence=''):
        text=guide[passage-1]['text'] if group=='project' else nist[page-1]
        quote=locate(text,first,last) if first else ''
        items.append({'id':identifier,'group':group,'source_page':page,'source_passage':passage,'source_label':('Guide passage '+str(passage) if group=='project' else 'NIST PDF page '+str(page)), 'question':q,'answer':a,'aliases':aliases,'support_quote':quote,'absence_note':absence,'answerable':not bool(absence),'excerpt':text,'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'quote_offsets':[text.index(quote),text.index(quote)+len(quote)] if quote else None,'href':'corpus/'+('docmind-operations-guide.pdf' if group=='project' else 'nist-sp800-63b-4.pdf')+'#page='+str(page),'image':('pdf-qa/final-'+str(page)+'.png' if group=='project' else 'nist-qa/page-'+str(page)+'.png'),'note':note,'origin':'prefilled_synthetic_proposal'})
    repair='Repair of your earlier export. Please explicitly confirm or reject this proposed correction.'
    add('repair-project-6','project',2,'In source passage 6, where is the v2 parser equivalence manifest stored?','reports/releases/r2/parser_equivalence_quality_v2.json','The new parser equivalence manifest','parser_equivalence_quality_v2.json.',passage=6,note=repair,aliases='parser_equivalence_quality_v2.json')
    # Bound the full seven-item list by the following subsection heading.
    text=nist[15];start=locate(text,'AAL1 authentication SHALL','2.1.2. Authenticator and Verifier Requirements')
    quote=start[:start.rfind('2.1.2.')].rstrip()
    add('repair-nist-16','nist',16,'Which authenticator types are permitted for AAL1 authentication?','Password; look-up secret; out-of-band device; single-factor OTP; multi-factor OTP; single-factor cryptographic authentication; multi-factor cryptographic authentication.','AAL1 authentication SHALL','2.1.2. Authenticator and Verifier Requirements',note=repair)
    items[-1]['support_quote']=quote;items[-1]['quote_offsets']=[text.index(quote),text.index(quote)+len(quote)]
    add('repair-nist-95','nist',95,'Which RFC provides recommendations for the secure use of TLS and DTLS?','RFC 9325','Sheffer Y, Saint-Andre P, Fossati T (2022)','10.17487/RFC9325',note=repair,aliases='9325')
    add('repair-project-8','project',3,'What computing resources were used to run the experiments?','Available local compute and private free-tier T4 notebooks.','Runs used','T4 notebooks.',passage=8,note=repair,aliases='Local compute and free-tier T4 notebooks')
    specs=[
    (16,'What FIPS 140 validation level is required for AAL1 verifiers operated by or for federal agencies?','Level 1','Cryptography used by verifiers','[FIPS140] Level 1.'),
    (20,'May syncable authenticators be used at AAL3?','No','Since syncable authenticators','used at AAL3.'),
    (20,'What is the maximum overall AAL3 reauthentication timeout?','12 hours','At AAL3, the overall','12 hours.'),
    (20,'What inactivity timeout is recommended at AAL3?','No more than 15 minutes','The inactivity timeout SHOULD','15 minutes.'),
    (24,'What is an activation secret?','A password used locally as an activation factor for a multi-factor authenticator','Activation secret','multi-factor authenticator'),
    (26,'Which Unicode normalization form is recommended before hashing accepted Unicode passwords?','NFC','If Unicode characters are accepted','[UAX15].'),
    (29,'What is the minimum salt length for hashed look-up secrets?','32 bits','The salt value SHALL','stored hashes.'),
    (33,'What is the minimum decimal-digit length of random out-of-band authentication secrets?','Six decimal digits','The verifier SHALL generate','(or equivalent) in length'),
    (39,'What cryptography must multi-factor cryptographic authenticators use at AAL3?','Public-key cryptography','Multi-factor cryptographic authenticators used at AAL3','compromise of the verifier.'),
    (44,'After the initial consecutive biometric failure limit, what minimum delay is imposed before each subsequent attempt?','30 seconds','Once that limit has been reached','before each subsequent attempt'),
    (50,'Can the host-device unlock factor also be used in authentication?','Yes','However, the same activation factor','authentication operation.'),
    (56,'What minimum binding-code length applies when used with the entered identifier?','40 bits','The binding code SHALL','previous step.'),
    (62,'May an expired authenticator still be used for authentication?','No','When an authenticator expires','usable for authentication.'),
    (83,'Which attack is the use of distinct passwords intended to help avoid?','Password stuffing','The use of distinct passwords','other sites.'),
    (95,'In what year was Shannon’s A Mathematical Theory of Communication published?','1948','Shannon CE (1948)','Theory of Communication.')]
    for i,(page,q,a,first,last) in enumerate(specs):add('synthetic-nist-'+str(i+1),'nist',page,q,a,first,last)
    guide_specs=[
    (1,'What prevents an expired ingestion claimant from committing?','Lease checks before work and before commit.','Store transactions','after expiry.'),
    (2,'Does an explicit re-chunk preserve chunk identities?','No, it creates new chunk identities.','An explicit rebuild/re-chunk','not re-embedding.'),
    (3,'Can request bodies choose trusted audit identities?','No','Request bodies cannot','trusted identities.'),
    (5,'Are OCR and legacy-font conversion supported?','No','OCR and legacy-font conversion','unsupported.'),
    (7,'Which endpoint accepts multipart document uploads?','POST /upload','Use','multipart document uploads.'),
    (8,'What confidence level do the reported intervals use?','95 percent','Intervals below','95%;'),
    (10,'Where is the fresh v3 parser equivalence manifest?','reports/releases/r4/parser_equivalence_quality_v3.json','The fresh v3 parser equivalence manifest','parser_equivalence_quality_v3.json.'),
    (11,'Which local database stores corpus and embeddings atomically?','SQLite','The local SQLite profile','selected mode.'),
    (12,'Which PostgreSQL and pgvector versions does the opt-in profile use?','PostgreSQL 17 and pgvector 0.8.7','This opt-in profile uses','pgvector 0.8.7'),
    (12,'What embedding dimension is admitted by the opt-in profile?','384','Only normalized','embeddings are admitted.'),
    (13,'Which retrieval mode explicitly enables the legacy-style path?','hybrid_rerank','DOCMIND_RETRIEVAL_MODE=hybrid_rerank','legacy-style search path.'),
    (15,'Can optional NLI diagnostics grant or deny draft release?','No','Optional NLI diagnostics','draft release.')]
    for i,(passage,q,a,first,last) in enumerate(guide_specs):add('synthetic-guide-'+str(i+1),'project',(passage-1)//3+1,q,a,first,last,passage=passage)
    for i,(q,absence) in enumerate([('What is the monthly paid hosting price?','No monthly paid hosting price is specified in the complete guide.'),('Which OCR engine converts scanned pages?','No OCR engine is specified; the guide states OCR is unsupported.'),('What is the guaranteed production uptime percentage?','No guaranteed production uptime percentage is specified in the complete guide.')]):
        add('synthetic-absence-'+str(i+1),'project',1,q,'Not found in the selected documents',passage=1,absence=absence,note='Read the complete five-page guide before judging absence. This is not a claim about NIST.')
    content={'items':items,'cap_seconds':1200,'original_export_sha256':hashlib.sha256(export.read_bytes()).hexdigest(),'label_origin':'Human decisions only after explicit review; proposals are synthetic','additional_review_allowance_seconds':3600,'total_review_allowance_seconds':9000}
    packet={**content,'sha256':hashlib.sha256(json.dumps(content,sort_keys=True,ensure_ascii=True).encode()).hexdigest()}
    output.write_text(HTML.replace('__DATA__',json.dumps(packet,ensure_ascii=True).replace('<',r'\u003c')),encoding='utf-8')
    (root/'confirmation-packet.private.json').write_text(json.dumps(packet,indent=2,ensure_ascii=True),encoding='utf-8')
    return {'items':len(items),'packet_sha256':packet['sha256'],'human_labels_created':0,'cap_seconds':1200}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('export',type=Path);a=p.parse_args();print(json.dumps(prepare(a.root,a.export)))
