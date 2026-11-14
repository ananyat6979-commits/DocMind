"""Prepare a private source-versus-extraction packet, without assigning human labels."""
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import random


def safe_json(value):
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def prepare(spec, output):
    import pypdfium2 as pdfium
    if output.exists():
        raise FileExistsError("Review packets are immutable")
    if not 1 <= len(spec) <= 20:
        raise ValueError("Use 1-20 source-quality review pages")
    items = []
    for row in spec:
        pdf = Path(row["pdf"])
        pages = json.loads(Path(row["pages_cache"]).read_text(encoding="utf-8"))
        page = next(p for p in pages if p["page"] == row["page"])
        pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
        text_hash = hashlib.sha256(page["text"].encode()).hexdigest()
        identity = hashlib.sha256((pdf_hash+":"+str(row["page"])+":"+text_hash).encode()).hexdigest()
        with pdfium.PdfDocument(pdf) as document:
            rendered_page = document[row["page"]-1]
            bitmap = rendered_page.render(scale=1)
            stream = io.BytesIO()
            bitmap.to_pil().save(stream, format="PNG")
            bitmap.close()
            rendered_page.close()
        items.append({"review_id": identity, "pdf_sha256": pdf_hash,
            "page": row["page"], "filename": pdf.name, "text_sha256": text_hash,
            "text": page["text"], "image": base64.b64encode(stream.getvalue()).decode()})
    random.Random(7).shuffle(items)
    panels = []
    for index, row in enumerate(items):
        panels.append('<section class="item" id="item-'+str(index)+'"><h2>'+html.escape(row["filename"])+
            ' | PDF page '+str(row["page"])+'</h2><div class="columns"><img alt="Original PDF page" src="data:image/png;base64,'+
            row["image"]+'"><pre>'+html.escape(row["text"])+
            '</pre></div><label>Extraction quality <select id="label-'+str(index)+'">'+
            '<option value="">Unreviewed</option><option value="faithful">Readable and faithful</option>'+
            '<option value="damaged">Readable but damaged or incomplete</option><option value="unusable">Unusable</option>'+
            '<option value="uncertain">Uncertain</option></select></label><p><label>Notes: broken words, missing symbols, or usable passages<br>'+
            '<textarea id="notes-'+str(index)+'" rows="3"></textarea></label></p></section>')
    metadata = [{k: v for k, v in r.items() if k not in ("text", "image")} for r in items]
    document = """<!doctype html><html lang="en"><meta charset="utf-8"><title>Hindi PDF extraction review</title>
<style>body{font:18px system-ui;max-width:1500px;margin:24px auto;padding:12px}button,select,textarea{font:inherit}button{padding:10px;margin:8px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}img{width:100%}pre{white-space:pre-wrap;font:19px/1.7 system-ui;background:#f2f4f7;padding:16px;max-height:1000px;overflow:auto}.item{display:none}textarea{width:95%}.toolbar{position:sticky;top:0;background:white;padding:8px;border-bottom:1px solid #999}@media(max-width:800px){.columns{grid-template-columns:1fr}}</style>
<h1>Hindi PDF extraction review</h1><p>Compare the rendered page with extracted text. These are source-quality checks, not model answers or claim-support labels. Do not repair text here. Record uncertain cases honestly. Spend at most five minutes; unfinished items remain unreviewed. This uses the first five minutes of the existing ten-minute PDF rubric pilot.</p>
<p>Private packet: do not publish the source images, extracted text, notes or exported labels.</p>
<div class="toolbar"><button onclick="move(-1)">Previous</button><span id="progress"></span><button onclick="move(1)">Next</button><button onclick="save()">Export private labels</button><span id="clock"></span></div>
""" + "".join(panels) + """<script>
const metadata = METADATA;
let current=0,last=performance.now(),durations=metadata.map(()=>0);
function tick(){const now=performance.now();if(!document.hidden)durations[current]+=(now-last)/1000;last=now;}
function display(){document.querySelectorAll('.item').forEach((x,i)=>x.style.display=i===current?'block':'none');document.getElementById('progress').textContent=(current+1)+' / '+metadata.length;}
function move(delta){tick();current=Math.max(0,Math.min(metadata.length-1,current+delta));display();}
document.addEventListener('visibilitychange',()=>{last=performance.now();});
setInterval(()=>{tick();document.getElementById('clock').textContent=' Active time: '+Math.round(durations.reduce((a,b)=>a+b,0))+' seconds';},1000);
function save(){tick();const records=metadata.map((row,i)=>({...row,label:document.getElementById('label-'+i).value||null,notes:document.getElementById('notes-'+i).value,duration_seconds:durations[i],finished:!!document.getElementById('label-'+i).value}));const data={schema:'pdf-extraction-human-review-v1',scope:'Source extraction quality only; not answer correctness or claim support',records};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='docmind-hindi-pdf-quality-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
display();</script></html>""".replace("METADATA", safe_json(metadata))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return {"items": len(items), "human_labels": 0, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(json.loads(args.spec.read_text(encoding="utf-8")), args.out)))


if __name__ == "__main__":
    main()
