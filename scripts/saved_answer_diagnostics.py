"""Inspect saved answers without inventing missing evidence or human labels."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from docmind.text import sentence_spans
from docmind.agent.citations import CITATION_RE


def inspect_archive(path):
    result={"archive_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"files":[]}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(("answers.jsonl","heldout.jsonl","calibration.jsonl")) or info.file_size>16*1024*1024:continue
            rows=[json.loads(line) for line in archive.read(info).decode("utf-8").splitlines() if line.strip()]
            rows=[r for r in rows if isinstance(r.get("answer"),str)]
            counts={"records":len(rows),"with_complete_question_and_evidence":0,"claim_boundary_changes":0,"legacy_placeholder_failures":0,"with_citation_labels":0}
            for row in rows:
                answer=row["answer"];citations={}
                def protect(match):
                    token="CITEREF"+str(len(citations))+"TOKEN";citations[token]=match.group();return token
                masked=CITATION_RE.sub(protect,answer)
                old=re.split(r"(?<=[.!?])\s+|\n+",masked)
                try:old=[re.sub(r"CITEREF\d+TOKEN",lambda m:citations[m.group()],part).strip() for part in old if part.strip()]
                except KeyError:counts["legacy_placeholder_failures"]+=1;old=[]
                new=[span.text for span in sentence_spans(answer)]
                counts["claim_boundary_changes"]+=old!=new
                counts["with_citation_labels"]+=bool(CITATION_RE.search(answer))
                evidence=row.get("contexts") or row.get("sources") or row.get("retrieved_chunks")
                counts["with_complete_question_and_evidence"]+=bool(row.get("question") and evidence)
            result["files"].append({"member":info.filename,"counts":counts})
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("archives",nargs="+",type=Path);parser.add_argument("--out",type=Path,required=True);args=parser.parse_args()
    result={"scope":"Saved-answer structure diagnostics, not an attribution-quality benchmark",
        "archives":[inspect_archive(path) for path in args.archives],
        "attribution_precision":None,"attribution_recall":None,
        "limits":["Compact saved records may omit exact model input evidence","Missing inputs are not reconstructed by assumption","No automatic result is a human label"]}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
