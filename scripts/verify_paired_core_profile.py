"""Verify the frozen local-reader control from published numeric attempt records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from docmind.evaluation.paired import digest, summarize


def verify(root):
    def read(name):return json.loads((root/name).read_text(encoding="utf-8"))
    design=read("design.json");records=read("numeric_records.json");manifest=read("input_manifest.json")
    assert digest(design["rows"])==design["cohort_sha256"]
    assert len(records)==len(design["rows"])*2
    assert not ({r["id"] for r in design["rows"]}&set(manifest["pilot_ids"]+manifest["reserve_pilot_ids"]))
    assert {r["id"] for r in design["rows"]}<=set(manifest["evaluation_pool_ids"])
    assert Counter(r["type"] for r in design["rows"])=={"bridge":21,"comparison":14}
    assert [(r["id"],r["arm"]) for r in records]==[(r["id"],arm) for r in design["rows"] for arm in r["order"]]
    for row in design["rows"]:
        arms=[r for r in records if r["id"]==row["id"]]
        assert len({r["evidence_sha256"] for r in arms})==1
    for record in records:
        assert record["calls"]==len(record["call_records"])
        assert record["tokens"]==sum(c["input_tokens"]+c["output_tokens"] for c in record["call_records"])
        assert record["output_token_cap"]==design["output_token_cap"]==1024
        assert record["status"]=="completed" or record["exact_match"]==0
        assert not {"answer","short_answer","evidence","question","reference"}&set(record)
    assert summarize(design,records)==read("summary.json")
    usage=read("compute_usage.json");source=read("executed_source.json")
    assert source["source_commit"]==usage["source_commit"]
    assert source["sha256"]["scripts/paired_core_profile.py"]==manifest["script_sha256"]
    assert usage["elapsed_seconds_including_setup"]<=usage["session_ceiling_seconds"]==4800
    assert usage["reserved_gpu_seconds"]==9600 and len(usage["allocated_devices"])==2 and not usage["paid_compute"]
    for name,expected in read("publication_manifest.json")["sha256"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
    return {"pairs":len(design["rows"]),"attempts":len(records),"verified":True}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory",type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.directory)))
