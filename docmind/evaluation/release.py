"""Bounded release diagnostics. Drafts, machine checks and human labels stay distinct."""
import hashlib
import json
import math
import random
import re
from dataclasses import asdict
from pathlib import Path
from docmind.text import sentence_spans

SEED = 7
LANGUAGES = ("en", "hi")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def semantic_documents(documents):
    """Ignore transport paths and random object IDs, retain all model-consumed fields."""
    return [{"content": d.content, "metadata": d.metadata} for d in documents]


def cache_pdf(path, source, output):
    from docmind.ingestion.loader import file_sha256, load_document
    if source.get("language") not in LANGUAGES:
        raise ValueError("Only English and Hindi are in this measured release")
    if source.get("license_status") != "established" or not source.get("license_url") or not source.get("attribution"):
        raise ValueError("Document permissions and attribution must be established first")
    output = Path(output)
    if output.exists(): raise FileExistsError("Extraction caches are immutable")
    documents = load_document(path, expected_language=source["language"])
    raw_hash = file_sha256(path)
    pages = [{"page_id":digest([raw_hash, d.metadata["page"]]), "page":d.metadata["page"],
              "text":d.content,"metadata":d.metadata,"text_sha256":hashlib.sha256(d.content.encode()).hexdigest()}
             for d in documents]
    payload = {"schema":"pdf-cache-v1","source":source,"pdf_sha256":raw_hash,
               "pages":pages,"semantic_sha256":digest(semantic_documents(documents)),
               "parser_mode":"trusted local benchmark inputs","human_verified":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=True, indent=2),encoding="utf-8")
    return payload


def template_pool(caches, cap=200):
    if not 1 <= cap <= 200: raise ValueError("Pool cap must be between 1 and 200 per language")
    pools: dict[str,list[dict]] = {language:[] for language in LANGUAGES}
    for cache in caches:
        language = cache["source"]["language"]
        if language not in pools: continue
        for page in cache["pages"]:
            if page["metadata"]["extraction_quality"]["quarantined"]: continue
            for span in sentence_spans(page["text"]):
                text = span.text
                candidates = []
                if language == "en":
                    match = re.search(r"(?P<term>[A-Za-z][A-Za-z -]{2,60}) (?:is defined as|means|refers to) (?P<answer>[^.!?]{4,220})",text)
                    if match: candidates.append(("definition",'What does "'+match["term"].strip()+'" mean?',match["answer"].strip()))
                else:
                    match = re.search(r"(?P<answer>[^।]{4,180}) को (?P<term>[^।]{2,60}) कहते हैं",text)
                    if match: candidates.append(("definition",'"'+match["term"].strip()+'" किसे कहते हैं?',match["answer"].strip()))
                for match in re.finditer(r"(?<![\d.])(?:[0-9०-९]+(?:[.,][0-9०-९]+)*)(?:\s*(?:%|kg|cm|km|mm|years|days|मीटर|वर्ष|दिन|प्रतिशत))?",text):
                    anchor = text[max(0,match.start()-65):match.start()].strip()
                    if len(anchor)<12: continue
                    question = ('What value follows "'+anchor+'"?' if language=="en" else '"'+anchor+'" के बाद कौन-सा मान दिया गया है?')
                    candidates.append(("named_value",question,match.group()))
                for kind, question, answer in candidates:
                    identity = [cache["pdf_sha256"],page["page"],span.start,question]
                    pools[language].append({"id":language+"-"+digest(identity)[:20],"language":language,
                        "question":question,"question_origin":"fixed_template_synthetic_diagnostic",
                        "template":kind,"proposed_answer":answer,"verified":False,
                        "question_validity":None,"answer_correctness":None,"claim_support":None,
                        "source_url":cache["source"]["url"],"pdf_sha256":cache["pdf_sha256"],
                        "page":page["page"],"page_id":page["page_id"],
                        "proposed_evidence":{"start":span.start,"end":span.end,"quote":span.text},
                        "location_exact_match":page["text"][span.start:span.end]==span.text,
                        "location_check_is_correctness":False})
    output=[]
    for language, rows in pools.items():
        unique={r["id"]:r for r in rows}
        output.extend(sorted(unique.values(), key=lambda r:digest([SEED,r["id"]]))[:cap])
    return output


def freeze_cohort(pool, pilot, consumed_seconds, overhead_seconds=0, total_seconds=21600):
    """Use timing only. Both generation conditions must be costed before outcomes."""
    if not 0 <= consumed_seconds <= total_seconds <= 21600: raise ValueError("Invalid total budget")
    if overhead_seconds < 0: raise ValueError("Invalid overhead")
    costs = {}
    for language in LANGUAGES:
        costs[language] = 0.0
        for condition in ("baseline_capture", "fresh"):
            values=sorted(float(r["elapsed_seconds"]) for r in pilot
                          if r["language"]==language and r["condition"]==condition)
            if len(values)<3 or any(not math.isfinite(v) or v<=0 for v in values):
                raise ValueError("At least three finite pilot timings per language and condition are required")
            costs[language] += values[min(len(values)-1,math.ceil(.95*len(values))-1)]*1.25
    usable = max(0,(total_seconds-consumed_seconds)*.8-overhead_seconds)
    n = min(200,math.floor(usable/sum(costs.values())))
    pilot_ids = {r["id"] for r in pilot}
    candidates={lang:sorted((r for r in pool if r["language"]==lang and r["id"] not in pilot_ids),
        key=lambda r:digest([SEED,"cohort",r["id"]])) for lang in LANGUAGES}
    n=min(n,*(len(v) for v in candidates.values()))
    ids=[r["id"] for lang in LANGUAGES for r in candidates[lang][:n]]
    return {"schema":"frozen-cohort-v1","ids":ids,"n_per_language":n,
        "pilot_sha256":digest(pilot),"pool_sha256":digest(pool),"seconds_per_question":costs,
        "total_ceiling_seconds":total_seconds,"consumed_seconds":consumed_seconds,
        "reserve_fraction":.2,"overhead_seconds":overhead_seconds,
        "selection_uses_outcomes":False,"cohort_sha256":digest(ids)}


def cohort_summary(cohort, records):
    ids=cohort["ids"]
    by_id={r["id"]:r for r in records}
    if len(by_id)!=len(records) or set(by_id)-set(ids): raise ValueError("Invalid cohort records")
    completed=[by_id[i] for i in ids if i in by_id and not by_id[i].get("error")]
    return {"requested":len(ids),"completed":len(completed),
        "failed":sum(bool(r.get("error")) for r in records),
        "unserved_ids":[i for i in ids if i not in by_id],
        "completed_only_correctness":sum(bool(r.get("correct")) for r in completed)/len(completed) if completed else None,
        "requested_cohort_correctness":sum(bool(r.get("correct")) for r in completed)/len(ids) if ids else None,
        "correctness_label":"automatic diagnostic against proposed answers; not human validity"}


def blind_review_sample(rows, quotas, seed=SEED):
    """Return reviewer payload separately from the private routing key."""
    rng=random.Random(seed)
    tasks=[];key=[]
    for stratum, count in quotas.items():
        eligible=sorted((r for r in rows if r["routing_stratum"]==stratum),key=lambda r:r["id"])
        if count<0 or count>len(eligible): raise ValueError("Quota exceeds stratum")
        chosen=rng.sample(eligible,count)
        for row in chosen:
            review_id=digest(["blind",seed,row["id"]])[:20]
            tasks.append({"review_id":review_id,**{k:row[k] for k in ("question","answer","evidence")},
                          "question_validity":None,"answer_correctness":None,"claim_support":None,
                          "duration_seconds":None,"finished":False})
            key.append({"review_id":review_id,"source_id":row["id"],"routing_stratum":stratum,
                "inclusion_probability":count/len(eligible),"population":len(eligible),
                "input_sha256":digest([row["answer"],row["evidence"]])})
    rng.shuffle(tasks)
    return tasks,key


def reviewed_summary(tasks, key):
    lookup={r["review_id"]:r for r in key}
    finished=[r for r in tasks if r.get("finished") and r.get("answer_correctness") in (True,False)]
    complete=len(finished)==len(tasks) and bool(tasks)
    releases=[r for r in finished if lookup[r["review_id"]]["routing_stratum"]=="candidate_release"]
    errors=sum(r["answer_correctness"] is False for r in releases)
    return {"reviewed":len(finished),"unfinished":len(tasks)-len(finished),
        "population_extrapolation_allowed":complete,"report_scope":"completed probability sample" if complete else "reviewed subset only",
        "candidate_releases_reviewed":len(releases),"false_release_risk":errors/len(releases) if len(releases)>=25 else "insufficient evidence",
        "total_recorded_seconds":sum(r.get("duration_seconds") or 0 for r in tasks),
        "agreement":"single-reviewer consistency only; no inter-annotator agreement"}
