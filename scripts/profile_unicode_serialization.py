"""CPU-only serialization/tokenization component. No model generation or quality labels."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from docmind.evidence import untrusted_json
MODEL='Qwen/Qwen3-4B-Instruct-2507'
REVISION='cdbee75f17c01a7cc42f958dc650907174af0554'


def profile(rows,tokenizer):
    records=[]
    for index,row in enumerate(rows):
        payload={'question':row['question'],'untrusted_evidence':[{'chunk_id':'diagnostic-'+str(index),'text':row['evidence']['quote']}]}
        values={'ascii_json':json.dumps(payload,ensure_ascii=True),'readable_unicode_json':untrusted_json(payload)}
        assert all(json.loads(value)==payload for value in values.values())
        timings={key:[] for key in values}
        for value in values.values():tokenizer.encode(value)
        for repeat in range(20):
            order=list(values) if repeat%2==0 else list(reversed(values))
            for key in order:
                start=time.perf_counter_ns();tokenizer.encode(values[key]);timings[key].append((time.perf_counter_ns()-start)/1e6)
        record={'input_sha256':hashlib.sha256(json.dumps(payload,ensure_ascii=True,sort_keys=True).encode()).hexdigest(),'conditions':{}}
        for key,value in values.items():
            record['conditions'][key]={'bytes':len(value.encode()),'tokens':len(tokenizer.encode(value).ids),
                'cpu_encode_p50_ms':statistics.median(timings[key]),'cpu_encode_p95_ms':statistics.quantiles(timings[key],n=20,method='inclusive')[18],
                'timing_repeats':20}
        records.append(record)
    before=sum(r['conditions']['ascii_json']['tokens'] for r in records);after=sum(r['conditions']['readable_unicode_json']['tokens'] for r in records)
    return {'scope':'Hindi user-payload serialization and pinned tokenizer on CPU only; not chat-template cost, T4 prefill, generation latency or answer correctness',
        'source_status':'Private restricted-source synthetic diagnostic drafts, not human gold','n':len(records),'records':records,
        'total_ascii_tokens':before,'total_unicode_tokens':after,'token_reduction_fraction':1-after/before if before else None,
        'normalization':'Source Unicode preserved. Only JSON escaping changes; directional and selected invisible controls stay escaped.',
        'hosted_calls':0,'model_generations':0,'human_labels_created':0,'paid_compute':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('input',type=Path);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Component outputs are immutable')
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tokenizer_path=hf_hub_download(repo_id=MODEL,filename='tokenizer.json',revision=REVISION,token=False)
    result=profile(json.loads(args.input.read_text(encoding='utf-8')),Tokenizer.from_file(tokenizer_path))
    result.update(tokenizer={'model':MODEL,'revision':REVISION,'file_sha256':hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest(),
        'tokenizers_version':importlib.metadata.version('tokenizers')},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('n','total_ascii_tokens','total_unicode_tokens','token_reduction_fraction','model_generations','paid_compute')}))
