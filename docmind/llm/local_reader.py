"""Pinned local reader for the measured T4 profile, independent of hosted credentials."""
import time
import math
import threading
from docmind.llm.base import BaseLLM

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def content_token_confidence(log_scores,token_ids,excluded):
    values=[float(score) for score,token in zip(log_scores,token_ids) if token not in excluded]
    if not values:return 0.0
    if any(not math.isfinite(v) or v>1e-5 for v in values):raise FloatingPointError("Invalid token log probability")
    return min(1.0,math.exp(sum(values)/len(values)))


class LocalReader(BaseLLM):
    def __init__(self,device="cuda:0",max_input_tokens=4096,max_new_tokens=256,capture_token_scores=False):
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        torch.manual_seed(7)
        self.torch=torch;self.device=device
        self.inference_lock = threading.Lock()
        self.tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False,
            dtype=torch.float16,device_map={"":device},attn_implementation="sdpa").eval()
        self._model=MODEL
        self.max_input_tokens=max_input_tokens;self.max_new_tokens=max_new_tokens
        self.capture_token_scores=capture_token_scores
        self.calls=0;self.prompt_tokens=0;self.completion_tokens=0;self.call_records=[]
        self.identity={"model":MODEL,"revision":REVISION,"dtype":"float16","seed":7,
            "temperature":0,"do_sample":False,"attention":self.model.config._attn_implementation,
            "call_accounting":"attempts-v2",
            "max_input_tokens":max_input_tokens,"max_new_tokens":max_new_tokens}
        if capture_token_scores:self.identity["capture_token_scores"]=True

    def complete(self,messages,stop=None,**kwargs):
        """Retain failed attempts and unknown output counts in the timing log."""
        started=time.perf_counter()
        first: list[float] = []
        record: dict[str,int | float | str | bool | None] = {"input_tokens":None,"output_tokens":None,"generation_attempted":False,
                "max_length_reached":None,"numerical_failure":False,"status":"error"}
        try:
            from transformers import LogitsProcessor,LogitsProcessorList
            torch=self.torch
            text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            encoded=self.tokenizer(text,return_tensors="pt").to(self.device)
            length=encoded.input_ids.shape[-1]
            record["input_tokens"]=length
            if length>self.max_input_tokens:
                raise ValueError("Reader input exceeds token budget; no silent truncation")
            class CheckFinite(LogitsProcessor):
                def __call__(self,input_ids,scores):
                    if not first:first.append(time.perf_counter())
                    if not torch.isfinite(scores).all():
                        raise FloatingPointError("Nonfinite reader logits")
                    return scores
            options={"max_new_tokens":min(int(kwargs.get("max_tokens",self.max_new_tokens)),self.max_new_tokens),
                "do_sample":False,"pad_token_id":self.tokenizer.eos_token_id,
                "logits_processor":LogitsProcessorList([CheckFinite()])}
            deadline = kwargs.get("deadline")
            if deadline is not None:
                from transformers import StoppingCriteria,StoppingCriteriaList
                class BeforeDeadline(StoppingCriteria):
                    def __call__(self, input_ids, scores, **unused):
                        if deadline is not None and time.monotonic() >= deadline:
                            from docmind.llm.execution import DeadlineExceeded
                            raise DeadlineExceeded("Local reader deadline exceeded")
                        return False
                options["stopping_criteria"] = StoppingCriteriaList([BeforeDeadline()])
            capture=getattr(self,"capture_token_scores",False)
            if capture:options.update(return_dict_in_generate=True,output_scores=True)
            if stop:options.update(stop_strings=stop,tokenizer=self.tokenizer)
            self.calls+=1
            self.prompt_tokens+=length
            record["generation_attempted"]=True
            with torch.inference_mode():output=self.model.generate(**encoded,**options)
            sequences=output.sequences if capture else output
            tokens=sequences[0,length:]
            record["output_tokens"]=len(tokens)
            self.completion_tokens+=len(tokens)
            record["max_length_reached"]=len(tokens)>=options["max_new_tokens"]
            if capture:
                transition=self.model.compute_transition_scores(sequences,output.scores,normalize_logits=True)
                record["geometric_mean_token_probability"]=content_token_confidence(
                    transition[0].tolist(),tokens.tolist(),set(self.tokenizer.all_special_ids))
            result=self.tokenizer.decode(tokens,skip_special_tokens=True)
            for marker in ([stop] if isinstance(stop,str) else stop or []):result=result.split(marker,1)[0]
            record["status"]="completed"
            return result
        except Exception as exc:
            record["error"]=type(exc).__name__
            record["numerical_failure"]=isinstance(exc,FloatingPointError)
            raise
        finally:
            record["elapsed_seconds"]=time.perf_counter()-started
            record["time_to_first_token_seconds"]=first[0]-started if first else None
            self.call_records.append(record)
