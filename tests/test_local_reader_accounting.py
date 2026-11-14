"""No neural imports or generation: verify failure denominators and unknown counts."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
import pytest
from docmind.llm.local_reader import LocalReader


class Encoded(dict):
    def __init__(self, length):
        self.input_ids = SimpleNamespace(shape=(1,length))
        super().__init__(input_ids=self.input_ids)
    def to(self, device):
        return self


def reader(monkeypatch, length=4):
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(LogitsProcessor=object,LogitsProcessorList=list))
    instance=LocalReader.__new__(LocalReader)
    class Tokenizer:
        eos_token_id=0
        def apply_chat_template(self,*a,**kw):return "prompt"
        def __call__(self,*a,**kw):return Encoded(length)
    instance.tokenizer=Tokenizer()
    instance.torch=SimpleNamespace(inference_mode=nullcontext)
    instance.device="cpu";instance.max_input_tokens=8;instance.max_new_tokens=16
    instance.calls=instance.prompt_tokens=instance.completion_tokens=0
    instance.call_records=[]
    return instance


def test_numerical_failure_keeps_attempt_and_unknown_generated_tokens(monkeypatch):
    instance=reader(monkeypatch)
    def generate(**kwargs):raise FloatingPointError("nonfinite")
    instance.model=SimpleNamespace(generate=generate)
    with pytest.raises(FloatingPointError):instance.complete([])
    assert instance.calls==1 and instance.prompt_tokens==4
    row=instance.call_records[0]
    assert row["status"]=="error" and row["generation_attempted"] and row["numerical_failure"]
    assert row["output_tokens"] is None and row["max_length_reached"] is None
    assert row["elapsed_seconds"]>=0


def test_prefill_budget_failure_is_recorded_without_generation(monkeypatch):
    instance=reader(monkeypatch,length=9)
    with pytest.raises(ValueError,match="token budget"):instance.complete([])
    assert instance.calls==instance.prompt_tokens==0
    row=instance.call_records[0]
    assert row["input_tokens"]==9 and not row["generation_attempted"]
    assert row["error"]=="ValueError" and row["output_tokens"] is None


def test_success_retains_stop_behavior_and_token_counters(monkeypatch):
    instance=reader(monkeypatch)
    class Output:
        def __getitem__(self,key):return [1,2,3]
    instance.model=SimpleNamespace(generate=lambda **kwargs: Output())
    instance.tokenizer.decode=lambda *args,**kwargs: "300 STOPignored"
    assert instance.complete([],stop=["STOP"],max_tokens=3)=="300 "
    assert instance.calls==1 and instance.prompt_tokens==4 and instance.completion_tokens==3
    row=instance.call_records[0]
    assert row["status"]=="completed" and row["output_tokens"]==3 and row["max_length_reached"]
