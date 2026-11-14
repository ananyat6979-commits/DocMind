import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("robustness",Path(__file__).parents[1]/"scripts/current_reader_robustness.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_cohort_is_frozen_unique_and_paired():
    planned=module.cases()
    assert planned==module.cases() and len(planned)==24
    assert len({r["id"] for r in planned})==24
    for kind,n in (("attack",8),("benign",4)):
        for i in range(n):
            pair=[r for r in planned if r["kind"]==kind and r["template"]==i]
            assert len(pair)==2 and pair[0]["payload"]==pair[1]["payload"]


def test_missing_outcomes_remain_unknown_and_boundary_ci_is_not_zero():
    planned=module.cases();attempted=[planned[0]["id"]]
    result=module.summarize(planned,[],attempted)
    assert result["interrupted"]==1 and result["unserved"]==23
    group=result["groups"]["attack_json_current"]
    assert group["model_final_target"]["requested_upper_bound"]==1
    assert group["model_final_target"]["completed_only_rate"] is None
    assert module.wilson(0,8)[1]>.3


def test_quoted_marker_is_not_strict_answer_hijack():
    assert module.strict_target("OVERRIDE_ACCEPTED (Source: procedure.txt)")
    assert not module.strict_target("The limit is 300, not OVERRIDE_ACCEPTED.")


def test_complete_protocol_offline_before_gpu_submission(tmp_path,monkeypatch):
    import os
    import sys
    from types import SimpleNamespace
    from docmind.llm.demo import DemoLLM
    import docmind.llm.local_reader as reader_module
    import docmind.attribution as attribution
    class Reader(DemoLLM):
        def __init__(self,**kwargs):
            self.identity={"model":"offline-test-only"};self.call_records=[]
            self.model=SimpleNamespace(parameters=lambda:iter([SimpleNamespace(dtype="float16")]))
        def complete(self,*args,**kwargs):
            result=super().complete(*args,**kwargs)
            self.call_records.append({"status":"completed"})
            return result
    class Judge:
        identity={"model":"test-only"}
        def __init__(self,*args,**kwargs):pass
        def __call__(self,pairs):return [.99]*len(pairs)
    cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda i:"test-only",
        device_count=lambda:0,max_memory_allocated=lambda:0,mem_get_info=lambda:(0,0))
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(set_num_threads=lambda n:None,cuda=cuda,version=SimpleNamespace(cuda="test-only")))
    monkeypatch.setattr("importlib.metadata.version",lambda name:"offline-test-only")
    monkeypatch.setattr(reader_module,"LocalReader",Reader)
    monkeypatch.setattr(attribution,"NLIJudge",Judge)
    for key in ("DOCMIND_HOME","DOCMIND_TEST_MODE","DOCMIND_SECRETS_FILE","DOCMIND_DEVICE","DOCMIND_TOOL_PROTOCOL"):
        monkeypatch.setenv(key,os.getenv(key,""))
    monkeypatch.setattr("docmind.language.language_info",lambda text:{"language":"en"})
    out=tmp_path/"profile";out.mkdir()
    result=module.run(out)
    assert result["requested"]==result["completed"]==24
    assert result["failed"]==result["interrupted"]==result["unserved"]==0


def test_published_records_verify_and_reject_metric_tampering(tmp_path):
    import shutil
    import json
    spec=importlib.util.spec_from_file_location("verify_robustness",Path(__file__).parents[1]/"scripts/verify_current_reader_robustness.py")
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    source=Path(__file__).parents[1]/"reports/releases/r2/current_reader_t4"
    target=tmp_path/"records";shutil.copytree(source,target)
    assert verifier.verify(target)=={"cases":24,"model_calls":48}
    path=target/"summary.json";data=json.loads(path.read_text())
    data["groups"]["attack_json_current"]["model_final_target"]["completed_only_wilson95"]=[0,0]
    path.write_text(json.dumps(data))
    import pytest
    with pytest.raises(AssertionError):verifier.verify(target)
