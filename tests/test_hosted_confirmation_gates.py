from scripts.hosted_core_confirmation import permitted,MAX_CALLS,MAX_CHARGED_TOKENS,MAX_SECONDS


def test_explicit_dual_opt_in_is_required():
    assert not permitted(False,{})
    assert not permitted(True,{})
    assert not permitted(False,{'DOCMIND_ALLOW_HOSTED_EVALUATION':'1'})
    assert not permitted(True,{'DOCMIND_ALLOW_HOSTED_EVALUATION':'true'})
    assert permitted(True,{'DOCMIND_ALLOW_HOSTED_EVALUATION':'1'})
    assert (MAX_CALLS,MAX_CHARGED_TOKENS,MAX_SECONDS)==(500,1000000,7200)
