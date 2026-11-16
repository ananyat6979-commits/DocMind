import pytest
from docmind.claim_checks import numeric_conflicts


@pytest.mark.parametrize(
    "claim,quote,reason",
    [
        (
            "The limit is 900 degrees.",
            "The limit is 300 degrees.",
            "unsupported_numeric_literal",
        ),
        (
            "Temperature must exceed 300 degrees.",
            "Temperature must stay below 300 degrees.",
            "opposing_numeric_bound",
        ),
        (
            "Temperature is below 300 degrees.",
            "Temperature exceeds 300 degrees.",
            "opposing_numeric_bound",
        ),
    ],
)
def test_numeric_conflicts(claim, quote, reason):
    assert reason in numeric_conflicts(claim, [quote])


@pytest.mark.parametrize(
    "claim,quote",
    [
        ("Two seconds.", "It closes in two seconds."),
        ("2 seconds.", "It closes in two seconds."),
        ("1 hour.", "It takes 60 minutes."),
        ("1,000 units.", "There are 1000 units."),
        ("à¥§à¥¨.à¥« units.", "The number is 12.5 units."),
        ("Below 300 degrees.", "The limit must not exceed 300 degrees."),
        ("At least 300 degrees.", "The value is above 300 degrees."),
        ("A safe procedure.", "An unsafe procedure."),
    ],
)
def test_equivalence_and_explicit_semantic_limits(claim, quote):
    assert not numeric_conflicts(claim, [quote])


def test_extreme_exponents_cannot_crash_numeric_checks():
    assert numeric_conflicts("The value is 1e999999999.", ["The value is 300."])
