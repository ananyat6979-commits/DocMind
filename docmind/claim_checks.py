"""Narrow numeric contradiction checks. Passing does not establish support."""

import re
import unicodedata
from decimal import Decimal, InvalidOperation

NUMBER = r"[+-]?(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_DIGITS = re.compile(r"(?<![\w.])(" + NUMBER + r")(?!\w|\.[0-9])")
_WORDS = dict(
    zip(
        "zero one two three four five six seven eight nine ten eleven twelve".split(),
        map(Decimal, range(13)),
        strict=True,
    )
)
_BOUND = re.compile(
    r"\b(not below|not under|no less than|not less than|not more than|not exceed|no more than|at most|less than|below|under|at least|greater than|more than|above|over|exceeds?|exceeding)\s+("
    + NUMBER
    + r")",
    re.I,
)
_UPPER = {
    "not more than",
    "not exceed",
    "no more than",
    "at most",
    "less than",
    "below",
    "under",
}
_UNITS = {
    "second": 1,
    "minute": 60,
    "hour": 3600,
    "meter": 1,
    "centimeter": 0.01,
    "millimeter": 0.001,
}


def _text(value):
    return "".join(
        str(unicodedata.decimal(c)) if c.isdecimal() else c
        for c in unicodedata.normalize("NFC", value)
    ).casefold()


def _number(value):
    try:
        number = Decimal(value.replace(",", ""))
        return (
            number if number.is_finite() and abs(number.adjusted()) <= 10000 else None
        )
    except InvalidOperation:
        return None


def numeric_conflicts(claim, quotes):
    """Catch new literal numbers and opposite bounds at the same numeric value.

    No arithmetic inference, entity verification or general entailment is claimed.
    A conservative hold may require review for a valid derived numerical answer.
    """
    text, evidence = _text(claim), _text(" ".join(quotes))
    values = {_number(m.group()) for m in _DIGITS.finditer(evidence)}
    values.update(_WORDS[w] for w in re.findall(r"\b[a-z]+\b", evidence) if w in _WORDS)
    errors = []
    for m in _DIGITS.finditer(text):
        value = _number(m.group())
        if value is None:
            errors.append("unsupported_numeric_literal")
            continue
        if value in values:
            continue
        # Exact common-unit conversions, not approximate numerical similarity.
        tail = text[m.end() :]
        unit = re.match(
            r"\s+(seconds?|minutes?|hours?|meters?|centimeters?|millimeters?)\b", tail
        )
        converted = False
        if unit:
            key = unit.group(1).rstrip("s")
            family = (
                {"second", "minute", "hour"}
                if key in {"second", "minute", "hour"}
                else {"meter", "centimeter", "millimeter"}
            )
            for other in re.finditer(
                "("
                + NUMBER
                + r")\s+(seconds?|minutes?|hours?|meters?|centimeters?|millimeters?)\b",
                evidence,
            ):
                source_unit = other.group(2).rstrip("s")
                source_value = _number(other.group(1))
                if (
                    source_value is not None
                    and source_unit in family
                    and value * Decimal(str(_UNITS[key]))
                    == source_value * Decimal(str(_UNITS[source_unit]))
                ):
                    converted = True
        if not converted:
            errors.append("unsupported_numeric_literal")
    source_bounds = {
        (_number(m.group(2)), m.group(1) in _UPPER)
        for m in _BOUND.finditer(evidence)
        if not re.search(
            r"\b(?:not|never|no)\b", evidence[max(0, m.start() - 15) : m.start()]
        )
    }
    for m in _BOUND.finditer(text):
        if re.search(r"\b(?:not|never|no)\b", text[max(0, m.start() - 15) : m.start()]):
            continue
        value, upper = _number(m.group(2)), m.group(1) in _UPPER
        if (value, not upper) in source_bounds and (value, upper) not in source_bounds:
            errors.append("opposing_numeric_bound")
    return sorted(set(errors))
