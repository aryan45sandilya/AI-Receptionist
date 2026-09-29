"""Invariants of the tool schemas themselves.

Separate from test_tools.py because these need no database and no fixtures --
they read the schema definitions and check properties that quietly stop being
true otherwise.

Nothing on the wire enforces a tool's JSON schema: it is a hint to the model,
and the real checking happens in ``validate_arguments``, which is tested here
too. A schema that quietly stops being complete would take the validation with
it, since the validator reads these definitions.
"""

from __future__ import annotations

import pytest

from backend.app.agent.tools import _tool_definitions, tool_schemas, validate_arguments


def test_every_tool_declares_a_closed_complete_schema():
    """Every parameter required, nothing extra allowed.

    Even unenforced, this is what tells the model exactly what to send, and it
    is what validate_arguments reads to decide whether a call is usable.
    """
    for tool in _tool_definitions():
        schema = tool["parameters"]
        assert schema["additionalProperties"] is False, tool["name"]
        assert set(schema["required"]) == set(schema["properties"]), tool["name"]
        for name, spec in schema["properties"].items():
            assert spec.get("description") or spec.get("enum"), f"{tool['name']}.{name}"


def test_wire_format_is_openai_shaped():
    """Groq speaks OpenAI's tools format: {"type": "function", "function": {...}}."""
    for tool in tool_schemas():
        assert tool["type"] == "function"
        function = tool["function"]
        assert set(function) == {"name", "description", "parameters"}


def test_book_appointment_cannot_express_a_time():
    """The structural guard, asserted at the schema level.

    If a date or time parameter is ever added to book_appointment, the model
    regains the ability to state a time no tool gave it, and this test is the
    tripwire. It matters more with a smaller model, not less.
    """
    book = next(t for t in _tool_definitions() if t["name"] == "book_appointment")
    properties = set(book["parameters"]["properties"])
    assert properties == {"slot_id", "customer_name", "customer_phone"}
    for forbidden in ("date", "time", "when", "slot_start", "datetime"):
        assert forbidden not in properties


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_valid_arguments_pass_through():
    cleaned, error = validate_arguments(
        "check_availability",
        {"service": "Dental cleaning", "when": "tomorrow", "staff_name": None},
    )
    assert error is None
    assert cleaned == {"service": "Dental cleaning", "when": "tomorrow", "staff_name": None}


def test_a_missing_required_argument_is_reported_to_the_model():
    """A 70B model omits a field often enough that this has to be an answer it
    can act on, not an exception that ends the call."""
    cleaned, error = validate_arguments("check_availability", {"service": "cleaning"})
    assert cleaned is None
    assert "when" in error
    assert "call it again" in error.lower()


def test_optional_arguments_may_be_omitted_entirely():
    cleaned, error = validate_arguments(
        "check_availability", {"service": "cleaning", "when": "tomorrow"}
    )
    assert error is None
    assert cleaned["staff_name"] is None


@pytest.mark.parametrize("nullish", ["", "  ", "null", "None", "N/A", "undefined", "unknown"])
def test_strings_meaning_nothing_become_none(nullish):
    """Models write "null" as a string surprisingly often. Left alone it would
    reach the database as a customer named "null"."""
    cleaned, error = validate_arguments(
        "check_availability",
        {"service": "cleaning", "when": "tomorrow", "staff_name": nullish},
    )
    assert error is None
    assert cleaned["staff_name"] is None


def test_a_required_argument_that_is_only_whitespace_is_rejected():
    cleaned, error = validate_arguments(
        "check_availability", {"service": "   ", "when": "tomorrow", "staff_name": None}
    )
    assert cleaned is None
    assert "service" in error


def test_invented_arguments_are_dropped():
    """A model that hallucinates an extra parameter must not have it reach a
    handler as an unexpected keyword."""
    cleaned, error = validate_arguments(
        "check_availability",
        {"service": "cleaning", "when": "tomorrow", "staff_name": None, "urgency": "high"},
    )
    assert error is None
    assert "urgency" not in cleaned


def test_values_are_stripped():
    cleaned, _ = validate_arguments(
        "book_appointment",
        {"slot_id": "  s2 ", "customer_name": " Asha ", "customer_phone": None},
    )
    assert cleaned["slot_id"] == "s2"
    assert cleaned["customer_name"] == "Asha"


def test_unknown_tool_name_is_reported_not_raised():
    cleaned, error = validate_arguments("teleport_caller", {})
    assert cleaned is None
    assert "no tool named" in error


def test_handoff_reason_is_constrained_to_known_values():
    """The dashboard groups handoffs by reason, so it cannot be free text."""
    handoff = next(t for t in _tool_definitions() if t["name"] == "handoff_to_human")
    assert handoff["parameters"]["properties"]["reason"]["enum"]
