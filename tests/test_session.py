"""Tests for grounding a booking name in what the caller actually said.

No database. The bug this covers was found in a live call: asked to book "the
first one", the model called book_appointment with customer_name "[awaiting]" --
a placeholder it invented rather than a question it asked -- and a real booking
was created under it. The slot-id design already made an invented *time*
impossible; this closes the same hole for names.
"""

from __future__ import annotations

import uuid
from zoneinfo import ZoneInfo

import pytest

from backend.app.agent.session import AgentSession


def session_with(*utterances: str) -> AgentSession:
    session = AgentSession(
        business_id=uuid.uuid4(),
        business_name="Test Clinic",
        tz=ZoneInfo("Asia/Kolkata"),
        caller_phone="+919000000000",
    )
    for text in utterances:
        session.note_caller_speech(text)
    return session


def test_a_name_the_caller_said_is_accepted():
    assert session_with("my name is Asha Menon").name_was_spoken("Asha Menon")


def test_the_placeholder_that_was_actually_invented_is_refused():
    """The exact value from the live call."""
    assert not session_with("Yes, the first one works for me.").name_was_spoken("[awaiting]")


@pytest.mark.parametrize(
    "placeholder",
    ["Unknown", "Caller", "Customer", "Guest", "N/A", "TBD", "???", "-", "", "   "],
)
def test_common_placeholders_are_refused(placeholder):
    assert not session_with("yes the first one works for me please").name_was_spoken(placeholder)


def test_every_word_of_the_name_must_have_been_said():
    """"Asha Menon" is refused if only "Asha" was heard: the surname is a guess."""
    session = session_with("this is Asha")
    assert session.name_was_spoken("Asha")
    assert not session.name_was_spoken("Asha Menon")


def test_a_name_said_in_an_earlier_turn_still_counts():
    session = session_with("hi, Asha here", "a cleaning next Tuesday please", "the first one")
    assert session.name_was_spoken("Asha")


def test_matching_ignores_case_punctuation_and_spacing():
    session = session_with("It's ASHA... menon.")
    assert session.name_was_spoken("asha MENON")
    assert session.name_was_spoken("  Asha,  Menon ")


def test_apostrophes_and_hyphens_do_not_break_a_name():
    session = session_with("my name is Mary-Ann O'Brien")
    assert session.name_was_spoken("Mary-Ann O'Brien")
    assert session.name_was_spoken("Mary Ann OBrien") is False  # "obrien" was never said as one word


def test_a_speech_to_text_spelling_the_model_corrects_is_refused():
    """The transcript says "Maynon". If the model "fixes" it to Menon, that is
    the model guessing -- so it is made to book under what was heard, or ask."""
    session = session_with("My name is Asha Maynon")
    assert not session.name_was_spoken("Asha Menon")
    assert session.name_was_spoken("Asha Maynon")


def test_digits_are_not_a_name():
    assert not session_with("call me on 98765 43210").name_was_spoken("98765")


def test_names_in_other_scripts_are_matched():
    session = session_with("என் பெயர் ஆஷா")
    assert session.name_was_spoken("ஆஷா")


def test_nothing_said_means_nothing_can_be_booked_under():
    assert not session_with().name_was_spoken("Asha")
