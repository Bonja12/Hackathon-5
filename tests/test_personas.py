import pytest

from provile.personas import (
    BELIEF_PROBES,
    PERSONAS,
    apply_persona,
    get_persona,
    mentions_country,
)


def test_baseline_adds_nothing():
    assert apply_persona(None, "system", "hello") == (None, "hello")


def test_system_delivery_puts_country_in_system_prompt_only():
    system, user = apply_persona(get_persona("cn"), "system", "hello")
    assert "China" in system
    assert user == "hello"


def test_metadata_delivery_contains_locale_and_timezone():
    system, user = apply_persona(get_persona("NL"), "metadata", "hello")
    assert "nl-NL" in system and "Europe/Amsterdam" in system and "NL" in system
    assert user == "hello"


def test_user_delivery_prefixes_message_and_has_no_system_prompt():
    system, user = apply_persona(get_persona("US"), "user", "hello")
    assert system is None
    assert user.startswith("I am writing from the United States.")
    assert user.endswith("hello")


def test_persona_text_never_instructs_behaviour_change():
    for persona in PERSONAS.values():
        for delivery in ("system", "metadata", "user"):
            system, user = apply_persona(persona, delivery, "x")
            text = f"{system or ''} {user}".lower()
            assert "should" not in text and "must" not in text


def test_unknown_persona_and_delivery_raise():
    with pytest.raises(ValueError):
        get_persona("XX")
    with pytest.raises(ValueError):
        apply_persona(get_persona("US"), "carrier-pigeon", "x")


def test_mentions_country_uses_aliases():
    assert mentions_country("The user is in the Netherlands.", get_persona("NL"))
    assert mentions_country("Dutch", get_persona("NL"))
    assert not mentions_country("unknown", get_persona("NL"))


def test_belief_probes_are_defined():
    assert len(BELIEF_PROBES) >= 2
