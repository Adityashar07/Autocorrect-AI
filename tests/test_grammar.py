"""
Unit tests for fix_helping_verb (GrammarFixer) in app.py.
Validates Requirements 2.1, 2.2, 2.3, 2.4
"""
import pytest
from app import fix_helping_verb


# ── Requirement 2.1: pronouns + -ing verb → insert correct helping verb ──────

@pytest.mark.parametrize("subject,expected_verb", [
    ("I", "am"),
    ("he", "is"),
    ("she", "is"),
    ("it", "is"),
    ("we", "are"),
    ("they", "are"),
    ("you", "are"),
])
def test_all_pronouns_insert_correct_helping_verb(subject, expected_verb):
    """Req 2.1: each pronoun followed by -ing verb gets the right helping verb."""
    words = [subject, "running"]
    result, changes = fix_helping_verb(words)
    assert result == [subject, expected_verb, "running"], (
        f"Expected '{subject} {expected_verb} running', got {result}"
    )
    assert len(changes) == 1
    assert changes[0]["type"] == "grammar"


# ── Requirement 2.2: proper noun + -ing verb → insert "is" ───────────────────

def test_proper_noun_inserts_is():
    """Req 2.2: capitalized proper noun followed by -ing verb gets 'is'."""
    words = ["John", "walking"]
    result, changes = fix_helping_verb(words)
    assert result == ["John", "is", "walking"]
    assert len(changes) == 1
    assert changes[0]["type"] == "grammar"


def test_proper_noun_various_names():
    """Req 2.2: various proper nouns all get 'is'."""
    for name in ["Alice", "Bob", "Maria", "London"]:
        words = [name, "singing"]
        result, changes = fix_helping_verb(words)
        assert result == [name, "is", "singing"], f"Failed for proper noun '{name}'"
        assert changes[0]["type"] == "grammar"


# ── Requirement 2.3: helping verb already present → no duplication ────────────

@pytest.mark.parametrize("words", [
    ["he", "is", "running"],
    ["she", "is", "dancing"],
    ["they", "are", "playing"],
    ["we", "are", "working"],
    ["I", "am", "eating"],
    ["you", "are", "sleeping"],
    ["it", "is", "raining"],
])
def test_helping_verb_already_present_unchanged(words):
    """Req 2.3: when helping verb is already present, sentence is left unchanged."""
    result, changes = fix_helping_verb(words)
    assert result == words, f"Expected unchanged {words}, got {result}"
    assert changes == [], f"Expected no changes, got {changes}"


def test_proper_noun_with_helping_verb_unchanged():
    """Req 2.3: proper noun + helping verb + -ing → unchanged."""
    words = ["John", "is", "walking"]
    result, changes = fix_helping_verb(words)
    assert result == words
    assert changes == []


# ── Requirement 2.4: change record type is "grammar" ─────────────────────────

def test_change_record_type_is_grammar():
    """Req 2.4: grammar correction produces a Change Record of type 'grammar'."""
    words = ["they", "playing"]
    result, changes = fix_helping_verb(words)
    assert len(changes) == 1
    assert changes[0]["type"] == "grammar"
    assert "they" in changes[0]["detail"]
    assert "are" in changes[0]["detail"]


# ── Additional edge cases ─────────────────────────────────────────────────────

def test_no_ing_verb_no_change():
    """No -ing verb present → no helping verb inserted."""
    words = ["he", "runs"]
    result, changes = fix_helping_verb(words)
    assert result == ["he", "runs"]
    assert changes == []


def test_short_ing_word_not_treated_as_participle():
    """Words ending in -ing but too short (≤4 chars) are not treated as participles."""
    # "ring" is 4 chars, is_present_participle requires len > 4
    words = ["he", "ring"]
    result, changes = fix_helping_verb(words)
    assert result == ["he", "ring"]
    assert changes == []


def test_multiple_subjects_in_sentence():
    """Multiple subject+-ing pairs in one sentence each get a helping verb."""
    words = ["he", "running", "and", "she", "dancing"]
    result, changes = fix_helping_verb(words)
    assert "is" in result
    assert len(changes) == 2


def test_empty_words_list():
    """Empty input returns empty output with no changes."""
    result, changes = fix_helping_verb([])
    assert result == []
    assert changes == []


def test_return_type_is_tuple_of_list_and_list():
    """fix_helping_verb always returns (list, list)."""
    result, changes = fix_helping_verb(["he", "running"])
    assert isinstance(result, list)
    assert isinstance(changes, list)
