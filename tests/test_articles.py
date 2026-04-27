"""
Unit tests for fix_articles in app.py.
Covers Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""
import pytest
from app import fix_articles


# ── Requirement 4.1: "a" before vowel-sound word → replace with "an" ─────────

def test_a_before_vowel_replaced_with_an():
    words, changes = fix_articles(["a", "apple"])
    assert words[0] == "an"
    assert len(changes) == 1
    assert changes[0]["type"] == "article"

def test_a_before_elephant():
    words, changes = fix_articles(["a", "elephant"])
    assert words[0] == "an"

def test_a_before_orange():
    words, changes = fix_articles(["a", "orange"])
    assert words[0] == "an"


# ── Requirement 4.2: "an" before consonant-sound word → replace with "a" ─────

def test_an_before_consonant_replaced_with_a():
    words, changes = fix_articles(["an", "book"])
    assert words[0] == "a"
    assert len(changes) == 1
    assert changes[0]["type"] == "article"

def test_an_before_dog():
    words, changes = fix_articles(["an", "dog"])
    assert words[0] == "a"


# ── Requirement 4.3: Silent H words → use "an" ───────────────────────────────

def test_a_hour_becomes_an_hour():
    words, changes = fix_articles(["a", "hour"])
    assert words[0] == "an"

def test_a_honest_becomes_an_honest():
    words, changes = fix_articles(["a", "honest"])
    assert words[0] == "an"

def test_a_honor_becomes_an_honor():
    words, changes = fix_articles(["a", "honor"])
    assert words[0] == "an"

def test_a_heir_becomes_an_heir():
    words, changes = fix_articles(["a", "heir"])
    assert words[0] == "an"

def test_a_herb_becomes_an_herb():
    words, changes = fix_articles(["a", "herb"])
    assert words[0] == "an"


# ── Requirement 4.4: Consonant-sound vowel-letter words → use "a" ────────────

def test_an_university_becomes_a_university():
    words, changes = fix_articles(["an", "university"])
    assert words[0] == "a"

def test_an_european_becomes_a_european():
    words, changes = fix_articles(["an", "European"])
    assert words[0] == "a"

def test_an_unique_becomes_a_unique():
    words, changes = fix_articles(["an", "unique"])
    assert words[0] == "a"

def test_an_uniform_becomes_a_uniform():
    words, changes = fix_articles(["an", "uniform"])
    assert words[0] == "a"

def test_an_one_becomes_a_one():
    words, changes = fix_articles(["an", "one"])
    assert words[0] == "a"

def test_an_once_becomes_a_once():
    words, changes = fix_articles(["an", "once"])
    assert words[0] == "a"


# ── Requirement 4.5: Already correct → unchanged ─────────────────────────────

def test_an_apple_unchanged():
    words, changes = fix_articles(["an", "apple"])
    assert words[0] == "an"
    assert changes == []

def test_a_book_unchanged():
    words, changes = fix_articles(["a", "book"])
    assert words[0] == "a"
    assert changes == []

def test_an_hour_unchanged():
    words, changes = fix_articles(["an", "hour"])
    assert words[0] == "an"
    assert changes == []

def test_a_university_unchanged():
    words, changes = fix_articles(["a", "university"])
    assert words[0] == "a"
    assert changes == []


# ── Requirement 4.6: Change record type is "article" ─────────────────────────

def test_change_record_type_is_article():
    _, changes = fix_articles(["a", "apple"])
    assert changes[0]["type"] == "article"

def test_change_record_has_detail():
    _, changes = fix_articles(["a", "apple"])
    assert "detail" in changes[0]


# ── Case preservation ─────────────────────────────────────────────────────────

def test_capital_A_apple_becomes_An_apple():
    words, changes = fix_articles(["A", "apple"])
    assert words[0] == "An"

def test_all_caps_AN_book_becomes_A():
    words, changes = fix_articles(["AN", "book"])
    assert words[0] == "A"

def test_all_caps_A_apple_becomes_AN():
    # Single uppercase "A" should become "An" (not "AN") — it's title-case, not all-caps
    words, changes = fix_articles(["A", "apple"])
    assert words[0] == "An"


# ── Multi-word sentence ───────────────────────────────────────────────────────

def test_article_in_sentence():
    words, changes = fix_articles(["I", "ate", "a", "apple", "today"])
    assert words[2] == "an"
    assert len(changes) == 1

def test_no_article_in_sentence_unchanged():
    words, changes = fix_articles(["I", "ate", "the", "apple"])
    assert changes == []
