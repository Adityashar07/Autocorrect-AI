"""
Unit tests for fix_tense in app.py.
Validates Requirements 3.1, 3.2, 3.3, 3.4, 3.5
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import fix_tense


# ── Past tense (Req 3.1) ──────────────────────────────────────────────────────

class TestPastTense:
    def test_irregular_verb_converted_with_yesterday(self):
        """yesterday + irregular verb in base form → past form"""
        words = ["yesterday", "i", "go", "to", "school"]
        result, changes = fix_tense(words)
        assert "went" in result
        assert "go" not in result

    def test_irregular_verb_converted_with_last(self):
        words = ["last", "week", "she", "eat", "pizza"]
        result, changes = fix_tense(words)
        assert "ate" in result

    def test_irregular_verb_converted_with_ago(self):
        words = ["two", "days", "ago", "he", "run", "fast"]
        result, changes = fix_tense(words)
        assert "ran" in result

    def test_multiple_irregular_verbs_converted(self):
        words = ["yesterday", "they", "go", "and", "eat"]
        result, changes = fix_tense(words)
        assert "went" in result
        assert "ate" in result

    def test_past_tense_change_record_type(self):
        """Req 3.5: change record must have type 'tense'"""
        words = ["yesterday", "i", "go"]
        _, changes = fix_tense(words)
        assert len(changes) > 0
        assert all(c["type"] == "tense" for c in changes)

    def test_past_tense_change_record_detail(self):
        words = ["yesterday", "i", "go"]
        _, changes = fix_tense(words)
        assert any("went" in c["detail"] for c in changes)


# ── Future tense (Req 3.2) ────────────────────────────────────────────────────

class TestFutureTense:
    def test_will_inserted_with_tomorrow(self):
        """tomorrow + subject + base verb → 'will' inserted before verb"""
        words = ["tomorrow", "she", "go", "to", "work"]
        result, changes = fix_tense(words)
        assert "will" in result
        will_idx = result.index("will")
        go_idx = result.index("go")
        assert will_idx < go_idx

    def test_will_inserted_with_soon(self):
        words = ["he", "will", "come", "soon"]
        # already has 'will' — should not double-insert
        result, changes = fix_tense(words)
        assert result.count("will") == 1

    def test_will_inserted_with_next(self):
        words = ["next", "week", "i", "travel"]
        result, changes = fix_tense(words)
        assert "will" in result

    def test_future_change_record_type(self):
        """Req 3.5: change record must have type 'tense'"""
        words = ["tomorrow", "she", "go"]
        _, changes = fix_tense(words)
        assert len(changes) > 0
        assert all(c["type"] == "tense" for c in changes)

    def test_will_not_inserted_when_already_present(self):
        """Req 3.4: already-correct future sentence unchanged"""
        words = ["tomorrow", "she", "will", "go"]
        result, changes = fix_tense(words)
        assert result.count("will") == 1
        assert len(changes) == 0


# ── 3rd-person singular (Req 3.3) ────────────────────────────────────────────

class TestThirdPersonSingular:
    def test_he_plus_base_verb_conjugated(self):
        """he + base verb → verb+s"""
        words = ["he", "run", "every", "day"]
        result, changes = fix_tense(words)
        assert "runs" in result

    def test_she_plus_base_verb_conjugated(self):
        words = ["she", "eat", "lunch"]
        result, changes = fix_tense(words)
        assert "eats" in result

    def test_it_plus_base_verb_conjugated(self):
        words = ["it", "work", "well"]
        result, changes = fix_tense(words)
        assert "works" in result

    def test_irregular_3rd_person(self):
        """Irregular 3rd-person forms (go → goes, do → does)"""
        words = ["he", "go", "to", "school"]
        result, changes = fix_tense(words)
        assert "goes" in result

    def test_verb_ending_in_sh_gets_es(self):
        words = ["she", "wash", "dishes"]
        result, changes = fix_tense(words)
        assert "washes" in result

    def test_verb_ending_in_y_consonant_gets_ies(self):
        words = ["he", "study", "hard"]
        result, changes = fix_tense(words)
        assert "studies" in result

    def test_3rd_person_change_record_type(self):
        """Req 3.5: change record must have type 'tense'"""
        words = ["she", "run"]
        _, changes = fix_tense(words)
        assert len(changes) > 0
        assert all(c["type"] == "tense" for c in changes)


# ── Already-correct verbs unchanged (Req 3.4) ────────────────────────────────

class TestAlreadyCorrect:
    def test_past_form_not_double_modified(self):
        """If verb is already in past form, it should not be changed again"""
        words = ["yesterday", "i", "went", "home"]
        result, changes = fix_tense(words)
        assert "went" in result
        # 'went' is not in IRREGULAR_VERBS keys (it's a value), so no change
        assert len(changes) == 0

    def test_3rd_person_already_conjugated_unchanged(self):
        """he runs → no change (already ends in 's')"""
        words = ["he", "runs", "every", "day"]
        result, changes = fix_tense(words)
        assert result == ["he", "runs", "every", "day"]
        assert len(changes) == 0

    def test_future_already_has_will_unchanged(self):
        words = ["tomorrow", "i", "will", "go"]
        result, changes = fix_tense(words)
        assert result.count("will") == 1
        assert len(changes) == 0

    def test_no_temporal_marker_no_3rd_person_unchanged(self):
        """Sentence with no markers and non-3rd-person subject → unchanged"""
        words = ["i", "go", "to", "school"]
        result, changes = fix_tense(words)
        assert result == ["i", "go", "to", "school"]
        assert len(changes) == 0


# ── Return type ───────────────────────────────────────────────────────────────

class TestReturnType:
    def test_returns_tuple_of_list_and_list(self):
        words = ["he", "run"]
        result = fix_tense(words)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_empty_input(self):
        result, changes = fix_tense([])
        assert result == []
        assert changes == []
