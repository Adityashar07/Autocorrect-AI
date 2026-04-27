"""
Unit and property tests for Flask routes and REST API.
Feature: auto-correct-keyboard
"""
import json
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


# ── Unit Tests (task 9.3) ─────────────────────────────────────────────────────

class TestGetIndex:
    """Req 8.1 — UI displays text input area."""

    def test_get_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_get_root_renders_textarea(self, client):
        resp = client.get("/")
        assert b"<textarea" in resp.data


class TestPostCorrect:
    """Req 10.1 — empty form → error; Req 8.2/8.3/8.4/8.7 — result page."""

    def test_empty_text_returns_error_message(self, client):
        """Req 10.1: empty submission must show error, not crash."""
        resp = client.post("/correct", data={"text": ""})
        assert resp.status_code == 200
        assert b"error" in resp.data.lower() or b"please" in resp.data.lower()

    def test_whitespace_only_returns_error_message(self, client):
        """Req 10.1: whitespace-only is treated as empty."""
        resp = client.post("/correct", data={"text": "   "})
        assert resp.status_code == 200
        assert b"error" in resp.data.lower() or b"please" in resp.data.lower()

    def test_valid_text_renders_result_page(self, client):
        """Req 8.2: result page shows original and corrected text."""
        resp = client.post("/correct", data={"text": "hello world"})
        assert resp.status_code == 200
        # result.html contains these labels
        assert b"Original Text" in resp.data
        assert b"Corrected Text" in resp.data

    def test_result_page_contains_original_text(self, client):
        """Req 8.2: original text is present in the result."""
        resp = client.post("/correct", data={"text": "hello world"})
        assert b"hello world" in resp.data

    def test_result_page_no_corrections_message(self, client):
        """Req 8.4: 'no corrections needed' shown when text is already correct."""
        resp = client.post("/correct", data={"text": "Hello world."})
        assert resp.status_code == 200
        assert b"No corrections needed" in resp.data or b"Perfect" in resp.data

    def test_result_page_has_back_navigation_link(self, client):
        """Req 8.7: navigation link back to input page."""
        resp = client.post("/correct", data={"text": "hello world"})
        assert b'href="/"' in resp.data

    def test_result_page_shows_change_badges(self, client):
        """Req 8.3: change records with type badges are displayed."""
        # "i running" needs grammar fix (missing helping verb) + punctuation
        resp = client.post("/correct", data={"text": "i running"})
        assert resp.status_code == 200
        # The badge CSS classes should be present
        assert b"change-badge" in resp.data


class TestApiCorrect:
    """Req 9.1/9.3/10.3 — JSON API endpoint."""

    def test_returns_200_with_valid_json(self, client):
        """Req 9.4: HTTP 200 for valid requests."""
        resp = client.post(
            "/api/correct",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_response_has_corrected_and_details_keys(self, client):
        """Req 9.2: response contains 'corrected' string and 'details' list."""
        resp = client.post(
            "/api/correct",
            data=json.dumps({"text": "hello world"}),
            content_type="application/json",
        )
        body = resp.get_json()
        assert "corrected" in body
        assert "details" in body
        assert isinstance(body["corrected"], str)
        assert isinstance(body["details"], list)

    def test_empty_text_returns_empty_corrected(self, client):
        """Req 9.3: empty string → empty corrected + empty details."""
        resp = client.post(
            "/api/correct",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        body = resp.get_json()
        assert body["corrected"] == ""
        assert body["details"] == []

    def test_missing_text_field_defaults_to_empty_string(self, client):
        """Req 10.3: missing 'text' field treated as empty string."""
        resp = client.post(
            "/api/correct",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["corrected"] == ""
        assert body["details"] == []

    def test_no_json_body_defaults_to_empty_string(self, client):
        """Req 10.3: request with no body at all treated as empty string."""
        resp = client.post("/api/correct", data="", content_type="application/json")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["corrected"] == ""
        assert body["details"] == []

    def test_corrections_applied_and_returned(self, client):
        """Req 9.2: corrections are reflected in the response."""
        resp = client.post(
            "/api/correct",
            data=json.dumps({"text": "i running"}),
            content_type="application/json",
        )
        body = resp.get_json()
        # Should have at least one change record
        assert len(body["details"]) > 0


# ── Property Test (task 9.2) ──────────────────────────────────────────────────
# Feature: auto-correct-keyboard, Property 15: API response structure and status
# Validates: Requirements 9.2, 9.4

@pytest.fixture
def prop_client():
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c


@given(text=st.text(min_size=0, max_size=200))
@settings(max_examples=100, deadline=None)
def test_property_15_api_response_structure(text):
    """
    Property 15: For any valid POST to /api/correct with a JSON body containing
    a 'text' field, the response should have HTTP 200 and a JSON body with both
    a 'corrected' string field and a 'details' list field.

    Validates: Requirements 9.2, 9.4
    """
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        resp = client.post(
            "/api/correct",
            data=json.dumps({"text": text}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body is not None
        assert "corrected" in body
        assert "details" in body
        assert isinstance(body["corrected"], str)
        assert isinstance(body["details"], list)
