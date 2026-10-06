"""Trending today: Gemini response parsing, caching, key handling, photo privacy (words only), endpoint."""
import io
import json
import urllib.error

import pytest

from server import trends as T
from server import web_api as A
from tests.test_lumen_api import Fake, client, scene, url

GOOD = {"candidates": [{"content": {"parts": [{"text": "Here you go:\n```json\n" + json.dumps([
    {"title": "35mm film look", "why": "Big on Instagram this month.", "prompt": "Give the photo a warm 35mm film look"},
    {"title": "35MM FILM LOOK", "why": "dup", "prompt": "x"},
    {"title": "Butter yellow", "why": "Pinterest colour of the season.", "prompt": "Make the main garment butter yellow"},
    {"title": "", "prompt": "no title"}]) + "\n```"}]},
    "groundingMetadata": {"groundingChunks": [{"web": {"uri": "https://a.example/x", "title": "A"}},
                                              {"web": {"uri": "https://a.example/x", "title": "A"}},
                                              {"web": {"uri": "https://b.example/y", "title": "B"}}]}}]}


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    T._CACHE.clear()


def recorder(resp=GOOD):
    calls = []

    def post(u, body, headers, timeout):
        calls.append((u, body, headers))
        return resp
    return post, calls


def test_parse_dedupes_and_collects_sources():
    items, src = T.parse(GOOD, 6)
    assert [i["title"] for i in items] == ["35mm film look", "Butter yellow"]
    assert [s["url"] for s in src] == ["https://a.example/x", "https://b.example/y"]


def test_request_uses_search_grounding_and_header_key():
    post, calls = recorder()
    r = T.trends("wedding photos", post=post, now=1.8e9)
    u, body, headers = calls[0]
    assert body["tools"] == [{"google_search": {}}] and headers["x-goog-api-key"] == "test-key"
    assert "test-key" not in u and "wedding photos" in body["contents"][0]["parts"][0]["text"]
    assert r["topic"] == "wedding photos" and len(r["trends"]) == 2 and r["cached"] is False


def test_cache_then_expiry():
    post, calls = recorder()
    T.trends(post=post, now=1.8e9)
    assert T.trends(post=post, now=1.8e9 + 3600)["cached"] is True and len(calls) == 1
    T.trends(post=post, now=1.8e9 + 7 * 3600)
    assert len(calls) == 2


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    with pytest.raises(T.TrendsError, match="adobe-ps-trends"):
        T.trends(post=recorder()[0])


def test_retired_model_falls_through_then_bad_key_message():
    seen = []

    def post(u, body, headers, timeout):
        seen.append(u)
        code = 404 if len(seen) == 1 else 403
        raise urllib.error.HTTPError(u, code, "x", {}, io.BytesIO(b"denied"))
    with pytest.raises(T.TrendsError, match="Check the API key"):
        T.trends(post=post)
    assert len(seen) == 2


def test_endpoint_photo_goes_as_words_only(monkeypatch):
    sent = []

    def fake_trends(topic, words, n):
        sent.append((topic, words))
        return {"trends": [{"title": "t", "why": "w", "prompt": "p"}], "sources": []}
    monkeypatch.setattr(T, "trends", fake_trends)
    c = client(Fake())
    r = c.post("/api/v1/trends", json={"image": url(scene()), "for_photo": True, "topic": "product"}).json()
    assert r["trends"][0]["prompt"] == "p" and sent == [("product", "a vase")]    # Qwen's words, not the image
    c.post("/api/v1/trends", json={"image": url(scene())})
    assert sent[-1] == ("", "")


def test_endpoint_error_is_400(monkeypatch):
    monkeypatch.setattr(T, "trends", lambda *a: (_ for _ in ()).throw(T.TrendsError("quota")))
    r = client(Fake()).post("/api/v1/trends", json={})
    assert r.status_code == 400 and r.json()["detail"] == "quota"
