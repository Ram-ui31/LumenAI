"""LumenAI "Trending today": Gemini with Google Search grounding finds what photo looks are popular right now and turns
each into an edit instruction our pipeline can carry out.

The key lives only in the Modal secret ``adobe-ps-trends`` (GEMINI_API_KEY); it is never sent to the browser.
The user's photo is never sent to Google: for "trends for this photo" Qwen3-VL describes it in words first, and only
that short description goes into the search prompt.

Answers are cached (6 h for the general list, 1 h per topic/photo description) so the API is called rarely.
"""

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MODELS = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-3.8-flash", "gemini-flash-latest", "gemini-2.5-flash") if m]

CAN_DO = ("recolour an object or garment; remove objects (with their shadows); change the lighting or time of day; "
          "colour grading and film looks; change or blur the background; expand the frame to a new aspect ratio; "
          "add an object; change a person's pose; restyle the whole photo in an art style; sharpen a blurry photo")

PROMPT = """You are a trend researcher for a photo-editing app. Use Google Search to find photo-editing and visual
looks that are popular RIGHT NOW (this week or this month) on Instagram, TikTok, Pinterest and photography sites{topic}.
Today's date is {date}.{photo}

Return {n} different trends. Each one must be something this app can do to an existing photo with one instruction.
The app can: {can_do}.

Reply with JSON only, no other text:
[{{"title": "short name of the look (max 5 words)",
   "why": "one sentence: where it is trending and why",
   "prompt": "the exact edit instruction to apply it to a photo, e.g. 'Give the photo a warm 35mm film look with soft grain and faded blacks'"}}]"""


class TrendsError(RuntimeError):
    pass


def _key() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise TrendsError("Trends need the Gemini key: add GEMINI_API_KEY to the Modal secret 'adobe-ps-trends'.")
    return k


def call_gemini(prompt: str, timeout: float = 60.0, post=None) -> dict:
    """POST generateContent with Google Search grounding. ``post(url, body, headers) -> dict`` is injectable (tests)."""
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.7}}
    headers = {"Content-Type": "application/json", "x-goog-api-key": _key()}
    post = post or _post
    last = None
    for model in MODELS:
        try:
            return post(API.format(model=model), body, headers, timeout)
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:             # model name retired: try the next one
                continue
            detail = e.read().decode(errors="replace")[:300]
            if e.code in (400, 401, 403):
                raise TrendsError(f"Gemini refused the request ({e.code}). Check the API key in the Modal secret "
                                  f"'adobe-ps-trends'. {detail}") from e
            if e.code == 429:
                raise TrendsError("Gemini's free quota is used up for now. Try again in a minute.") from e
            raise TrendsError(f"Gemini error {e.code}: {detail}") from e
    raise TrendsError(f"No Gemini model answered (tried {', '.join(MODELS)}): {last}")


def _post(url: str, body: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse(resp: dict, n: int) -> tuple[list[dict], list[dict]]:
    """-> (trends, sources). Text parts may wrap the JSON in prose or ``` fences; sources come from grounding."""
    cands = resp.get("candidates") or []
    if not cands:
        raise TrendsError("Gemini returned no answer.")
    c = cands[0]
    text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []))
    m = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(m.group(0)) if m else []
    except json.JSONDecodeError:
        items = []
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        t, p = str(it.get("title", "")).strip()[:60], str(it.get("prompt", "")).strip()[:300]
        if not t or not p or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append({"title": t, "why": str(it.get("why", "")).strip()[:220], "prompt": p})
    src, urls = [], set()
    for ch in (c.get("groundingMetadata") or {}).get("groundingChunks", []) or []:
        w = ch.get("web") or {}
        if w.get("uri") and w["uri"] not in urls:
            urls.add(w["uri"])
            src.append({"title": (w.get("title") or "source")[:80], "url": w["uri"]})
    if not out:
        raise TrendsError("Gemini did not return any usable trends. Try again.")
    return out[:n], src[:8]


_CACHE: dict = {}
_LOCK = threading.Lock()


def trends(topic: str = "", photo_words: str = "", n: int = 6, post=None, now: float | None = None) -> dict:
    topic = re.sub(r"\s+", " ", topic or "").strip()[:80]
    photo_words = re.sub(r"\s+", " ", photo_words or "").strip()[:300]
    key = (topic.lower(), photo_words.lower(), n)
    now = time.time() if now is None else now
    ttl = 6 * 3600 if not topic and not photo_words else 3600
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit["at"] < ttl:
            return {**hit["data"], "cached": True}
    prompt = PROMPT.format(
        n=n, can_do=CAN_DO, date=time.strftime("%d %B %Y", time.gmtime(now)),
        topic=f", especially for {topic}" if topic else "",
        photo=(f"\nThe user's photo shows: {photo_words}. Pick trends that would suit this photo." if photo_words else ""))
    items, src = parse(call_gemini(prompt, post=post), n)
    data = {"trends": items, "sources": src, "topic": topic, "for_photo": bool(photo_words),
            "date": time.strftime("%Y-%m-%d", time.gmtime(now))}
    with _LOCK:
        _CACHE[key] = {"at": now, "data": data}
        while len(_CACHE) > 32:
            _CACHE.pop(next(iter(_CACHE)))
    return {**data, "cached": False}
