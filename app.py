"""Local chat UI for Ollama.

Run:  python app.py
Open: http://127.0.0.1:5000
"""
import html as html_lib
import json
import os
import re
import sqlite3
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("CHAT_DB", BASE_DIR / "chats.db"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
TITLE_MODEL = os.environ.get("TITLE_MODEL", "").strip()  # optional: fixed model for chat titles
OLLAMA_SITE = os.environ.get("OLLAMA_SITE", "https://ollama.com").rstrip("/")  # where the downloadable-model list comes from

app = Flask(__name__)


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:  # commit / rollback
            yield conn
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id            TEXT PRIMARY KEY,
                title         TEXT NOT NULL DEFAULT 'New chat',
                model         TEXT,
                system_prompt TEXT NOT NULL DEFAULT '',
                title_auto    INTEGER NOT NULL DEFAULT 1,
                created_at    REAL NOT NULL,
                updated_at    REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                model      TEXT,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """
        )
        # upgrade databases created by an older version of this app
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(chats)")}
        if "title_auto" not in cols:
            conn.execute("ALTER TABLE chats ADD COLUMN title_auto INTEGER NOT NULL DEFAULT 1")


def like_pattern(q: str) -> str:
    """Escape LIKE wildcards so the search is a plain substring match."""
    q = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{q}%"


def error(msg: str, status: int = 400):
    return jsonify({"error": msg}), status


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


# --------------------------------------------------------------------------
# Models (proxied from Ollama)
# --------------------------------------------------------------------------
@app.get("/api/models")
def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException as e:
        return error(f"Can't reach Ollama at {OLLAMA_URL} ({e.__class__.__name__}). Is it running?", 502)
    names = sorted(m["name"] for m in r.json().get("models", []))
    return jsonify({"models": names})


# --------------------------------------------------------------------------
# Chats
# --------------------------------------------------------------------------
@app.get("/api/chats")
def list_chats():
    q = request.args.get("q", "").strip()
    with get_db() as conn:
        if q:
            pat = like_pattern(q)
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.model, c.updated_at,
                       (SELECT substr(m.content, max(instr(lower(m.content), lower(?)) - 40, 1), 140)
                          FROM messages m
                         WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\'
                         ORDER BY m.id LIMIT 1) AS snippet
                  FROM chats c
                 WHERE c.title LIKE ? ESCAPE '\\'
                    OR EXISTS (SELECT 1 FROM messages m
                                WHERE m.chat_id = c.id AND m.content LIKE ? ESCAPE '\\')
                 ORDER BY c.updated_at DESC
                """,
                (q, pat, pat, pat),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, model, updated_at, NULL AS snippet FROM chats ORDER BY updated_at DESC"
            ).fetchall()
    return jsonify({"chats": [dict(r) for r in rows]})


@app.post("/api/chats")
def create_chat():
    data = request.get_json(silent=True) or {}
    now = time.time()
    chat_id = uuid.uuid4().hex
    with get_db() as conn:
        conn.execute(
            "INSERT INTO chats (id, title, model, system_prompt, created_at, updated_at) VALUES (?,?,?,?,?,?)",
            (chat_id, "New chat", data.get("model"), data.get("system_prompt", ""), now, now),
        )
    return jsonify({"id": chat_id, "title": "New chat"}), 201


@app.get("/api/chats/<chat_id>")
def get_chat(chat_id):
    with get_db() as conn:
        chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            return error("Chat not found", 404)
        msgs = conn.execute(
            "SELECT role, content, model FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()
    out = dict(chat)
    out["messages"] = [dict(m) for m in msgs]
    return jsonify(out)


@app.patch("/api/chats/<chat_id>")
def update_chat(chat_id):
    data = request.get_json(silent=True) or {}
    fields = {k: data[k] for k in ("title", "model", "system_prompt") if k in data}
    if not fields:
        return error("Nothing to update")
    if "title" in fields:
        fields["title"] = (fields["title"] or "").strip()[:200] or "Untitled"
        fields["title_auto"] = 0  # a manual rename is never overwritten by auto-titling
    sets = ", ".join(f"{k} = ?" for k in fields)
    with get_db() as conn:
        cur = conn.execute(f"UPDATE chats SET {sets} WHERE id = ?", (*fields.values(), chat_id))
        if cur.rowcount == 0:
            return error("Chat not found", 404)
    return jsonify({"ok": True})


@app.delete("/api/chats/<chat_id>")
def delete_chat(chat_id):
    with get_db() as conn:
        conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    return "", 204


# --------------------------------------------------------------------------
# Sending a message: stream Ollama's reply back as newline-delimited JSON
#   {"t": "token text"}   a piece of the reply
#   {"error": "..."}      something went wrong
# The assistant message is saved when the stream ends, including when you
# press Stop (the partial reply is kept).
# --------------------------------------------------------------------------
@app.post("/api/chats/<chat_id>/messages")
def send_message(chat_id):
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return error("Empty message")

    now = time.time()
    with get_db() as conn:
        chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            return error("Chat not found", 404)
        model = data.get("model") or chat["model"]
        if not model:
            return error("No model selected")

        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?,?,?,?)",
            (chat_id, "user", content, now),
        )
        title = chat["title"]
        if title == "New chat":
            title = " ".join(content.split())[:50] or "New chat"
        conn.execute(
            "UPDATE chats SET model = ?, title = ?, updated_at = ? WHERE id = ?",
            (model, title, now, chat_id),
        )
        history = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
        ).fetchall()
        system_prompt = chat["system_prompt"]

    messages = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages += [{"role": m["role"], "content": m["content"]} for m in history]

    def generate():
        parts = []
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={"model": model, "messages": messages, "stream": True},
                stream=True,
                timeout=(5, None),  # 5s to connect, then wait as long as the model needs
            ) as r:
                if r.status_code != 200:
                    try:
                        msg = r.json().get("error", r.text)
                    except ValueError:
                        msg = r.text
                    yield json.dumps({"error": f"Ollama error: {msg}"}) + "\n"
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        yield json.dumps({"error": obj["error"]}) + "\n"
                        return
                    piece = obj.get("message", {}).get("content", "")
                    if piece:
                        parts.append(piece)
                        yield json.dumps({"t": piece}) + "\n"
                    if obj.get("done"):
                        break
        except requests.RequestException as e:
            yield json.dumps({"error": f"Lost connection to Ollama ({e.__class__.__name__})"}) + "\n"
        finally:
            text = "".join(parts)
            if text:
                with get_db() as conn:
                    conn.execute(
                        "INSERT INTO messages (chat_id, role, content, model, created_at) VALUES (?,?,?,?,?)",
                        (chat_id, "assistant", text, model, time.time()),
                    )
                    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (time.time(), chat_id))

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


# --------------------------------------------------------------------------
# Auto-generated chat titles
# The page calls this once after the first exchange of a new chat. The model
# writes a short title; it is only applied if you haven't renamed the chat.
# Set TITLE_MODEL to always use one small model for this (otherwise the
# chat's own model is used, which is already loaded in memory).
# --------------------------------------------------------------------------
def clean_title(raw: str) -> str:
    t = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.S | re.I)
    t = re.sub(r"<think>.*", "", t, flags=re.S | re.I).strip()  # unterminated block
    t = t.splitlines()[0] if t else ""
    t = re.sub(r"^(chat\s+)?title\s*[:\-]\s*", "", t.strip(), flags=re.I)
    t = t.strip(" \t\"'`*#").rstrip(".")
    return t[:60]


@app.post("/api/chats/<chat_id>/title")
def generate_title(chat_id):
    with get_db() as conn:
        chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
        if not chat:
            return error("Chat not found", 404)
        if not chat["title_auto"]:
            return jsonify({"title": chat["title"], "changed": False})
        msgs = conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id LIMIT 2", (chat_id,)
        ).fetchall()
    if len(msgs) < 2:
        return error("Not enough messages yet", 409)

    model = TITLE_MODEL or chat["model"]
    convo = f"User: {msgs[0]['content'][:600]}\n\nAssistant: {msgs[1]['content'][:600]}"
    prompt = [
        {
            "role": "system",
            "content": "You write short titles for chat conversations. Reply with ONLY a title of "
                       "2 to 6 words that captures the topic. No quotes, no trailing punctuation, "
                       "no explanation. Use the same language as the user.",
        },
        {"role": "user", "content": f"Write a title for this conversation:\n\n{convo}"},
    ]

    def ask(extra):
        return requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": prompt, "stream": False,
                  "options": {"temperature": 0.3, "num_predict": 100}, **extra},
            timeout=90,
        )

    try:
        r = ask({"think": False})   # skip "thinking" on reasoning models
        if r.status_code == 400:    # model/Ollama version doesn't know the option
            r = ask({})
        r.raise_for_status()
        raw = r.json().get("message", {}).get("content", "")
    except requests.RequestException as e:
        return error(f"Title generation failed ({e.__class__.__name__})", 502)

    title = clean_title(raw)
    if not title:
        return jsonify({"title": chat["title"], "changed": False})
    with get_db() as conn:
        cur = conn.execute(
            "UPDATE chats SET title = ?, title_auto = 0 WHERE id = ? AND title_auto = 1", (title, chat_id)
        )
    return jsonify({"title": title if cur.rowcount else chat["title"], "changed": bool(cur.rowcount)})


# --------------------------------------------------------------------------
# Model management (thin wrappers around Ollama's own API)
# --------------------------------------------------------------------------
def ollama_msg(r) -> str:
    try:
        return r.json().get("error", r.text)
    except ValueError:
        return r.text


def unreachable(e):
    return error(f"Can't reach Ollama at {OLLAMA_URL} ({e.__class__.__name__}). Is it running?", 502)


@app.get("/api/models/details")
def model_details():
    try:
        tags = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        tags.raise_for_status()
    except requests.RequestException as e:
        return unreachable(e)
    loaded = set()
    try:  # which models are currently held in memory
        ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if ps.ok:
            loaded = {m.get("name") for m in ps.json().get("models", [])}
    except requests.RequestException:
        pass
    out = []
    for m in tags.json().get("models", []):
        d = m.get("details") or {}
        out.append({
            "name": m["name"],
            "size": m.get("size", 0),
            "parameter_size": d.get("parameter_size", ""),
            "quantization": d.get("quantization_level", ""),
            "family": d.get("family", ""),
            "loaded": m["name"] in loaded,
        })
    out.sort(key=lambda x: x["name"])
    return jsonify({"models": out})


# Download a model. Streams newline-delimited JSON:
#   {"status": "...", "total": n|null, "completed": n|null}   progress
#   {"error": "..."}                                          failure
# Cancelling is safe: Ollama resumes a partial download next time.
@app.post("/api/models/pull")
def pull_model():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    if not name:
        return error("Enter a model name, e.g. llama3.2")

    def generate():
        try:
            with requests.post(
                f"{OLLAMA_URL}/api/pull",
                json={"model": name, "stream": True},
                stream=True,
                timeout=(5, 300),
            ) as r:
                if r.status_code != 200:
                    yield json.dumps({"error": ollama_msg(r)}) + "\n"
                    return
                for line in r.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        yield json.dumps({"error": obj["error"]}) + "\n"
                        return
                    yield json.dumps({
                        "status": obj.get("status", ""),
                        "total": obj.get("total"),
                        "completed": obj.get("completed"),
                    }) + "\n"
        except requests.RequestException as e:
            yield json.dumps({"error": f"Lost connection to Ollama ({e.__class__.__name__})"}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson")


@app.delete("/api/models")
def delete_model():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    if not name:
        return error("No model name given")
    try:
        # newer Ollama reads "model", older versions read "name"
        r = requests.delete(f"{OLLAMA_URL}/api/delete", json={"model": name, "name": name}, timeout=30)
    except requests.RequestException as e:
        return unreachable(e)
    if r.status_code != 200:
        return error(ollama_msg(r), 502)
    return jsonify({"ok": True})


@app.post("/api/models/unload")
def unload_model():
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    if not name:
        return error("No model name given")
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json={"model": name, "keep_alive": 0}, timeout=30)
    except requests.RequestException as e:
        return unreachable(e)
    if r.status_code != 200:
        return error(ollama_msg(r), 502)
    return jsonify({"ok": True})


# --------------------------------------------------------------------------
# Browse the Ollama library (models you COULD download)
# Ollama has no official API for this, so we read ollama.com's public pages:
#   /library              -> model names, parameter counts (1b, 3b...), capabilities
#   /library/<name>/tags  -> every variant with its download size
# Results are cached for an hour. If the site layout ever changes or you're
# offline, the endpoints return an error and the page falls back to typing a
# model name by hand, which always works.
# --------------------------------------------------------------------------
_site_cache = {}
SITE_TTL = 3600
NAME_RE = r"[A-Za-z0-9][A-Za-z0-9._-]*"
SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(TB|GB|MB|KB)\b", re.I)


def cached(key, producer):
    hit = _site_cache.get(key)
    if hit and time.time() - hit[0] < SITE_TTL:
        return hit[1]
    value = producer()
    if value:  # never cache an empty/failed parse
        _site_cache[key] = (time.time(), value)
    return value


def fetch_site(path: str) -> str:
    r = requests.get(OLLAMA_SITE + path, timeout=10, headers={"User-Agent": "local-ollama-chat/1.0"})
    r.raise_for_status()
    return r.text


def _text(fragment: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()


def _attr_texts(block: str, attr: str):
    return [_text(x) for x in re.findall(rf"<span[^>]*\b{attr}\b[^>]*>(.*?)</span>", block, re.S)]


def parse_library(page: str):
    models, seen = [], set()
    for block in re.split(r"(?=<li\b[^>]*\bx-test-model\b)", page)[1:]:
        m = re.search(rf'href="/library/({NAME_RE})"', block)
        if not m or m.group(1) in seen:
            continue
        seen.add(m.group(1))
        desc = re.search(r"<p[^>]*max-w-lg[^>]*>(.*?)</p>", block, re.S)
        pulls = _attr_texts(block, "x-test-pull-count")
        models.append({
            "name": m.group(1),
            "sizes": [s for s in _attr_texts(block, "x-test-size") if s],
            "capabilities": [c for c in _attr_texts(block, "x-test-capability") if c],
            "pulls": pulls[0] if pulls else "",
            "description": _text(desc.group(1)) if desc else "",
        })
    if not models:  # markup changed: at least list the names
        for name in re.findall(rf'href="/library/({NAME_RE})"', page):
            if name not in seen:
                seen.add(name)
                models.append({"name": name, "sizes": [], "capabilities": [], "pulls": "", "description": ""})
    return models


def parse_tags(page: str, name: str):
    matches = list(re.finditer(r'href="/library/' + re.escape(name) + r':([^"#?]+)"', page))
    tags, seen, cloud = [], set(), False
    for i, m in enumerate(matches):
        tag = html_lib.unescape(m.group(1))
        if tag in seen:
            continue
        seen.add(tag)
        if "cloud" in tag.lower():  # cloud variants run on Ollama's servers: nothing to download
            cloud = True
            continue
        # the row's text runs until the next DIFFERENT tag (each row's link appears twice in the markup)
        end = next((n.start() for n in matches[i + 1:] if n.group(1) != m.group(1)), len(page))
        s = SIZE_RE.search(_text(page[m.end(): min(end, m.end() + 3000)]))
        tags.append({"tag": tag, "size": f"{s.group(1)} {s.group(2).upper()}" if s else ""})
    if not tags and not cloud:
        return None  # nothing recognisable on the page
    return {"tags": tags, "cloud_only": cloud and not tags}


@app.get("/api/library")
def library():
    try:
        models = cached("library", lambda: parse_library(fetch_site("/library")))
    except requests.RequestException as e:
        return error(f"Can't reach ollama.com ({e.__class__.__name__}). Are you online?", 502)
    if not models:
        return error("Couldn't read the model list from ollama.com (the site layout may have changed).", 502)
    return jsonify({"models": models})


@app.get("/api/library/<name>/tags")
def library_tags(name):
    if not re.fullmatch(NAME_RE, name):
        return error("Bad model name")
    try:
        result = cached("tags:" + name, lambda: parse_tags(fetch_site(f"/library/{name}/tags"), name))
    except requests.RequestException as e:
        return error(f"Can't reach ollama.com ({e.__class__.__name__}). Are you online?", 502)
    if not result:
        return error("Couldn't read the variants for this model.", 502)
    return jsonify(result)


if __name__ == "__main__":
    init_db()
    print(f"Ollama:  {OLLAMA_URL}")
    print(f"Chat UI: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)
