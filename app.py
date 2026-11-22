"""Local chat UI for Ollama.

Run:  python app.py
Open: http://127.0.0.1:5000
"""
import json
import os
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


if __name__ == "__main__":
    init_db()
    print(f"Ollama:  {OLLAMA_URL}")
    print(f"Chat UI: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True)
