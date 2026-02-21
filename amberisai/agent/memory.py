"""
memory.py
=========
Loads and structures context for the AI agent:
  - Baby profile (name, age, allergies)
  - Latest session analysis (audio + image JSON)
  - Session history + trends
  - Chat history (in-memory per session)
"""

import json
import sqlite3
import logging
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

# In-memory chat history: { session_id: [ {role, content}, ... ] }
_chat_histories: dict = defaultdict(list)

# Rate limiting: { baby_id: [timestamps] }
import time
_rate_limits: dict = defaultdict(list)

RATE_LIMIT_MAX = 10       # max requests
RATE_LIMIT_WINDOW = 60    # per 60 seconds


def check_rate_limit(baby_id: str) -> bool:
    """Returns True if request is allowed, False if rate limited."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    key = str(baby_id)

    # Clean old timestamps
    _rate_limits[key] = [t for t in _rate_limits[key] if t > window_start]

    if len(_rate_limits[key]) >= RATE_LIMIT_MAX:
        return False

    _rate_limits[key].append(now)
    return True


def get_db_path() -> str:
    import os
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'amberisai.db'
    )


def get_baby_profile(baby_id: int) -> Optional[dict]:
    """Fetch baby profile from DB."""
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM baby WHERE id = ?", (baby_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        profile = dict(row)
        # Parse allergies JSON
        if profile.get('allergies'):
            try:
                profile['allergies'] = json.loads(profile['allergies'])
            except Exception:
                profile['allergies'] = []
        return profile
    except Exception as e:
        logger.error(f"[Memory] get_baby_profile error: {e}")
        return None


def get_session_data(session_id: str) -> Optional[dict]:
    """Fetch a single session's audio + image analysis."""
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM session WHERE session_id = ?", (session_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        session = dict(row)
        for field in ['audio_json', 'image_json']:
            if session.get(field):
                try:
                    session[field] = json.loads(session[field])
                except Exception:
                    pass
        return session
    except Exception as e:
        logger.error(f"[Memory] get_session_data error: {e}")
        return None


def get_baby_sessions(baby_id: int, limit: int = 10) -> list:
    """Fetch recent sessions for a baby (for trend analysis)."""
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM session WHERE baby_id = ? ORDER BY timestamp DESC LIMIT ?",
            (baby_id, limit)
        )
        rows = cur.fetchall()
        conn.close()
        sessions = []
        for row in rows:
            s = dict(row)
            for field in ['audio_json', 'image_json']:
                if s.get(field):
                    try:
                        s[field] = json.loads(s[field])
                    except Exception:
                        pass
            sessions.append(s)
        return sessions
    except Exception as e:
        logger.error(f"[Memory] get_baby_sessions error: {e}")
        return []


def calculate_trends(sessions: list) -> dict:
    """
    Analyze session history to extract trends.
    Returns a structured trends dict.
    """
    if not sessions:
        return {"summary": "No history available yet.", "conditions": {}, "total_sessions": 0}

    condition_counts = defaultdict(int)
    confidence_sum = defaultdict(float)
    audio_sessions = 0
    image_sessions = 0

    for s in sessions:
        if s.get('audio_json'):
            audio = s['audio_json']
            cond = audio.get('detected_condition') or audio.get('primary_condition', 'unknown')
            conf = audio.get('confidence', 0)
            condition_counts[cond] += 1
            confidence_sum[cond] += conf
            audio_sessions += 1

        if s.get('image_json'):
            image = s['image_json']
            cond = image.get('detected_condition', 'unknown')
            image_sessions += 1
            condition_counts[f"skin:{cond}"] += 1

    # Most common condition
    most_common = max(condition_counts, key=condition_counts.get) if condition_counts else "none"

    # Avg confidence per condition
    avg_confidence = {
        c: round(confidence_sum[c] / condition_counts[c], 2)
        for c in confidence_sum
    }

    # Build summary string
    lines = []
    for cond, count in sorted(condition_counts.items(), key=lambda x: -x[1]):
        pct = round(count / max(audio_sessions, 1) * 100)
        avg_conf = avg_confidence.get(cond, 0)
        lines.append(f"{cond.upper()}: {count}x ({pct}% of sessions, avg {avg_conf*100:.0f}% confidence)")

    return {
        "summary": " | ".join(lines) if lines else "No patterns detected yet.",
        "conditions": dict(condition_counts),
        "most_common": most_common,
        "total_sessions": len(sessions),
        "audio_sessions": audio_sessions,
        "image_sessions": image_sessions,
        "avg_confidence": avg_confidence
    }


def get_chat_history(session_id: str) -> list:
    """Get chat history for a session."""
    return _chat_histories[session_id]


def add_to_chat_history(session_id: str, role: str, content: str):
    """Append a message to chat history. Trim to last 20 messages."""
    _chat_histories[session_id].append({"role": role, "content": content})
    if len(_chat_histories[session_id]) > 20:
        _chat_histories[session_id] = _chat_histories[session_id][-20:]


def build_full_context(
    session_id: Optional[str],
    baby_id: Optional[int],
    query: str
) -> dict:
    """
    Master function — assembles everything the agent needs.
    Returns a context dict used to build the system prompt.
    """
    context = {
        "baby_profile": None,
        "latest_session": None,
        "trends": None,
        "chat_history": [],
        "query": query
    }

    # Chat history
    if session_id:
        context["chat_history"] = get_chat_history(session_id)

    # Latest session analysis
    if session_id:
        context["latest_session"] = get_session_data(session_id)
        # If session has baby_id, use it
        if context["latest_session"] and context["latest_session"].get("baby_id"):
            baby_id = context["latest_session"]["baby_id"]

    # Baby profile + trends
    if baby_id:
        context["baby_profile"] = get_baby_profile(baby_id)
        sessions = get_baby_sessions(baby_id, limit=15)
        context["trends"] = calculate_trends(sessions)

    return context