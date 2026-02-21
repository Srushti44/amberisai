import sqlite3
import json
import hashlib
import os
from config import DB_PATH


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    salt = "amberisai_salt_2026"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    # Users table — parent accounts
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            phone TEXT,
            created_at TEXT DEFAULT (datetime('now'))
        )
    ''')

    # Baby table — linked to user
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS baby (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            nickname TEXT NOT NULL,
            date_of_birth TEXT,
            age_days INTEGER,
            gender TEXT,
            weight_kg REAL,
            blood_group TEXT,
            allergies TEXT DEFAULT '[]',
            medical_notes TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
    ''')

    # Sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS session (
            id TEXT PRIMARY KEY,
            baby_id INTEGER,
            user_id INTEGER,
            timestamp TEXT,
            audio_json TEXT,
            image_json TEXT,
            visualization_url TEXT,
            FOREIGN KEY (baby_id) REFERENCES baby(id),
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("[DB] Initialized successfully.")


# ── AUTH ──────────────────────────────────────────────────────────────────────

def register_user(email, password, full_name, phone=""):
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO user (email, password_hash, full_name, phone) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), hash_password(password), full_name, phone)
        )
        user_id = cursor.lastrowid
        conn.commit()
        return {"success": True, "user_id": user_id}
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Email already registered"}
    finally:
        conn.close()


def login_user(email, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user WHERE email = ?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {"success": False, "error": "Email not found"}
    if row["password_hash"] != hash_password(password):
        return {"success": False, "error": "Incorrect password"}
    return {
        "success": True,
        "user_id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"]
    }


def get_user(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, full_name, phone, created_at FROM user WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


# ── BABY ──────────────────────────────────────────────────────────────────────

def create_baby(nickname, age_days, allergies, user_id=None,
                date_of_birth=None, gender=None, weight_kg=None,
                blood_group=None, medical_notes=None):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO baby 
           (user_id, nickname, date_of_birth, age_days, gender, weight_kg, 
            blood_group, allergies, medical_notes)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, nickname, date_of_birth, age_days, gender, weight_kg,
         blood_group, json.dumps(allergies), medical_notes)
    )
    baby_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return baby_id


def get_baby(baby_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM baby WHERE id = ?", (baby_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["allergies"] = json.loads(d["allergies"]) if d["allergies"] else []
        return d
    return None


def get_babies_by_user(user_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM baby WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["allergies"] = json.loads(d["allergies"]) if d["allergies"] else []
        result.append(d)
    return result


def update_baby(baby_id, **kwargs):
    conn = get_db()
    cursor = conn.cursor()
    if "allergies" in kwargs:
        kwargs["allergies"] = json.dumps(kwargs["allergies"])
    fields = ", ".join([f"{k} = ?" for k in kwargs])
    values = list(kwargs.values()) + [baby_id]
    cursor.execute(f"UPDATE baby SET {fields} WHERE id = ?", values)
    conn.commit()
    conn.close()


# ── SESSIONS ──────────────────────────────────────────────────────────────────

def save_session(session_id, baby_id, timestamp, audio_json=None,
                 image_json=None, visualization_url=None, user_id=None):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO session 
        (id, baby_id, user_id, timestamp, audio_json, image_json, visualization_url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, baby_id, user_id, timestamp,
        json.dumps(audio_json) if audio_json else None,
        json.dumps(image_json) if image_json else None,
        visualization_url
    ))
    conn.commit()
    conn.close()


def get_session(session_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM session WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["audio_json"] = json.loads(d["audio_json"]) if d["audio_json"] else None
        d["image_json"] = json.loads(d["image_json"]) if d["image_json"] else None
        return d
    return None


def get_sessions_by_baby(baby_id, limit=20):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM session WHERE baby_id = ? ORDER BY timestamp DESC LIMIT ?",
        (baby_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["audio_json"] = json.loads(d["audio_json"]) if d["audio_json"] else None
        d["image_json"] = json.loads(d["image_json"]) if d["image_json"] else None
        result.append(d)
    return result


def get_sessions_by_user(user_id, limit=50):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM session WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
        (user_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["audio_json"] = json.loads(d["audio_json"]) if d["audio_json"] else None
        d["image_json"] = json.loads(d["image_json"]) if d["image_json"] else None
        result.append(d)
    return result


def get_user_by_email(email: str):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user WHERE email = ?", (email.lower().strip(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def register_google_user(email: str, full_name: str, google_id: str, picture: str = ''):
    """Register a user who signed in via Google — no password needed."""
    conn = get_db()
    cursor = conn.cursor()
    try:
        # Store google_id as password_hash so account is distinguishable
        cursor.execute(
            "INSERT INTO user (email, password_hash, full_name, phone) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), f"GOOGLE:{google_id}", full_name, '')
        )
        user_id = cursor.lastrowid
        conn.commit()
        return {"success": True, "user_id": user_id}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()