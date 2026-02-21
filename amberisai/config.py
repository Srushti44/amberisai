import os

DB_PATH = os.path.join("data", "amberisai.db")

ALLOWED_AUDIO = {'wav', 'mp3'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

UPLOAD_FOLDER = "uploads"
VISUALS_FOLDER = "static/visuals"