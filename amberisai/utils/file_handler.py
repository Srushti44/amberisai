import os
import uuid
from werkzeug.utils import secure_filename
from config import ALLOWED_AUDIO, ALLOWED_IMAGE, UPLOAD_FOLDER


def allowed_audio(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_AUDIO


def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE


def save_upload(file, file_type='audio'):
    """Save uploaded file to uploads/ folder. Returns saved file path."""
    if not file or file.filename == '':
        return None, "No file provided"

    if file_type == 'audio' and not allowed_audio(file.filename):
        return None, f"Invalid audio format. Allowed: {ALLOWED_AUDIO}"

    if file_type == 'image' and not allowed_image(file.filename):
        return None, f"Invalid image format. Allowed: {ALLOWED_IMAGE}"

    original_name = secure_filename(file.filename)
    ext = original_name.rsplit('.', 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, unique_name)

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file.save(save_path)

    return save_path, None


def cleanup_file(file_path):
    """Delete a file from disk."""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"[Cleanup] Could not delete {file_path}: {e}")