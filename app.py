import os, base64, uuid, shutil, re, io, zipfile, logging
from flask import Flask, render_template, request, jsonify, send_file, abort
from PIL import Image
import numpy as np
import face_recognition
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pillow_heif
import firebase_admin
from firebase_admin import credentials, db
from dotenv import load_dotenv
import time
from functools import wraps


# Robust Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

load_dotenv()

firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DB_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MSG_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}

# Initialize Firebase
try:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": os.getenv("FIREBASE_DB_URL")
    })
except Exception as e:
    logging.error("Firebase init failed: %s", e)

pillow_heif.register_heif_opener()

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MATCHED_FOLDER = 'static/matched'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MATCHED_FOLDER, exist_ok=True)

progress_data = {"current": 0, "total": 0}  # Progress

# ---- Input Validation ----
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.heic'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file


def retry(exceptions, tries=3, delay=1, backoff=2):
    def decorator_retry(func):
        @wraps(func)
        def wrapper_retry(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logging.warning(f"{func.__name__} failed with {e}, retrying in {_delay} seconds...")
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            # Last attempt, propagate exception if it fails
            return func(*args, **kwargs)
        return wrapper_retry
    return decorator_retry


def validate_file(file):
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, "Unsupported file type"
    file.seek(0, io.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return False, "File exceeds size limit"
    return True, ""

# ---- Error Handlers ----
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request", "message": str(error)}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500

# ---- Firebase Progress ----
def update_firebase_progress(current, total):
    try:
        ref = db.reference("face_match_progress")
        ref.set({"current": current, "total": total})
    except Exception as e:
        logging.error("Firebase update failed: %s", e)

# ---- Image loading ----
@retry((Exception,), tries=2, delay=1)
def load_image_any_format(data):
    try:
        image = Image.open(data if isinstance(data, str) else io.BytesIO(data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        logging.error(f"Image load error: {e}")
        return None

def extract_face_encodings(img):
    try:
        if img is None:
            return []
        locations = face_recognition.face_locations(img)
        return face_recognition.face_encodings(img, locations)
    except Exception as e:
        logging.error("Encoding error: %s", e)
        return []

def extract_folder_id(link):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', link)
    return match.group(1) if match else None

@retry((Exception,), tries=3, delay=1, backoff=2)
def get_drive_service():
    try:
        creds = Credentials.from_authorized_user_file('token.json', scopes=['https://www.googleapis.com/auth/drive.readonly'])
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        logging.error("GDrive auth error: %s", e)
        raise

@retry((Exception,), tries=3, delay=2, backoff=2)
def download_drive_images(folder_id):
    try:
        service = get_drive_service()
        query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed=false"
        result = service.files().list(q=query, fields="files(id, name)").execute()
        return result.get('files', [])
    except Exception as e:
        logging.error("Drive file list error: %s", e)
        return []

def compare_faces(reference_path, drive_files):
    global progress_data
    matched, unmatched = [], []
    ref_img = load_image_any_format(reference_path)
    ref_enc = extract_face_encodings(ref_img)
    if not ref_enc:
        logging.warning("No face found in reference image")
        return matched, unmatched

    try:
        service = get_drive_service()
    except Exception:
        return matched, drive_files  # All files unmatched if Drive can't connect

    progress_data = {"current": 0, "total": len(drive_files)}
    update_firebase_progress(0, len(drive_files))

    for idx, file in enumerate(drive_files):
        try:
            request = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            img_data = fh.read()
            img = load_image_any_format(img_data)
            encs = extract_face_encodings(img)
            found_match = False
            for test_enc in encs:
                distances = face_recognition.face_distance(ref_enc, test_enc)
                if any(d < 0.6 for d in distances):
                    matched.append((file['name'], img_data))
                    found_match = True
                    break
            if not found_match:
                unmatched.append(file['name'])
        except Exception as e:
            logging.error(f"Error comparing {file['name']}: {e}")
            unmatched.append(file['name'])
        progress_data["current"] = idx + 1
        update_firebase_progress(progress_data["current"], progress_data["total"])

    return matched, unmatched

def compare_faces_local(reference_path, uploaded_files):
    global progress_data
    matched, unmatched = [], []
    ref_img = load_image_any_format(reference_path)
    ref_enc = extract_face_encodings(ref_img)
    if not ref_enc:
        logging.warning("No face found in reference image")
        return matched, unmatched

    progress_data = {"current": 0, "total": len(uploaded_files)}
    update_firebase_progress(0, len(uploaded_files))

    for i, file in enumerate(uploaded_files):
        try:
            file_valid, reason = validate_file(file)
            if not file_valid:
                logging.warning(f"Rejected file {file.filename}: {reason}")
                unmatched.append(file.filename)
                continue
            img_data = file.read()
            img = load_image_any_format(img_data)
            encs = extract_face_encodings(img)
            found_match = False
            for test_enc in encs:
                distances = face_recognition.face_distance(ref_enc, test_enc)
                if any(d < 0.6 for d in distances):
                    matched.append((file.filename, img_data))
                    found_match = True
                    break
            if not found_match:
                unmatched.append(file.filename)
        except Exception as e:
            logging.error("Error comparing %s: %s", file.filename, e)
            unmatched.append(file.filename)
        progress_data["current"] = i + 1
        update_firebase_progress(progress_data["current"], progress_data["total"])

    return matched, unmatched

@app.route("/progress")
def progress():
    return jsonify(progress_data)

@app.route("/", methods=["GET", "POST"])
def index():
    global progress_data
    matched, unmatched = [], []
    filepath = None
    progress_data = {"current": 0, "total": 0}
    update_firebase_progress(0, 0)

    if request.method == "POST":
        shutil.rmtree(UPLOAD_FOLDER, ignore_errors=True)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        source_type = request.form.get("source_select")
        folder_id = extract_folder_id(request.form.get("drive_link")) if source_type == "drive" else None
        local_files = request.files.getlist("local_folder[]") if source_type == "local" else []

        # --- Reference Image Validation ---
        uploaded_file = request.files.get("popup_file")
        if uploaded_file and uploaded_file.filename:
            file_valid, reason = validate_file(uploaded_file)
            if not file_valid:
                logging.warning(f"Rejected reference file: {reason}")
                return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[], error=reason)
            filename = str(uuid.uuid4()) + os.path.splitext(uploaded_file.filename)[-1]
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                uploaded_file.save(filepath)
            except Exception as e:
                logging.error("File save error: %s", e)
                return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[], error="Failed to save uploaded file.")
        elif request.form.get("popup_webcam_image"):
            try:
                encoded = request.form["popup_webcam_image"].split(",")[1]
                image_data = base64.b64decode(encoded)
                filename = f"{uuid.uuid4()}.png"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(image_data)
            except Exception as e:
                logging.error("Webcam decode error: %s", e)
                return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[], error="Webcam image decode failed.")

        if filepath:
            try:
                if folder_id:
                    drive_files = download_drive_images(folder_id)
                    matched, unmatched = compare_faces(filepath, drive_files)
                elif local_files:
                    matched, unmatched = compare_faces_local(filepath, local_files)
                else:
                    return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[], error="No local folder or drive link specified.")

                # In-memory ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for fname, img_data in matched:
                        zipf.writestr(fname, img_data)
                zip_buffer.seek(0)
                return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='matched_photos.zip')
            except Exception as e:
                logging.error("Processing error: %s", e)
                return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[], error="An error occurred during processing.")

    return render_template("sample.html", firebase_config=firebase_config, matched=[], unmatched=[])

# ---- API: Face Match with Google Drive (base64) ----
@app.route("/api/match/drive/base64", methods=["POST"])
def api_match_drive_base64():
    try:
        data = request.get_json(force=True)
        if not data or "reference_image_base64" not in data or "drive_link" not in data:
            return jsonify({"error": "Missing reference_image_base64 or drive_link"}), 400

        try:
            # Decode reference image from base64
            encoded = data["reference_image_base64"].split(",")[-1]
            image_data = base64.b64decode(encoded)
        except Exception:
            return jsonify({"error": "Base64 decode failed"}), 400

        ref_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.png")
        try:
            with open(ref_path, "wb") as f:
                f.write(image_data)
        except Exception as e:
            logging.error("Failed to write ref image: %s", e)
            return jsonify({"error": "Failed saving reference image"}), 500

        folder_id = extract_folder_id(data["drive_link"])
        if not folder_id:
            return jsonify({"error": "Invalid Google Drive folder link"}), 400

        drive_files = download_drive_images(folder_id)
        matched, _ = compare_faces(ref_path, drive_files)

        # ZIP of matched images
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for fname, img_data in matched:
                zipf.writestr(fname, img_data)
        zip_buffer.seek(0)
        return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='matched_photos.zip')
    except Exception as e:
        logging.error("API error: %s", e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
