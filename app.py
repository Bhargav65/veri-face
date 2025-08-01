from flask import Flask, render_template, request, jsonify, send_file
import os, base64, uuid, shutil, re, io, zipfile
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
import json
import time
from concurrent.futures import ProcessPoolExecutor
import warnings

# MongoDB imports
from pymongo import MongoClient
import gridfs

warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")

# ========== ENV, FIREBASE and MONGODB SETUP ==================
ref_encodings_cache = {}
BATCH_SIZE = 30  # Tune this for performance vs memory

load_dotenv() 
token_info = json.loads(os.getenv("GOOGLE_TOKEN_JSON"))
firebase_config = json.loads(os.getenv("FIREBASE_CONFIG_JSON"))

firebase_key = os.getenv("FIREBASE_KEY_JSON")
cred = credentials.Certificate(json.loads(firebase_key))
url = "https://realtimedataapp-ecaf2-default-rtdb.firebaseio.com"
firebase_admin.initialize_app(cred, {'databaseURL': url})

MONGODB_URI = os.getenv("MONGODB_URI")
mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client['faceapp_db']  # You can name as needed
fs = gridfs.GridFS(mongo_db)

def update_firebase_progress(session_id, current, total):
    try:
        ref = db.reference(f"face_match_progress/{session_id}")
        ref.set({
            "current": current,
            "total": total,
            "expiresAt": int(time.time()) + 86400
        })
    except Exception as e:
        print(f"Firebase update failed: {e}")

pillow_heif.register_heif_opener()

app = Flask(__name__)

# ==============================================================

def check_firebase_ready():
    try:
        test_ref = db.reference("readiness_check")
        test_ref.set({"ping": "pong"})
        if test_ref.get()["ping"] != "pong":
            raise Exception("Firebase ping failed")
        test_ref.delete()
        print("✅ Firebase ready")
        return True
    except Exception as e:
        print(f"❌ Firebase not ready: {e}")
        return False

def check_google_drive_ready():
    try:
        service = get_drive_service()
        result = service.files().list(pageSize=1).execute()
        print("✅ Google Drive API ready")
        return True
    except Exception as e:
        print(f"❌ Google Drive not ready: {e}")
        return False

def load_image_any_format(data):
    try:
        image = Image.open(data if isinstance(data, str) else io.BytesIO(data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"❌ Image load error: {e}")
        return None

def extract_face_encodings(img):
    try:
        locations = face_recognition.face_locations(img)
        return face_recognition.face_encodings(img, locations)
    except:
        return []

def generate_session_id():
    return str(uuid.uuid4()) + "_" + str(int(time.time()))

def extract_folder_id(link):
    match = re.search(r'/folders/([a-zA-Z0-9_-]+)', link)
    return match.group(1) if match else None

def get_drive_service():
    creds = Credentials.from_authorized_user_info(token_info, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    return build('drive', 'v3', credentials=creds)

def download_drive_images(folder_id):
    service = get_drive_service()
    query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed=false"
    result = service.files().list(q=query, fields="files(id, name)").execute()
    return result.get('files', [])

def match_one_image(args):
    img_data, file_name, ref_enc = args
    try:
        img = load_image_any_format(img_data)
        encs = extract_face_encodings(img)
        for test_enc in encs:
            distances = face_recognition.face_distance(ref_enc, test_enc)
            if any(d < 0.6 for d in distances):
                return ('matched', file_name, img_data)
        return ('unmatched', file_name, None)
    except Exception as e:
        print(f"Error: {e}")
        return ('unmatched', file_name, None)

# === NEW: MongoDB handling functions ===

def save_reference_image_to_mongodb(session_id, image_bytes, filename):
    # Remove any previous image for this session
    for file_obj in fs.find({"session_id": session_id}):
        fs.delete(file_obj._id)    
    fs.put(image_bytes, filename=filename, session_id=session_id)
    return True

def get_reference_image_from_mongodb(session_id):
    file_obj = fs.find_one({"session_id": session_id})
    if file_obj:
        return file_obj.read()
    else:
        return None

# =========== MODIFY get_ref_encodings ========================
def get_ref_encodings(session_id, _reference_path_ignored=None):
    if session_id in ref_encodings_cache:
        return ref_encodings_cache[session_id]
    # Load image from MongoDB
    img_bytes = get_reference_image_from_mongodb(session_id)
    if img_bytes is None:
        return []
    img_arr = load_image_any_format(img_bytes)
    if img_arr is not None:
        enc = extract_face_encodings(img_arr)
        if enc:
            if len(ref_encodings_cache) > 100:
                ref_encodings_cache.pop(next(iter(ref_encodings_cache)))
            ref_encodings_cache[session_id] = enc
            return enc
    return []

# =============================================================

def compare_faces(reference_path, drive_files, session_id):
    matched, unmatched = [], []
    ref_enc = get_ref_encodings(session_id)
    if not ref_enc:
        return matched, unmatched

    service = get_drive_service()
    total_count = len(drive_files)
    update_firebase_progress(session_id, 0, total_count)

    tasks = []
    for file in drive_files:
        try:
            request = service.files().get_media(fileId=file['id'])
            fh = io.BytesIO()
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            fh.seek(0)
            img_data = fh.read()
            tasks.append((img_data, file['name'], ref_enc))
        except Exception as e:
            print(f"Download error: {e}")
            unmatched.append(file['name'])

    completed = 0
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i + BATCH_SIZE]
            results = list(executor.map(match_one_image, batch))
            for result in results:
                kind, fname, imgdata = result
                if kind == 'matched':
                    matched.append((fname, imgdata))
                else:
                    unmatched.append(fname)
                completed += 1
                update_firebase_progress(session_id, completed, total_count)
    return matched, unmatched

def compare_faces_local(reference_path, uploaded_files, session_id):
    matched, unmatched = [], []
    ref_enc = get_ref_encodings(session_id)
    if not ref_enc:
        return matched, unmatched

    total_count = len(uploaded_files)
    update_firebase_progress(session_id, 0, total_count)
    tasks = []
    for file in uploaded_files:
        try:
            img_data = file.read()
            tasks.append((img_data, file.filename, ref_enc))
        except Exception as e:
            print(f"Read error: {e}")
            unmatched.append(file.filename)
    completed = 0
    with ProcessPoolExecutor() as executor:
        for i in range(0, len(tasks), BATCH_SIZE):
            batch = tasks[i:i + BATCH_SIZE]
            results = list(executor.map(match_one_image, batch))
            for result in results:
                kind, fname, imgdata = result
                if kind == 'matched':
                    matched.append((fname, imgdata))
                else:
                    unmatched.append(fname)
                completed += 1
                update_firebase_progress(session_id, completed, total_count)
    return matched, unmatched

@app.route("/progress/<session_id>")
def progress(session_id):
    try:
        ref = db.reference(f"face_match_progress/{session_id}")
        data = ref.get() or {"current": 0, "total": 0}
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET", "POST"])
def index():
    global progress_data
    if request.method == "POST":
        session_id = request.form.get("session_id") 
    else:
        session_id = generate_session_id()    
    matched, unmatched = [], []
    filepath = None
    progress_data = {"session_id":session_id,"current": 0, "total": 0}
    update_firebase_progress(session_id,0, 0)

    if request.method == "POST":
        source_type = request.form.get("source_select")
        folder_id = extract_folder_id(request.form.get("drive_link")) if source_type == "drive" else None
        local_files = request.files.getlist("local_folder[]") if source_type == "local" else []

        uploaded_file = request.files.get("popup_file")
        ref_img_bytes = None
        ref_img_filename = None
        if uploaded_file and uploaded_file.filename:
            ref_img_bytes = uploaded_file.read()
            ext = os.path.splitext(uploaded_file.filename)[-1]
            ref_img_filename = str(uuid.uuid4()) + ext
        elif request.form.get("popup_webcam_image"):
            try:
                encoded = request.form["popup_webcam_image"].split(",")[1]
                ref_img_bytes = base64.b64decode(encoded)
                ref_img_filename = f"{uuid.uuid4()}.png"
            except Exception as e:
                print(f"Webcam decode error: {e}")

        # Save reference image to MongoDB instead of filesystem
        if ref_img_bytes:
            save_reference_image_to_mongodb(session_id, ref_img_bytes, ref_img_filename)

            if folder_id:
                drive_files = download_drive_images(folder_id)
                matched, unmatched = compare_faces(None, drive_files, session_id)
            elif local_files:
                matched, unmatched = compare_faces_local(None, local_files, session_id)

            if len(matched) == 0:
                return render_template("index.html",
                        firebase_config=firebase_config,
                        session_id="error",
                        matched=[],
                        unmatched=[],
                        error="No face detected in the reference image.") 
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for fname, img_data in matched:
                    clean_name = os.path.basename(fname)
                    zipf.writestr(clean_name, img_data)
            zip_buffer.seek(0)
            for file_obj in fs.find({"session_id": session_id}):
                fs.delete(file_obj._id)
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='matched_photos.zip')
    return render_template("index.html", firebase_config=firebase_config, session_id=session_id, matched=[], unmatched=[])

@app.route("/clean-expired", methods=["POST"])
def clean_expired_sessions():
    try:
        ref = db.reference("face_match_progress")
        data = ref.get()
        now = int(time.time())
        if not data:
            return jsonify({"message": "No sessions to check."}), 200
        deleted = 0
        for session_id, entry in data.items():
            expires_at = entry.get("expiresAt")
            if expires_at and expires_at < now:
                db.reference(f"face_match_progress/{session_id}").delete()
                # Also clean up reference images from MongoDB
                file_obj = fs.find_one({"session_id": session_id})
                if file_obj:
                    fs.delete(file_obj._id)
                print(f"✅ Deleted expired session: {session_id}")
                deleted += 1
        return jsonify({
            "message": f"Expired sessions deleted: {deleted}"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🔎 Checking dependencies before startup...")
    if not check_firebase_ready():
        print("🔥 Firebase failed to initialize -- exiting.")
        exit(1)
    if not check_google_drive_ready():
        print("🔥 Google Drive API failed to initialize -- exiting.")
        exit(1)
    print("🚀 All dependencies ready. Starting app...")
    app.run(debug=True)
