import os
import imaplib
import email
import hashlib
import glob
from email.header import decode_header
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient

# ---------- Tesseract ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------- App ----------
app = Flask(__name__)
CORS(app)

# ---------- Folders ----------
UPLOAD_FOLDER = "uploads"
ORIGINAL_FOLDER = os.path.join("outputs", "original")
TRANSLATED_FOLDER = os.path.join("outputs", "translated")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ORIGINAL_FOLDER, exist_ok=True)
os.makedirs(TRANSLATED_FOLDER, exist_ok=True)

# ---------- Email Settings ----------
IMAP_SERVER = "imap.gmail.com"
EMAIL_ACCOUNT = "studentkmrl@gmail.com"
APP_PASSWORD = "latn oqgl tvew vffa"  # ⚠ better to move into env vars

# ---------- MongoDB ----------
MONGO_URI = "mongodb://localhost:27017/"   # change if using Atlas
client = MongoClient(MONGO_URI)
db = client["ocr_system"]
files_collection = db["files"]

# ---------- AI Model ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- Utility Functions ----------
def compute_file_hash(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

def compute_text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

def is_semantically_similar(new_text, existing_text, threshold=0.80):
    emb1, emb2 = get_embedding(new_text), get_embedding(existing_text)
    similarity = util.cos_sim(emb1, emb2).item()
    return similarity >= threshold, similarity

def is_redundant(new_file_hash, new_text, filename):
    new_text_hash = compute_text_hash(new_text)

    for stored in files_collection.find({}, {"_id": 0}):
        # Exact duplicate file
        if stored.get("file_hash") and stored["file_hash"] == new_file_hash:
            return True, f"❌ Duplicate file detected: {filename}"

        # Same content
        if stored["text_hash"] == new_text_hash:
            return True, f"❌ Same content detected (different filename): {filename}"

        # Slightly updated content
        similar, score = is_semantically_similar(new_text, stored["text"])
        if similar:
            files_collection.delete_one({"filename": stored["filename"]})
            return False, f"♻ Updated version detected (replacing old, sim={score:.2f})"

    return False, None

def save_file_info(filename, file_hash, text):
    doc = {
        "filename": filename,
        "file_hash": file_hash,
        "text_hash": compute_text_hash(text),
        "text": text
    }
    files_collection.insert_one(doc)

def ocr_image(image_path, lang="eng"):
    try:
        return pytesseract.image_to_string(Image.open(image_path), lang=lang).strip()
    except Exception as e:
        return f"OCR Error: {str(e)}"

def ocr_pdf(pdf_path, lang="eng"):
    text_output = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            # First try direct text extraction
            page_text = page.get_text()
            if not page_text.strip():
                pix = page.get_pixmap()
                temp_img = "temp_page.png"
                pix.save(temp_img)
                page_text = ocr_image(temp_img, lang)
                os.remove(temp_img)
            text_output.append(page_text)
        return "\n".join(text_output)
    except Exception as e:
        return f"PDF OCR Error: {str(e)}"

def translate_text(text, src_lang):
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except Exception as e:
        return f"Translation Error: {str(e)}"

def save_text(text, folder, filename):
    path = os.path.join(folder, filename + ".txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path

# ---------- Upload Endpoint ----------
@app.route("/ocr", methods=["POST"])
def ocr_endpoint():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Language detection by filename
    if "malayalam" in file.filename.lower():
        lang_code, tesseract_lang = "ml", "mal+eng"
    elif "tamil" in file.filename.lower():
        lang_code, tesseract_lang = "ta", "tam+eng"
    else:
        lang_code, tesseract_lang = "en", "eng"

    # OCR
    if file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        original_text = ocr_image(filepath, tesseract_lang)
    elif file.filename.lower().endswith(".pdf"):
        original_text = ocr_pdf(filepath, tesseract_lang)
    else:
        return jsonify({"error": f"⚠ Unsupported file format: {file.filename}"}), 400

    # Deduplication
    file_hash = compute_file_hash(filepath)
    redundant, message = is_redundant(file_hash, original_text, file.filename)
    if redundant:
        os.remove(filepath)
        return jsonify({"status": "skipped", "reason": message}), 200

    # Save to MongoDB
    save_file_info(file.filename, file_hash, original_text)

    # Save backups
    save_text(original_text, ORIGINAL_FOLDER, file.filename)

    translated_text = ""
    if lang_code != "en":
        translated_text = translate_text(original_text, src_lang=lang_code)
        save_text(translated_text, TRANSLATED_FOLDER, file.filename)

    return jsonify({
        "status": "processed",
        "filename": file.filename,
        "original": original_text,
        "translated": translated_text,
        "message": message or "✅ New file stored"
    })

# ---------- Email Processing ----------
def fetch_unread_emails():
    notifications = {"processed": 0, "skipped": 0, "unsupported_files": [], "messages": []}
    mail = imaplib.IMAP4_SSL(IMAP_SERVER)
    mail.login(EMAIL_ACCOUNT, APP_PASSWORD)
    mail.select("inbox")

    status, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()

    if not email_ids:
        notifications["messages"].append("✅ No unread emails.")
        mail.logout()
        return notifications

    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)
        subject, encoding = decode_header(msg["Subject"])[0]
        if isinstance(subject, bytes):
            subject = subject.decode(encoding if encoding else "utf-8")

        for part in msg.walk():
            if part.get_content_disposition() == "attachment":
                filename = part.get_filename()
                if not filename:
                    continue
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                with open(filepath, "wb") as f:
                    f.write(part.get_payload(decode=True))

                # Language detection
                if "malayalam" in filename.lower():
                    lang_code, tesseract_lang = "ml", "mal+eng"
                elif "tamil" in filename.lower():
                    lang_code, tesseract_lang = "ta", "tam+eng"
                else:
                    lang_code, tesseract_lang = "en", "eng"

                # OCR
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    original_text = ocr_image(filepath, tesseract_lang)
                elif filename.lower().endswith(".pdf"):
                    original_text = ocr_pdf(filepath, tesseract_lang)
                else:
                    notifications["unsupported_files"].append(filename)
                    continue

                # Deduplication
                file_hash = compute_file_hash(filepath)
                redundant, message = is_redundant(file_hash, original_text, filename)
                if redundant:
                    os.remove(filepath)
                    notifications["skipped"] += 1
                    notifications["messages"].append(message)
                    continue

                # Save to MongoDB
                save_file_info(filename, file_hash, original_text)

                # Save backups
                save_text(original_text, ORIGINAL_FOLDER, filename)
                if lang_code != "en":
                    translated_text = translate_text(original_text, src_lang=lang_code)
                    save_text(translated_text, TRANSLATED_FOLDER, filename)

                notifications["processed"] += 1
                notifications["messages"].append(f"📩 '{subject}' processed: {filename}")

    mail.logout()
    return notifications

@app.route("/check_emails", methods=["GET"])
def check_emails():
    notifications = fetch_unread_emails()
    return notifications

# ---------- Get all stored files ----------
@app.route("/get_files", methods=["GET"])
def get_files():
    files_data = []
    for stored in files_collection.find({}, {"_id": 0}):
        translated_path = os.path.join(TRANSLATED_FOLDER, stored["filename"] + ".txt")
        translated_text = ""
        if os.path.exists(translated_path):
            with open(translated_path, "r", encoding="utf-8") as f:
                translated_text = f.read()

        files_data.append({
            "filename": stored["filename"],
            "status": "processed",
            "original": stored["text"],
            "translated": translated_text
        })
    return jsonify({"files": files_data})

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
