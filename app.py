from flask import Flask, request, jsonify, session, send_file, redirect, url_for
from flask_mysqldb import MySQL
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os, io
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_PATH = os.path.join(BASE_DIR, "data/best_demnet_model (1).keras")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "DEMNET_SECRET"

# 🔥 CORS — REQUIRED FOR GITHUB PAGES
CORS(
    app,
    supports_credentials=True,
    origins=["https://manish81ranjan.github.io"]
)

app.config.update(
    SESSION_COOKIE_SAMESITE="None",
    SESSION_COOKIE_SECURE=True
)

# --------------------------------------------------
# MYSQL (Railway)
# --------------------------------------------------
app.config["MYSQL_HOST"] = os.getenv("MYSQL_HOST")
app.config["MYSQL_USER"] = os.getenv("MYSQL_USER")
app.config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD")
app.config["MYSQL_DB"] = os.getenv("MYSQL_DATABASE")
app.config["MYSQL_PORT"] = int(os.getenv("MYSQL_PORT", 3306))

mysql = MySQL(app)

# --------------------------------------------------
# LOAD MODEL (LAZY)
# --------------------------------------------------
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    img = Image.open(path).convert("L").resize((128, 128))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 128, 128, 1)

def generate_gradcam(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128, 128))
    edges = cv2.Sobel(img, cv2.CV_64F, 1, 1)
    heat = np.uint8(255 * (edges / (edges.max() + 1e-9)))
    heatmap = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.resize(cv2.imread(path), (128, 128))
    overlay = cv2.addWeighted(base, 0.6, heatmap, 0.4, 0)
    out = os.path.join("static", "gradcam.png")
    cv2.imwrite(out, overlay)
    return out

# --------------------------------------------------
# AUTH
# --------------------------------------------------
@app.route("/signup", methods=["POST"])
def signup():
    data = request.json
    cur = mysql.connection.cursor()

    try:
        cur.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (
                data["name"],
                data["email"],
                generate_password_hash(data["password"])
            )
        )
        mysql.connection.commit()
        return jsonify({"status": "success"})
    except:
        mysql.connection.rollback()
        return jsonify({"status": "error", "message": "User exists"})
    finally:
        cur.close()

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    cur = mysql.connection.cursor()

    cur.execute("SELECT id,name,email,password FROM users WHERE email=%s", (data["email"],))
    user = cur.fetchone()
    cur.close()

    if user and check_password_hash(user[3], data["password"]):
        return jsonify({
    "status": "success",
    "user": {
        "id": user[0],
        "name": user[1],
        "email": user[2]
    }
})


    return jsonify({"status": "error"})

@app.route("/logout")
def logout():
    session.clear()
    return jsonify({"status": "logged_out"})

@app.route("/profile")
def profile():
    if "user_id" not in session:
        return jsonify({"logged": False})

    cur = mysql.connection.cursor()
    cur.execute("SELECT name,email FROM users WHERE id=%s", (session["user_id"],))
    user = cur.fetchone()
    cur.close()

    return jsonify({
        "logged": True,
        "name": user[0],
        "email": user[1]
    })

# --------------------------------------------------
# MRI PREDICTION
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    patient = request.form["patient_name"]
    mri_id = request.form["mri_id"]
    file = request.files["mri_image"]

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    img = preprocess_image(path)
    preds = get_model().predict(img)[0]
    idx = int(np.argmax(preds))
    confidence = round(float(preds[idx]) * 100, 2)

    classes = ["Mild", "Moderate", "Normal", "Very Mild"]
    prediction = classes[idx]

    gradcam = generate_gradcam(path)

    cur = mysql.connection.cursor()
    cur.execute("""
        INSERT INTO patient_reports
        (user_id,patient_name,mri_id,prediction,confidence)
        VALUES (%s,%s,%s,%s,%s)
    """, (session["user_id"], patient, mri_id, prediction, confidence))
    mysql.connection.commit()
    cur.close()

    session["report"] = {
        "patient": patient,
        "mri": mri_id,
        "prediction": prediction,
        "confidence": confidence
    }

    return jsonify({
        "label": prediction,
        "confidence": confidence,
        "gradcam_url": url_for("static", filename="gradcam.png")
    })

# --------------------------------------------------
# PDF REPORT
# --------------------------------------------------
@app.route("/download-report")
def download_report():
    if "report" not in session:
        return jsonify({"error": "No report"})

    r = session["report"]
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawCentredString(300, 800, "DEMNET MRI REPORT")

    pdf.setFont("Helvetica", 12)
    y = 740
    for k, v in r.items():
        pdf.drawString(80, y, f"{k.upper()} : {v}")
        y -= 30

    pdf.showPage()
    pdf.drawImage("static/gradcam.png", 100, 300, 400, 400)
    pdf.save()

    buf.seek(0)
    return send_file(buf, download_name="MRI_Report.pdf", as_attachment=True)

# --------------------------------------------------
# HEALTH
# --------------------------------------------------
@app.route("/")
def health():
    return jsonify({"status": "DEMNET backend running"})

# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

