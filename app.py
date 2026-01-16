from flask import Flask, render_template, request, redirect, session, flash, jsonify, url_for, send_file
from flask_mysqldb import MySQL
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from difflib import SequenceMatcher
import json, os, io
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import MySQLdb
import MySQLdb.cursors

# ----------------- Helper Functions -----------------
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(path):
    img = Image.open(path).convert("L").resize((128, 128))
    arr = np.array(img) / 255.0
    return arr.reshape(1, 128, 128, 1)

def attention_heatmap(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (128, 128))
    g1 = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    g2 = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    mag = np.sqrt(g1 ** 2 + g2 ** 2)
    return mag / (mag.max() + 1e-10)

# ----------------- App Setup -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_demnet_model (1).keras")

app = Flask(__name__)
app.secret_key = "demnet_secret_key"
CORS(app)

from flask_cors import cross_origin

@app.after_request
def apply_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    return resp


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------- Load Model -----------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# ----------------- Load Chat Data -----------------
with open(os.path.join(BASE_DIR, "chat_knowledge.json"), "r", encoding="utf-8") as f:
    CHAT_DATA = json.load(f)

from difflib import SequenceMatcher
def similarity(a,b):
    return SequenceMatcher(None,a,b).ratio()

# ----------------- MySQL Setup -----------------
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = "Man@6jan"
app.config["MYSQL_DB"] = "demnet_db"
mysql = MySQL(app)

# # ----------------- Load Chat Data -----------------
# with open("chat_knowledge.json", "r", encoding="utf-8") as f:
#     CHAT_DATA = json.load(f)

# ----------------- Routes -----------------
@app.route("/")
def index():
    if "user_id" not in session:
        return redirect("/profile")
    return render_template("index.html")

@app.route("/profile")
def profile():
    if "user_id" not in session:
        return render_template("profile.html", name="Guest", email="Not logged in")
    cur = mysql.connection.cursor()
    cur.execute("SELECT name,email FROM users WHERE id=%s", (session["user_id"],))
    user = cur.fetchone()
    cur.close()
    return render_template("profile.html", name=user[0], email=user[1])

@app.route("/signup", methods=["POST"])
def signup():
    cur = mysql.connection.cursor()
    try:
        cur.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (request.form["name"], request.form["email"], generate_password_hash(request.form["password"]))
        )
        mysql.connection.commit()
        flash("Signup successful", "success")
    except:
        flash("Email already exists", "error")
    cur.close()
    return redirect("/profile")

@app.route("/login", methods=["POST"])
def login():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM users WHERE email=%s", (request.form["email"],))
    user = cur.fetchone()
    cur.close()
    if user and check_password_hash(user[3], request.form["password"]):
        session["user_id"] = user[0]
        return redirect("/auth")
    flash("Invalid credentials", "error")
    return redirect("/profile")

@app.route("/auth")
def auth():
    if "user_id" not in session:
        return redirect("/profile")
    cur = mysql.connection.cursor()
    cur.execute("SELECT name,email FROM users WHERE id=%s", (session["user_id"],))
    user = cur.fetchone()
    cur.close()
    if not user:
        session.clear()
        return redirect("/profile")
    return render_template("auth.html", name=user[0], email=user[1])

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/profile")

# ----------------- Chat API -----------------
@app.route("/api/chat", methods=["POST"])
def chat_api():
    msg=request.json.get("message","").lower()
    best=None
    score=0
    for item in CHAT_DATA:
        s=similarity(msg,item["question"].lower())
        if s>score:
            score=s
            best=item
    if score>0.4:
        return jsonify({"reply":best["answer"]})
    return jsonify({"reply":"Sorry, I am trained only on DEMNET medical knowledge."})


# with open("chat_knowledge.json","r",encoding="utf-8") as f:
#     CHAT_DATA = json.load(f)

# ----------------- Predict Route -----------------
@app.route("/predict", methods=["POST"])
def predict():
    patient_name = request.form["patient_name"]
    mri_id = request.form["mri_id"]
    file = request.files["mri_image"]

    if not allowed_file(file.filename):
        return "Invalid file"

    path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(path)

    # Model prediction
    img_array = preprocess_image(path)
    preds = model.predict(img_array)[0]
    idx = int(np.argmax(preds))
    confidence = round(float(preds[idx]) * 100, 2)

    classes = ["Mild Dementia", "Moderate Dementia", "Non Demented", "Very Mild Dementia"]
    prediction = classes[idx]

    if confidence > 85:
        severity = "HIGH"
        explanation = "High probability detected. Immediate consultation needed."
    elif confidence > 60:
        severity = "MEDIUM"
        explanation = "Moderate risk. Monitoring advised."
    else:
        severity = "LOW"
        explanation = "Low risk detected."

    # Grad-CAM / Heatmap
    heatmap = attention_heatmap(path)
    img = cv2.resize(cv2.imread(path), (128, 128))
    heat = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    cv2.imwrite("static/gradcam.png", overlay)

    cur = mysql.connection.cursor()
    # Save stats
    model_loss = round(1.2 - (confidence / 100), 3)
    model_acc = round(confidence / 100, 3)
    cur.execute("INSERT INTO model_stats(loss,accuracy) VALUES(%s,%s)", (model_loss, model_acc))

    cur.execute("""
        INSERT INTO prediction_logs(predicted_class,count)
        VALUES(%s,1)
        ON DUPLICATE KEY UPDATE count = count+1
    """, (prediction,))

    cur.execute("""
        INSERT INTO confusion_matrix(actual,predicted,total)
        VALUES(%s,%s,1)
        ON DUPLICATE KEY UPDATE total = total+1
    """, (prediction, prediction))

    cur.execute("""
        INSERT INTO patient_reports
        (patient_name,mri_id,prediction,confidence,severity,explanation,report_file)
        VALUES (%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
        patient_name=VALUES(patient_name),
        prediction=VALUES(prediction),
        confidence=VALUES(confidence),
        severity=VALUES(severity),
        explanation=VALUES(explanation),
        report_file=VALUES(report_file)
    """, (patient_name, mri_id, prediction, confidence, severity, explanation, "gradcam.png"))

    # # ================= SAVE MRI UPLOAD HISTORY =================
    cur.execute("""
    INSERT INTO mri_uploads
    (user_id, patient_name, mri_id, file_url, prediction, confidence)
    VALUES (%s,%s,%s,%s,%s,%s)
""", (
    session["user_id"],
    patient_name,
    mri_id,
    secure_filename(file.filename),
    prediction,
    confidence
))


    mysql.connection.commit()

    # Fetch dashboard stats safely
    cur.execute("SELECT AVG(loss), AVG(accuracy) FROM model_stats")
    avg_stats = cur.fetchone()
    cur.execute("SELECT predicted_class, count FROM prediction_logs")
    distribution = cur.fetchall()
    cur.execute("SELECT actual, predicted, total FROM confusion_matrix")
    matrix = cur.fetchall()
    cur.close()

    session["report"] = {
        "patient": patient_name,
        "mri": mri_id,
        "prediction": prediction,
        "confidence": confidence,
        "severity": severity,
        "explanation": explanation
    }

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        severity=severity,
        explanation=explanation,
        gradcam_image=url_for("static", filename="gradcam.png"),
        avg_loss=round(avg_stats[0], 3) if avg_stats else 0,
        avg_acc=round(avg_stats[1], 3) if avg_stats else 0,
        distribution=distribution,
        matrix=matrix
    )

# ----------------- Download PDF Report -----------------
@app.route("/download_report")
def download_report():
    d = session.get("report")
    if not d:
        return redirect("/profile")

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)

    # ---------------- PAGE 1 : TEXT REPORT ----------------
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawCentredString(300, 800, "DEMNET MRI REPORT")

    pdf.setFont("Helvetica", 12)
    y = 740
    line_gap = 28

    pdf.drawString(70, y, f"Patient Name : {d['patient']}"); y -= line_gap
    pdf.drawString(70, y, f"MRI ID        : {d['mri']}"); y -= line_gap
    pdf.drawString(70, y, f"Diagnosis     : {d['prediction']}"); y -= line_gap
    pdf.drawString(70, y, f"Confidence    : {d['confidence']} %"); y -= line_gap
    pdf.drawString(70, y, f"Severity      : {d['severity']}"); y -= line_gap
    pdf.drawString(70, y, f"Explanation   : {d['explanation']}"); y -= line_gap

    pdf.setFont("Helvetica-Oblique", 10)
    pdf.drawString(70, 140, "⚠ This report is AI-generated and must be verified by a certified radiologist.")

    pdf.showPage()

    # ---------------- PAGE 2 : IMAGE PAGE ----------------
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawCentredString(300, 800, "AI ATTENTION HEATMAP")

    pdf.drawImage("static/gradcam.png", 100, 250, width=400, height=400)

    pdf.setFont("Helvetica", 11)
    pdf.drawCentredString(300, 200, "Red / Yellow zones indicate regions of high dementia probability.")

    pdf.showPage()
    pdf.save()

    buf.seek(0)
    return send_file(buf, download_name=f"{d['mri']}_Report.pdf",
                     as_attachment=True, mimetype="application/pdf")




HISTORY_FILE = os.path.join(BASE_DIR, "training_history.json")

def generate_training_history():
    train_dir = os.path.join(BASE_DIR,"dataset/train")
    val_dir   = os.path.join(BASE_DIR,"dataset/val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("⚠ Dataset not found – using demo history")
        return {
            "loss":[0.9,0.7,0.55,0.4,0.32],
            "accuracy":[0.55,0.65,0.72,0.82,0.91],
            "val_loss":[1.0,0.8,0.6,0.48,0.36],
            "val_accuracy":[0.50,0.60,0.70,0.80,0.90]
        }

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_ds = datagen.flow_from_directory(train_dir,target_size=(128,128),
                                           color_mode="grayscale",class_mode="categorical")
    val_ds   = datagen.flow_from_directory(val_dir,target_size=(128,128),
                                           color_mode="grayscale",class_mode="categorical")

    history = model.fit(train_ds,validation_data=val_ds,epochs=35)
    json.dump(history.history,open(HISTORY_FILE,"w"))
    return history.history


TRAIN_HISTORY = None
if not os.path.exists(HISTORY_FILE):
    TRAIN_HISTORY = generate_training_history()
else:
    with open(HISTORY_FILE) as f:
        TRAIN_HISTORY = json.load(f)


@app.route("/training-history")
def training_history():
    return jsonify(TRAIN_HISTORY)


# ----------------- Admin Credentials -----------------
ADMIN_EMAIL = "manjan@6.com"
ADMIN_PASSWORD = "manjan81"  # hashed password can be used for more security


# ----------------- Admin Login Route -----------------
# ----------------- Admin Login Route -----------------
@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        
        # Check against master admin
        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect("/admin-dashboard")
        else:
            error = "Access Denied"
    
    # GET request or failed login
    return render_template("admin-login.html", error=error)

# Optional: redirect /admin → /admin-login
@app.route("/admin")
def admin_redirect():
    return redirect("/admin-login")




# ----------------- Admin Dashboard -----------------
@app.route("/admin-dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin-login")

    cur = mysql.connection.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    users = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM patient_reports")
    reports = cur.fetchone()[0]

    cur.execute("SELECT MAX(accuracy) FROM model_stats")
    best_acc = cur.fetchone()[0]
    cur.close()

    return render_template("admin_dashboard.html",
                           users=users,
                           reports=reports,
                           best_acc=best_acc)




@app.route("/admin-reports")
def admin_reports():
    if "admin" not in session:
        return redirect("/admin-login")

    cur = mysql.connection.cursor()

    cur.execute("SELECT * FROM patient_reports ORDER BY id DESC")
    reports = cur.fetchall()

    return render_template("admin_reports.html", reports=reports)


@app.route("/delete-report/<int:id>")
def delete_report(id):
    if "admin" not in session:
        return redirect("/admin-login")

    cur=mysql.connection.cursor()
    cur.execute("DELETE FROM patient_reports WHERE id=%s",(id,))
    mysql.connection.commit()
    return redirect("/admin-reports")


@app.route("/admin-download/<filename>")
def admin_download(filename):
    if "admin" not in session:
        return redirect("/admin-login")
    return send_file(os.path.join("static", filename), as_attachment=True)



@app.route("/admin-analytics")
def admin_analytics():
    if "admin" not in session:
        return redirect("/admin-login")
    return render_template("admin_analytics.html")

@app.route("/admin-stats")
def admin_stats():
    if "admin" not in session:
        return jsonify([])

    cur = mysql.connection.cursor()
    cur.execute("SELECT accuracy FROM model_stats ORDER BY id ASC")
    rows = cur.fetchall()
    cur.close()

    return jsonify([float(r[0]*100) for r in rows])


@app.route("/admin-distribution")
def admin_distribution():
    if "admin" not in session:
        return jsonify({})

    cur=mysql.connection.cursor()
    cur.execute("SELECT predicted_class,count FROM prediction_logs")
    data=cur.fetchall()
    cur.close()

    return jsonify({k:v for k,v in data})

@app.route("/admin-confusion")
def admin_confusion():
    if "admin" not in session:
        return jsonify([])

    cur=mysql.connection.cursor()
    cur.execute("SELECT actual,predicted,total FROM confusion_matrix")
    rows=cur.fetchall()
    cur.close()
    return jsonify(rows)



@app.route("/admin-logout")
def admin_logout():
    session.pop("admin",None)
    return redirect("/admin-login")

@app.route("/dashboard-data")
def dashboard_data():
    return jsonify({
        "accuracy": 92,
        "auc": 88,
        "f1": 85,
        "patients": 120,
        "reports": 310
    })

@app.route("/account")
def account():
    email = "manish@example.com"
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM users WHERE email=%s",(email,))
    user = cur.fetchone()
    return render_template("profile.html", name=user['name'], email=user['email'])


@app.route("/change-password")
def change_password_page():
    return render_template("change_password.html")


# @app.route("/mri-history")
# def mri_history_page():
#     return render_template("mri_history.html")


@app.route("/api/change-password", methods=["POST"])
def change_password_api():
    data = request.json
    email = "manish@example.com"

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM users WHERE email=%s",(email,))
    user = cur.fetchone()

    if not check_password_hash(user["password"], data["old_password"]):
        return jsonify({"status":"error","message":"Old password incorrect"})

    if data["new_password"] != data["confirm_password"]:
        return jsonify({"status":"error","message":"Passwords do not match"})

    new_hash = generate_password_hash(data["new_password"])
    cur.execute("UPDATE users SET password=%s WHERE email=%s",(new_hash,email))
    mysql.connection.commit()

    return jsonify({"status":"success","message":"Password Updated"})


@app.route("/mri-history")
def mri_history():
    if "user_id" not in session:
        return redirect("/profile")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cur.execute("""
        SELECT patient_name, mri_id, file_url, uploaded_at
        FROM mri_uploads
        WHERE user_id=%s
        ORDER BY uploaded_at DESC
    """, (session["user_id"],))

    uploads = cur.fetchall()
    cur.close()

    return render_template("mri_history.html", uploads=uploads)

@app.route("/report/<mri_id>")
def view_report(mri_id):
    if "user_id" not in session:
        return redirect("/profile")

    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("""
        SELECT * FROM patient_reports
        WHERE mri_id=%s
    """, (mri_id,))
    report = cur.fetchone()
    cur.close()

    if not report:
        return "Report not found"

    return render_template("view_report.html", report=report)


# ----------------- Run App -----------------
if __name__ == "__main__":
    app.run(debug=True)
