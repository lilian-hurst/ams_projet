"""
====================================================
 Serveur Flask — Analyse de radio pulmonaire
====================================================
Lance ce script sur ton Mac :
    pip install flask qrcode pillow
    python qr_code.py

Pepper affiche automatiquement le QR code au démarrage.
Scanne le QR code → envoie une radio → Pepper annonce le résultat.

Mac, téléphone et Pepper doivent être sur le même réseau WiFi.
====================================================
"""
import time

from flask import Flask, request, jsonify, render_template_string
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import io
import socket
import base64
import qrcode
import qi

app = Flask(__name__)

ROBOT_IP   = "192.168.13.202"
ROBOT_PORT = 9559
MODEL_PATH = "model_full.pt"

# ─────────────────────────────────────────────
# 1. Chargement du modèle
# ─────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint  = torch.load(MODEL_PATH, map_location=device)
model_name  = checkpoint["model_name"]
class_names = checkpoint["class_names"]
img_size    = checkpoint["img_size"]
num_classes = checkpoint["num_classes"]

print(f"Modèle  : {model_name}")
print(f"Classes : {class_names}")

def build_model(model_name, num_classes):
    if model_name == "efficientnet_b2":
        model = models.efficientnet_b2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
    return model

model = build_model(model_name, num_classes).to(device)
model.load_state_dict(checkpoint["state_dict"])
model.eval()
print(f"Modèle chargé sur {device}\n")

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 2. Connexion Pepper
# ─────────────────────────────────────────────
pepper_tts     = None
pepper_tablet  = None
pepper_session = None

def connect_pepper():
    global pepper_tts, pepper_tablet, pepper_session
    try:
        pepper_session = qi.Session()
        pepper_session.connect(f"tcp://{ROBOT_IP}:{ROBOT_PORT}")
        pepper_tts    = pepper_session.service("ALTextToSpeech")
        pepper_tablet = pepper_session.service("ALTabletService")
        print("Pepper connecté!")
    except Exception as e:
        print(f"Pepper non disponible : {e}")

connect_pepper()


# ─────────────────────────────────────────────
# 3. Génération du QR code + affichage tablette
# ─────────────────────────────────────────────
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    finally:
        s.close()

def generate_qr_base64(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img    = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def show_qr_on_tablet(url):
    if pepper_tablet is None:
        return
    qr_b64 = generate_qr_base64(url)
    html   = f"""
    <html><head><meta charset="UTF-8">
    <style>
      body {{
        margin: 0; background: #0f172a;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 100vh; font-family: Arial, sans-serif;
      }}
      h1  {{ color: #38bdf8; font-size: 26px; margin-bottom: 16px; }}
      img {{ width: 260px; height: 260px; border-radius: 12px; }}
      p   {{ color: #94a3b8; font-size: 15px; margin-top: 16px; text-align: center; }}
      .url {{ color: #64748b; font-size: 12px; margin-top: 8px; }}
    </style></head>
    <body>
      <h1>Analyse Radio Pulmonaire</h1>
      <img src="data:image/png;base64,{qr_b64}" />
      <p>Scannez ce QR code<br>pour envoyer une radio</p>
      <div class="url">{url}</div>
    </body></html>
    """
    pepper_tablet.showWebview()
    pepper_tablet.loadUrl(
        f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}"
    )

def show_result_on_tablet(label, confidence):
    if pepper_tablet is None:
        return
    if label == "PNEUMONIE":
        color = "#ef4444"
    elif label == "COVID":
        color = "#f97316"
    else:
        color = "#22c55e"
    html  = f"""
    <html><head><meta charset="UTF-8">
    <style>
      body {{
        margin: 0; background: #0f172a;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        height: 100vh; font-family: Arial, sans-serif;
      }}
      .result     {{ font-size: 64px; font-weight: bold; color: {color}; }}
      .confidence {{ font-size: 28px; color: #e2e8f0; margin-top: 12px; }}
      .sub        {{ font-size: 16px; color: #64748b; margin-top: 8px; }}
    </style></head>
    <body>
      <div class="result">{label}</div>
      <div class="confidence">{confidence*100:.1f}% de confiance</div>
      <div class="sub">Scannez à nouveau pour une nouvelle analyse</div>
    </body></html>
    """
    pepper_tablet.showWebview()
    pepper_tablet.loadUrl(
        f"data:text/html;base64,{base64.b64encode(html.encode()).decode()}"
    )


# ─────────────────────────────────────────────
# 4. Validation + Prédiction
# ─────────────────────────────────────────────




def is_valid_image(pil_image):
    gray = pil_image.convert("L")
    arr  = np.array(gray, dtype=np.float32)
    mean = arr.mean()
    if mean < 30:
        return False, "Image trop sombre."
    if mean > 220:
        return False, "Image trop claire."
    kernel  = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(arr, (3, 3))
    lap_var = np.var((windows * kernel).sum(axis=(-1, -2)))
    if lap_var < 20:
        return False, "Image trop floue."
    return True, ""

def predict(pil_image):
    img_tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img_tensor), dim=1).squeeze().tolist()
    idx   = probs.index(max(probs))
    label = class_names[idx]
    if label in ("PNEUMONIA", "PNEUMONIE"):
        result     = "PNEUMONIE"
        confidence = probs[class_names.index("PNEUMONIA")]
    elif label in ("COVID", "COVID-19"):
        result = "COVID"
        confidence = probs[class_names.index("COVID")]
    else:
        result     = "NORMAL"
        confidence = probs[class_names.index("NORMAL")]
    return result, confidence


# ─────────────────────────────────────────────
# 5. Interface web
# ─────────────────────────────────────────────
HTML_PAGE = """
<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Analyse Radio</title>
  <style>
    * { box-sizing:border-box; margin:0; padding:0; }
    body {
      font-family: Arial, sans-serif;
      background: #0f172a; color: #e2e8f0;
      min-height: 100vh;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      padding: 24px;
    }
    h1  { font-size: 22px; color: #38bdf8; margin-bottom: 6px; }
    .sub { font-size: 13px; color: #64748b; margin-bottom: 24px; text-align: center; }
    .card {
      background: #1e293b; border-radius: 16px;
      padding: 28px; width: 100%; max-width: 440px;
      display: flex; flex-direction: column; align-items: center; gap: 18px;
    }

    /* Zone de drop */
    .drop-zone {
      width: 100%; height: 180px;
      border: 2px dashed #334155; border-radius: 12px;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      cursor: pointer; color: #64748b; font-size: 14px; gap: 8px;
      transition: border-color 0.2s, background 0.2s;
      position: relative; overflow: hidden; flex-shrink: 0;
    }
    .drop-zone:hover    { border-color: #38bdf8; }
    .drop-zone.dragover { border-color: #38bdf8; background: #1e3a4a; }
    .drop-zone input    { display: none; }
    .preview-img {
      position: absolute; width: 100%; height: 100%;
      object-fit: contain; border-radius: 10px; background: #0f172a;
    }

    /* Boutons */
    .btn-row { display: flex; gap: 10px; width: 100%; }
    button {
      flex: 1; padding: 13px;
      background: #38bdf8; color: #0f172a;
      font-size: 15px; font-weight: bold;
      border: none; border-radius: 10px;
      cursor: pointer; transition: background 0.2s;
    }
    button:hover    { background: #0ea5e9; }
    button:disabled { background: #334155; color: #64748b; cursor: not-allowed; }
    .btn-secondary {
      background: #334155; color: #e2e8f0;
    }
    .btn-secondary:hover { background: #475569; }

    /* Résultat */
    .result-box { width:100%; padding:20px; border-radius:12px; text-align:center; display:none; }
    .pneumonie  { background:#450a0a; border:2px solid #ef4444; }
    .covid      { background:#431407; border:2px solid #f97316; }
    .normal     { background:#052e16; border:2px solid #22c55e; }
    .result-label { font-size: 40px; font-weight: bold; margin-bottom: 6px; }
    .result-conf  { font-size: 15px; color: #94a3b8; }
    .result-info  { font-size: 12px; color: #475569; margin-top: 6px; }
    .spinner { display:none; font-size:14px; color:#94a3b8; }
    .error   { color:#f87171; font-size:14px; text-align:center; display:none; }
  </style>
</head>
<body>
  <h1>Analyse Radio Pulmonaire</h1>
  <p class="sub">Choisissez une radio — le résultat sera annoncé par Pepper</p>

  <div class="card">

    <!-- Zone de drop (étape 1) -->
    <div class="drop-zone" id="dropZone">
      <span>📁</span>
      <span>Glissez une image ici</span>
      <div class="btn-row" style="margin-top:10px;">
        <button class="btn-secondary" style="font-size:13px; padding:10px;" onclick="document.getElementById('fileStorage').click()">🖼 Choisir une image</button>
      </div>
      <input type="file" id="fileCamera"  accept="image/*" capture="environment">
      <input type="file" id="fileStorage" accept="image/*">
    </div>



    <button id="analyzeBtn" onclick="analyzeImage()" style="display:none; width:100%; padding:13px; background:#38bdf8; color:#0f172a; font-size:15px; font-weight:bold; border:none; border-radius:10px; cursor:pointer;">Analyser</button>
    <div class="spinner" id="spinner">⏳ Analyse en cours...</div>
    <div class="error"   id="errorBox"></div>

    <div class="result-box" id="resultBox">
      <div class="result-label" id="resultLabel"></div>
      <div class="result-conf"  id="resultConf"></div>
      <div class="result-info">Résultat affiché sur la tablette de Pepper</div>
    </div>
  </div>

  <script>
    const dropZone  = document.getElementById("dropZone");
    const spinner   = document.getElementById("spinner");
    const errorBox  = document.getElementById("errorBox");
    const resultBox = document.getElementById("resultBox");
    let currentFile = null;

    function resetDropZone() {
      currentFile = null;
      dropZone.innerHTML = `
        <span>📁</span>
        <span>Glissez une image ici</span>
        <div class="btn-row" style="margin-top:10px;">
          <button class="btn-secondary" style="font-size:13px; padding:10px;" onclick="document.getElementById('fileStorage').click()">🖼 Choisir une image</button>
        </div>
        <input type="file" id="fileCamera"  accept="image/*" capture="environment">
        <input type="file" id="fileStorage" accept="image/*">
      `;
      // Réattacher les listeners sur les nouveaux inputs
      document.getElementById("fileCamera").addEventListener("change",  e => loadFile(e.target.files[0]));
      document.getElementById("fileStorage").addEventListener("change", e => loadFile(e.target.files[0]));
      document.getElementById("analyzeBtn").style.display = "none";
      resultBox.style.display = "none";
      errorBox.style.display  = "none";
      dropZone.onclick = null;
    }

    function loadFile(file) {
      if (!file || !file.type.startsWith("image/")) return;
      currentFile = file;
      const reader = new FileReader();
      reader.onload = e => {
        dropZone.innerHTML = `
          <img class="preview-img" src="${e.target.result}">
          <div onclick="resetDropZone()" style="position:absolute;bottom:8px;right:8px;background:rgba(0,0,0,0.6);border-radius:6px;padding:6px 10px;font-size:13px;color:#e2e8f0;cursor:pointer;z-index:10;">✏️ Changer</div>
        `;
        dropZone.onclick = null;
      };
      reader.readAsDataURL(file);
      document.getElementById("analyzeBtn").style.display = "block";
      resultBox.style.display = "none";
      errorBox.style.display  = "none";
    }

    document.getElementById("fileCamera").addEventListener("change",  e => loadFile(e.target.files[0]));
    document.getElementById("fileStorage").addEventListener("change", e => loadFile(e.target.files[0]));

    dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", e => {
      e.preventDefault();
      dropZone.classList.remove("dragover");
      loadFile(e.dataTransfer.files[0]);
    });

    async function analyzeImage() {
      if (!currentFile) return;
      document.getElementById("analyzeBtn").disabled = true;
      spinner.style.display   = "block";
      errorBox.style.display  = "none";
      resultBox.style.display = "none";

      const formData = new FormData();
      formData.append("image", currentFile);

      try {
        const response = await fetch("/predict", { method: "POST", body: formData });
        const data     = await response.json();
        spinner.style.display = "none";

        if (!response.ok || data.error) {
          errorBox.textContent   = data.error || "Erreur inconnue.";
          errorBox.style.display = "block";
          document.getElementById("analyzeBtn").disabled = false;
          return;
        }

        const label = data.label;
        const conf  = (data.confidence * 100).toFixed(1);
        document.getElementById("resultLabel").textContent = label;
        document.getElementById("resultConf").textContent  = "Confiance : " + conf + "%";
        resultBox.className     = "result-box " + (label === "PNEUMONIE" ? "pneumonie" : label === "COVID" ? "covid" : "normal");
        resultBox.style.display = "block";
        document.getElementById("analyzeBtn").disabled = false;

      } catch(err) {
        spinner.style.display  = "none";
        errorBox.textContent   = "Erreur de connexion au serveur.";
        errorBox.style.display = "block";
        document.getElementById("analyzeBtn").disabled = false;
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "Aucune image reçue."}), 400
    try:
        img_bytes = request.files["image"].read()
        pil_image = Image.open(io.BytesIO(img_bytes))

        # Corriger l'orientation EXIF automatiquement
        try:
            exif = pil_image._getexif()
            if exif:
                orientation = exif.get(274)  # tag 274 = Orientation
                rotations = {3: 180, 6: 270, 8: 90}
                if orientation in rotations:
                    pil_image = pil_image.rotate(rotations[orientation], expand=True)
        except Exception:
            pass  # pas d'EXIF, pas grave

        pil_image = pil_image.convert("RGB")
    except Exception:
        return jsonify({"error": "Impossible de lire l'image."}), 400

    valid, reason = is_valid_image(pil_image)
    if not valid:
        if pepper_tts:
            pepper_tts.say(f"Image invalide. {reason}")
        return jsonify({"error": reason}), 400

    label, confidence = predict(pil_image)
    print(f"Résultat : {label} ({confidence*100:.1f}%)")

    try:
        if pepper_tts:
            if label == "PNEUMONIE":
                pepper_tts.say(f"Attention, détection de pneumonie. Confiance de {confidence*100:.0f} pourcents.")
            elif label == "COVID":
                pepper_tts.say(f"Attention, détection possible de COVID. Confiance de {confidence*100:.0f} pourcents.")
            else:
                pepper_tts.say(f"Résultat normal, aucune anomalie détectée. Confiance de {confidence*100:.0f} pourcents.")
        show_result_on_tablet(label, confidence)
        pepper_tts.say(f"Attention, je suis un robot d'assistance et je ne remplace pas l'avis d'un spécialiste")
        time.sleep(7)
        show_qr_on_tablet(url)
    except Exception as e:
        print(f"Pepper non disponible : {e}")

    return jsonify({"label": label, "confidence": confidence})


# ─────────────────────────────────────────────
# 6. Démarrage
# ─────────────────────────────────────────────
if __name__ == "__main__":
    local_ip = get_local_ip()
    url      = f"http://{local_ip}:5000"

    print(f"\n{'='*50}")
    print(f"  Serveur démarré !")
    print(f"  URL : {url}")
    print(f"{'='*50}\n")

    # Afficher le QR code sur la tablette Pepper
    try:
        if pepper_tablet:
            show_qr_on_tablet(url)
            if pepper_tts:
                pepper_tts.say("Scannez le QR code sur ma tablette pour envoyer une radio à analyser.")
    except Exception as e:
        print(f"Tablette non disponible : {e}")
        print("Le serveur fonctionne quand même, Pepper ne sera pas utilisé.")

    app.run(host="0.0.0.0", port=5000, debug=False)