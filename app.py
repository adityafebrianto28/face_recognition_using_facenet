import cv2
import joblib
import numpy as np
import torch
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1

# ================= CONFIG =================
MODEL_PATH = "facenet_fast_model_gpu.pkl"
LABEL_PATH = "label_encoder_gpu.pkl"
CONF_THRESHOLD = 0.7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

# ================= LOAD MODEL =================
saved = joblib.load(MODEL_PATH)
classifier = saved["classifier"]
normalizer = saved["normalizer"]
label_encoder = joblib.load(LABEL_PATH)

# ================= INIT MODEL =================
mtcnn = MTCNN(
    image_size=160,
    margin=20,
    min_face_size=40,
    thresholds=[0.5, 0.6, 0.6],
    factor=0.709,
    keep_all=True,
    device="cpu"  # stabil
)

facenet = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

# ================= FLASK =================
app = Flask(__name__)

@app.route("/")
def home():
    return "Face Recognition API Running"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    frame = cv2.resize(frame, (640, 480))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes, probs = mtcnn.detect(rgb)

    results = []

    if boxes is not None and len(boxes) > 0:
        faces = mtcnn(rgb)

        if faces is not None:
            if len(faces.shape) == 3:
                faces = faces.unsqueeze(0)

            faces = faces.to(DEVICE)

            with torch.no_grad():
                embeddings = facenet(faces).cpu().numpy()

            embeddings = normalizer.transform(embeddings)
            pred_labels = classifier.predict(embeddings)
            pred_probs = classifier.predict_proba(embeddings)

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = [int(v) for v in box]

                best_class = pred_labels[i]
                best_score = float(np.max(pred_probs[i]))

                if best_score >= CONF_THRESHOLD:
                    name = label_encoder.inverse_transform([best_class])[0]
                else:
                    name = "UNKNOWN"

                results.append({
                    "name": name,
                    "confidence": best_score,
                    "box": [x1, y1, x2, y2]
                })

    return jsonify({
        "faces": results,
        "total": len(results)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
