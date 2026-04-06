import cv2
import requests
import time

# ================= CONFIG =================
SERVER_URL = "http://192.168.0.109:5000/predict" #sesuaikan
INTERVAL = 0.2  # 5 FPS

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

last_time = 0
detections = []

print("[INFO] Tekan ESC untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # ================= SEND FRAME =================
    if current_time - last_time > INTERVAL:
        _, img_encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )

        try:
            files = {
                "image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")
            }

            response = requests.post(
                SERVER_URL,
                files=files,
                timeout=1
            )

            data = response.json()
            detections = data.get("faces", [])

        except Exception as e:
            print("ERROR:", e)
            detections = []

        last_time = current_time

    # ================= DRAW =================
    for face in detections:
        x1, y1, x2, y2 = face["box"]
        name = face["name"]
        conf = face["confidence"]

        if name != "UNKNOWN":
            color = (0, 255, 0)
            label = f"{name} ({conf:.2f})"
        else:
            color = (0, 0, 255)
            label = "UNKNOWN"

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Client Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()