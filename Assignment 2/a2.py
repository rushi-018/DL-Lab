from tensorflow.keras.models import load_model
import cv2
import sys

IMG_SIZE = 100  # Must match your training

# Load model
try:
    model = load_model("face_classifier.h5")
except Exception as e:
    print("Error loading model:", e)
    sys.exit(1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    sys.exit(1)

# Load Haar Cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("Cannot load Haar Cascade xml file")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face / 255.0
        face = face.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        pred = model.predict(face)[0][0]
        label = "Class 1" if pred > 0.5 else "Class 0"
        color = (0, 255, 0) if pred > 0.5 else (0, 0, 255)

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()