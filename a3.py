import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 100
NUM_CLASSES = 3  # Change this based on number of people/classes
BATCH_SIZE = 32
EPOCHS = 15

def create_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def capture_training_data():
    for class_id in range(NUM_CLASSES):
        person_name = input(f"Enter name for class {class_id}: ")
        os.makedirs(f"data/{person_name}", exist_ok=True)
        
        print(f"Capturing images for {person_name}. Press 'q' to finish.")
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0
        
        while count < 100:  # Capture 100 images per person
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(f"data/{person_name}/{count}.jpg", face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                count += 1
                print(f"Captured {count} images for {person_name}")
            
            cv2.imshow('Capturing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

def load_data():
    X, y = [], []
    class_names = []
    for idx, person_dir in enumerate(os.listdir('data')):
        class_names.append(person_dir)
        path = os.path.join('data', person_dir)
        for img_file in os.listdir(path):
            img = cv2.imread(os.path.join(path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append(idx)
    
    X = np.array(X) / 255.0
    X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = to_categorical(np.array(y), NUM_CLASSES)
    return X, y, class_names

def main():
    # Capture training data if needed
    if not os.path.exists('data'):
        capture_training_data()
    
    # Train or load model
    model_path = 'multiclass_face_classifier.keras'  # Changed from .h5 to .keras
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = load_model(model_path)
        _, _, class_names = load_data()  # Load class names
    else:
        print("Training new model...")
        X, y, class_names = load_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = create_model()
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))
        model.save(model_path)
    
    # Live prediction with improved performance
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
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
            
            # Use predict_on_batch for better performance
            prediction = model.predict_on_batch(face)[0]
            class_idx = np.argmax(prediction)
            confidence = prediction[class_idx]
            
            # Improved visualization
            label = f"{class_names[class_idx]} ({confidence:.2f})"
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        cv2.imshow('Multi-class Face Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):  # Add option to retrain
            print("Retraining model...")
            cap.release()
            cv2.destroyAllWindows()
            if os.path.exists(model_path):
                os.remove(model_path)
            main()
            return
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()