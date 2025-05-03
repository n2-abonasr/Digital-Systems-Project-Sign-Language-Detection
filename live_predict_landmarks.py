import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load your trained model
model = load_model('best_landmarks_model.keras')

# Load label encoder (rebuild it manually)
labels = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
encoder = LabelEncoder()
encoder.fit(labels)

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame to avoid mirror image
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

            landmarks = np.array(landmarks).reshape(1, -1)

            prediction = model.predict(landmarks, verbose=0)
            predicted_class = np.argmax(prediction)
            predicted_letter = encoder.inverse_transform([predicted_class])[0]
            confidence = np.max(prediction)

            # Draw box and letter
            cv2.putText(frame, f"{predicted_letter} ({confidence:.2f})", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Sign Language Detection', frame)

    # Exit on ESC
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
