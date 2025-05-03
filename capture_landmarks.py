import cv2
import mediapipe as mp
import csv
import os

# Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Create dataset folder if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# CSV file setup
csv_file = open('dataset/landmarks.csv', mode='a', newline='')
csv_writer = csv.writer(csv_file)

# Classes (only A-Z)
labels = [chr(i) for i in range(ord('A'), ord('Z')+1)]

print("\n Starting Landmark Capture...")
print("Press the letter key for the label you are capturing (A-Z).")
print("Press ESC to quit.\n")

# Webcam capture
cap = cv2.VideoCapture(0)
current_label = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if current_label:
                # Extract 21 (x, y, z) landmark points
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                # Write to CSV with the label
                csv_writer.writerow([current_label] + landmarks)

    cv2.putText(frame, f"Current Label: {current_label if current_label else 'None'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Landmark Capture', frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif 65 <= key <= 90:  # A-Z keys
        current_label = chr(key)
        print(f" Capturing for label: {current_label}")

cap.release()
csv_file.close()
cv2.destroyAllWindows()
print(" Capture session ended. Dataset saved to /dataset/landmarks.csv")
