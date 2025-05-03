import cv2
import mediapipe as mp
import numpy as np
import threading
from tensorflow.keras.models import load_model
from tkinter import Tk, Button, Label, Text, END

# Load your trained model
model = load_model("best_landmarks_model.keras")

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Global shared states
current_letter = ""
sentence = ""
running = True

def preprocess_landmarks(landmarks):
    flat_list = []
    for lm in landmarks.landmark:
        flat_list.extend([lm.x, lm.y, lm.z])
    return np.array(flat_list).reshape(1, -1)

def predict_from_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        processed = preprocess_landmarks(landmarks)
        prediction = model.predict(processed, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction)
        return chr(predicted_index + 65), confidence  # Map index to A-Z
    else:
        return None, None

def webcam_loop():
    global current_letter, running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        # Pass the full frame for prediction
        letter, confidence = predict_from_frame(frame)
        if letter and confidence > 0.8:
            current_letter = letter
        else:
            current_letter = ""

        # Display prediction on the full screen
        cv2.putText(frame, f"Letter: {current_letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

        cv2.imshow("ASL Live Sentence Builder", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

def start_webcam():
    global running
    running = True
    threading.Thread(target=webcam_loop).start()

def stop_webcam():
    global running
    running = False

def add_letter():
    global sentence, current_letter
    if current_letter:
        sentence += current_letter
        update_sentence_display()

def add_space():
    global sentence
    sentence += " "
    update_sentence_display()

def delete_last():
    global sentence
    sentence = sentence[:-1]
    update_sentence_display()

def clear_sentence():
    global sentence
    sentence = ""
    update_sentence_display()

def update_sentence_display():
    sentence_display.delete("1.0", END)
    sentence_display.insert(END, sentence)

def main():
    global sentence_display

    root = Tk()
    root.title("ASL Sentence Builder")
    root.geometry("400x400")

    Button(root, text="Start Webcam", command=start_webcam).pack(pady=5)
    Button(root, text="Add Letter", command=add_letter).pack(pady=5)
    Button(root, text="Add Space", command=add_space).pack(pady=5)
    Button(root, text="Delete Last", command=delete_last).pack(pady=5)
    Button(root, text="Clear Sentence", command=clear_sentence).pack(pady=5)
    Button(root, text="Quit", command=lambda: [stop_webcam(), root.destroy()]).pack(pady=15)

    Label(root, text="Sentence Output:", font=("Helvetica", 12)).pack(pady=5)
    sentence_display = Text(root, height=4, width=40)
    sentence_display.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
