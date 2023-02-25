import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from generate_data import DataGenerator



file_path = f"./hand_gesture_coords_counting.csv"
mode = "test"
# model = joblib.load(
#     "./hand_gesture.joblib"
# )
model = joblib.load(
    "./hand_gesture_counting.joblib"
)
data_generator = DataGenerator(output=file_path, landmark_classes_num=21)


def decode_hand_gesture(landmark):
    return [landmark.x, landmark.y, landmark.z]


def predict_gesture(data, frame):
    X = pd.DataFrame(columns=data_generator.landmarks_classes[1:], data=[data])
    class_name_index = model.predict(X)[0]
    
    proba = model.predict_proba(X)[0][class_name_index]
    if proba > 0.6:
        cv2.putText(
            frame,
            f"{data_generator.classes[class_name_index]} ({round(proba, 2)})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )

    return {"class_name": data_generator.classes[class_name_index], "prob": round(proba, 2)}


def detect():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # mp_holistic = mp.solutions.holistic
    # holistic = mp_holistic.Holistic(
    #     min_detection_confidence=0.5, min_tracking_confidence=0.5
    # )

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)


    while cap.isOpened():
        _, frame = cap.read()
        # frame = cv2.flip(frame, 1)

        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = pose.process(frame)
        # results = holistic.process(frame)
        hand_results = hands.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # hand detection
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                data = [
                    decode_hand_gesture(landmark)
                    for landmark in hand_landmarks.landmark
                ]
                row = np.array(data).flatten().tolist()

                if mode == "train":
                    data_generator.save_gesture_record(hand_landmarks.landmark)
                else:
                    predict_gesture(row, frame)

        cv2.imshow("frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect()
