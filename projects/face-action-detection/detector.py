import cv2
import time
import math
import numpy as np
import mediapipe as mp


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def draw_landmarks(image, face_landmarks, connections, spec):
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        # connections=connections,
        landmark_drawing_spec=spec,
        connection_drawing_spec=spec,
    )


def detect_eye_close(image, point1, point2, text, threshold=5, offset_y=0):
    if point1 and point2:
        dist = math.dist(point1, point2)
        if dist < threshold:
            cv2.putText(
                image,
                text,
                (image.shape[1] - 300, 50 + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )


def detect_mouse_open(image, point1, point2, threshold=10, offset_y=0):
    if point1 and point2:
        dist = math.dist(point1, point2)
        if dist > threshold:
            cv2.putText(
                image,
                "mouse open",
                (image.shape[1] - 300, 50 + offset_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )


def detect():
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)
    stretch_start = time.localtime().tm_sec
    directions_list = []

    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        start = time.time()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False

        results = face_mesh.process(frame)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        h, w, c = frame.shape
        face_2d = []
        face_3d = []

        stretch_time_dict = {"left": 0, "right": 0, "up": 0, "down": 0}
        # adjust the threshold value according to the camera height
        direction_threshold = {"left": -5, "right": 5, "up": 5, "down": -5}

        landmark_dict = {
            "left-eye-upper": 159,
            "left-eye-lower": 145,
            "right-eye-upper": 386,
            "right-eye-lower": 374,
            "mouse-upper": 13,
            "mouse-lower": 14,
        }
        left_eyes_dict = {"upper": None, "lower": None}
        right_eyes_dict = {"upper": None, "lower": None}
        mouse_dict = {"upper": None, "lower": None}

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x, y = landmark.x * w, landmark.y * h

                    if idx == 1:
                        nose_2d = (x, y)
                        nose_3d = (x, y, landmark.z * 3000)

                    if idx == landmark_dict["left-eye-upper"]:
                        left_eyes_dict["upper"] = [x, y]

                    elif idx == landmark_dict["left-eye-lower"]:
                        left_eyes_dict["lower"] = [x, y]

                    elif idx == landmark_dict["right-eye-upper"]:
                        right_eyes_dict["upper"] = [x, y]

                    elif idx == landmark_dict["right-eye-lower"]:
                        right_eyes_dict["lower"] = [x, y]

                    elif idx == landmark_dict["mouse-upper"]:
                        mouse_dict["upper"] = [x, y]

                    elif idx == landmark_dict["mouse-lower"]:
                        mouse_dict["lower"] = [x, y]

                    x = int(x)
                    y = int(y)

                    face_2d.append([x, y])
                    face_3d.append([x, y, landmark.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * w
                skew = 0
                center = (h / 2, w / 2)
                cam_matrix = np.array(
                    [
                        [focal_length, skew, center[0]],
                        [0, focal_length, w / center[1]],
                        [0, 0, 1],
                    ]
                )

                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # reconstruct 3D from 2D image
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    face_3d, face_2d, cam_matrix, dist_matrix
                )

                # rotate rotation vector to matrix
                rmat, jac = cv2.Rodrigues(rotation_vector)
                # calculate euler angle
                angles, mtrxR, matrxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # de-normalize value
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                current_direction = None
                stretch_time = 0

                # See where the user's head tilting
                if y < direction_threshold["left"]:
                    text = "Left"
                    current_direction = "left"

                elif y > direction_threshold["right"]:
                    text = "Right"
                    current_direction = "right"

                elif x < direction_threshold["down"]:
                    text = "Down"
                    current_direction = "down"

                elif x > direction_threshold["up"]:
                    text = "Up"
                    current_direction = "up"

                else:
                    text = "Forward"
                    current_direction = None
                    stretch_start = time.localtime().tm_sec

                if current_direction:
                    stretch_time = time.localtime().tm_sec - stretch_start
                    stretch_time_dict[current_direction] = stretch_time
                    directions_list.append(current_direction)

                if len(directions_list) >= 2:
                    # check direction change
                    if directions_list[-1] != directions_list[-2]:
                        stretch_start = time.localtime().tm_sec
                        directions_list.clear()
                    else:
                        del directions_list[:-2]

                # # Display the nose direction
                # nose_3d_projection, jacobian = cv2.projectPoints(
                #     nose_3d,
                #     rotation_vector,
                #     translation_vector,
                #     cam_matrix,
                #     dist_matrix,
                # )
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 20), int(nose_2d[1] - x * 20))

                cv2.line(frame, p1, p2, (0, 255, 0), 3)

                # Add the text on the image
                cv2.putText(
                    frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"({stretch_time} seconds)",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "x: " + str(np.round(x, 2)),
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "y: " + str(np.round(y, 2)),
                    (20, 250),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "z: " + str(np.round(z, 2)),
                    (20, 300),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                2,
            )

            draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec
            )

            detect_eye_close(
                frame,
                left_eyes_dict["lower"],
                left_eyes_dict["upper"],
                "left right close",
            )
            detect_eye_close(
                frame,
                right_eyes_dict["lower"],
                right_eyes_dict["upper"],
                "right eye close",
                offset_y=50,
            )

            detect_mouse_open(
                frame, mouse_dict["lower"], mouse_dict["upper"], offset_y=100
            )

        cv2.imshow("frame", frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == "__main__":
    detect()
