import cv2
import csv
import pathlib
import numpy as np

from datetime import datetime
from typing import TypedDict, List


class Landmark(TypedDict):
    x: float
    y: float
    z: float


class DataGenerator(object):
    def __init__(self, output: str, landmark_classes_num: int) -> None:
        # self.classes = ["like", "unlike", "ok", "stop", "money", "unknown"]
        self.classes = ["1", "2", "3", "4", "5", "unknown"]

        self.output = output

        # init landmarks columns name
        self.landmarks_classes = ["class"]
        for i in range(landmark_classes_num):
            self.landmarks_classes += [
                "x{}".format(i),
                "y{}".format(i),
                "z{}".format(i),
            ]

    def save_gesture_record(self, landmarks: List[Landmark]):

        k = cv2.waitKey(1)

        class_name = None
        landmark_data = [self.decode_hand_gesture(landmark) for landmark in landmarks]
        row: List[float] = np.array(landmark_data).flatten().tolist()

        if k == ord("1"): 
            class_name = self.classes[0]
        elif k == ord("2"): 
            class_name = self.classes[1]
        elif k == ord("3"): 
            class_name = self.classes[2]
        elif k == ord("4"): 
            class_name = self.classes[3]
        elif k == ord("5"):
            class_name = self.classes[4]
        elif k == 32:  # press space, 'unknown' always located in last
            class_name = self.classes[-1]

        if class_name:
            row.insert(0, class_name)
            self.insert_csv_data(row)
            print(
                "[{}]: save {} class data".format(
                    datetime.now().strftime("%H:%M:%S"), class_name
                )
            )

    def decode_hand_gesture(self, landmark: Landmark) -> List[float]:
        return [landmark.x, landmark.y, landmark.z]

    def insert_csv_data(self, row: List[List[float]]):
        # init file if not exists
        file_path = pathlib.Path(self.output)
        if file_path.is_file() is not True:
            with open(self.output, mode="w", newline="") as f:
                csv_writer = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                csv_writer.writerow(self.landmarks_classes)

        with open(self.output, mode="a", newline="") as f:
            csv_writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(row)
