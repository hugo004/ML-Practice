import cv2
import numpy as np
import imutils
import math

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    kernel = np.ones((3, 3), np.uint8)

    roi = frame[100:500, 100:500]
    cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 1)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.dilate(mask, kernel=kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if len(contours) > 0:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        cnt_area = cv2.contourArea(cnt)
        area_ratio = ((hull_area - cnt_area) / cnt_area) * 100

        hull = cv2.convexHull(approx, returnPoints=False)
        hull[::-1].sort(axis=0)
        defects = cv2.convexityDefects(approx, hull)

        # l = no. of defects
        l = 0

        # code for finding no. of defects due to fingers
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(approx[s][0])
                end = tuple(approx[e][0])
                far = tuple(approx[f][0])
                pt = (100, 180)

                # find length of all sides of triangle
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                s = (a + b + c) / 2
                ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

                # distance between point and convex hull
                d = (2 * ar) / a

                # apply cosine rule here
                angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
                if angle <= 90 and d > 30:
                    l += 1
                    cv2.circle(roi, far, 3, [255, 0, 0], -1)

                # draw lines around hand
                cv2.line(roi, start, end, [0, 255, 0], 2)

        l += 1

        # print corresponding gestures which are in their ranges
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if cnt_area < 2000:
                cv2.putText(
                    frame,
                    "Put hand in the box",
                    (0, 50),
                    font,
                    2,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                if area_ratio < 12:
                    cv2.putText(
                        frame, "0", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA
                    )
                elif area_ratio < 17.5:
                    cv2.putText(
                        frame,
                        "Best of luck",
                        (0, 50),
                        font,
                        2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )

                else:
                    cv2.putText(
                        frame, "1", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA
                    )

        elif l == 2:
            cv2.putText(frame, "2", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 3:

            if area_ratio < 27:
                cv2.putText(frame, "3", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, "ok", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 4:
            cv2.putText(frame, "4", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 5:
            cv2.putText(frame, "5", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        elif l == 6:
            cv2.putText(
                frame, "reposition", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA
            )

        else:
            cv2.putText(
                frame, "reposition", (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA
            )

    cv2.imshow("frame", frame)
    cv2.imshow("roi", mask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
