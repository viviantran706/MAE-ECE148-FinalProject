#!/usr/bin/env python3

"""
colorDection.py

Use an OAK camera with DepthAI v3 and run our red/orange heat detection
on each frame, showing:
- Annotated image with boxes + % heat
- Binary heat mask window
"""

import cv2
import depthai as dai
import numpy as np


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Takes a BGR frame, returns (annotated_frame, heat_mask)
    """

    # 1) Convert from BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) Define color ranges for RED and ORANGE in HSV
    # Hue in OpenCV: [0, 179]
    # Red near 0 and 179, orange ~ 5–30

    # ---- RED ----
    red_lower1 = np.array([0, 130, 80], dtype=np.uint8)
    red_upper1 = np.array([5, 255, 255], dtype=np.uint8)

    red_lower2 = np.array([170, 130, 80], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    # ---- ORANGE ----
    orange_lower = np.array([10, 130, 130], dtype=np.uint8)
    orange_upper = np.array([26, 255, 255], dtype=np.uint8)



    # 3) Masks for red and orange
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

    # 4) Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)

    # 5) Combined heat mask (red + orange)
    heat_mask = cv2.bitwise_or(red_mask, orange_mask)

    # 6) Copy of original frame to draw on
    annotated_frame = frame.copy()

    # 7) RED blobs
    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area < 500:  # ignore small specks
            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi_mask = red_mask[y:y + h, x:x + w]
        total_pixels = roi_mask.size
        hot_pixels = np.count_nonzero(roi_mask)
        heat_percent = (hot_pixels / total_pixels) * 100.0

        # Pink-ish box
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            (255, 100, 255),
            2,
        )

        text = f"Red {heat_percent:.1f}%"
        cv2.putText(
            annotated_frame,
            text,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    # 8) ORANGE blobs
    orange_contours, _ = cv2.findContours(
        orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in orange_contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        roi_mask = orange_mask[y:y + h, x:x + w]
        total_pixels = roi_mask.size
        hot_pixels = np.count_nonzero(roi_mask)
        heat_percent = (hot_pixels / total_pixels) * 100.0

        # Light box for orange
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            (255, 210, 250),
            2,
        )

        text = f"Orange {heat_percent:.1f}%"
        cv2.putText(
            annotated_frame,
            text,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    return annotated_frame, heat_mask


def main():
    # ---------- Build DepthAI v3 pipeline ----------
    with dai.Pipeline() as pipeline:
        # Camera node (unified Camera in v3)
        cam = pipeline.create(dai.node.Camera).build()

        # Request a 640x480 BGR output stream and create its queue
        video_queue = cam.requestOutput((640, 480)).createOutputQueue()

        # Start pipeline on the device
        pipeline.start()
        print("Press 'q' to quit.")

        # ---------- Main loop ----------
        while pipeline.isRunning():
            img_msg = video_queue.get()          # dai.ImgFrame
            frame = img_msg.getCvFrame()         # np.ndarray (BGR)

            annotated_frame, heat_mask = process_frame(frame)

            cv2.imshow("Heat Detection (OAK)", annotated_frame)
            cv2.imshow("Heat Mask", heat_mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
