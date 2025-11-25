"""
simulate_oak_heat_detection.py

This script simulates the behavior of our OAK-Lite + Raspberry Pi vision pipeline,
but runs entirely on a laptop WITHOUT any special hardware.

Core idea:
- We write a function `process_frame(frame)` that:
    * takes a BGR image (NumPy array from OpenCV)
    * converts it to HSV
    * detects RED and ORANGE pixels
    * builds a "heat mask"
    * finds blobs and calculates how much of each blob is hot (percent heat)
    * draws boxes and labels with the heat percentage

- On the laptop, we can:
    * Load a single test image from disk, OR
    * Use a webcam / phone-as-webcam

- Later, on the Raspberry Pi + OAK-Lite, we will:
    * Replace the part that gets `frame` with a DepthAI pipeline
    * Keep the SAME `process_frame(frame)` function.
"""

import cv2
import numpy as np

# ==============================
# CONFIGURATION FLAGS
# ==============================

# If True, use a webcam (or DroidCam as a virtual webcam).
# If False, just use a single test image from disk.
USE_WEBCAM = False

dir = "/Users/vivia/OneDrive/Documents/GitHub/MAE-ECE196-FinalProject/test_img/"
# Path to a test image (used when USE_WEBCAM = False)
TEST_IMAGE_PATH = dir + "Building_Heat_Ratio_test_pic.jpg"


def process_frame(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Process a single frame to detect red + orange areas and overlay heat info.

    Args:
        frame: BGR image from OpenCV (height x width x 3, uint8)

    Returns:
        annotated_frame: BGR image with boxes + percentage labels drawn
        heat_mask: binary mask where hot pixels (red/orange) are 255, others are 0
    """

    # 1) Convert from BGR (OpenCV default) to HSV
    #    HSV = Hue, Saturation, Value (brightness)
    #    Hue is the color type (red, orange, etc.), which makes color thresholding easier.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2) Define color ranges for RED and ORANGE in HSV (In OpenCV range)
    #
    # Hue range in OpenCV is [0, 179]:
    # - Red usually appears around 0 and 179 (wraps around)
    # - Orange appears around 10–25
    #
    # We also set minimum Saturation (S)  and Value (V) to avoid shadows and dull colors .

    # Lower red range (around 0 degrees)
    red_lower1 = np.array([0, 80, 80], dtype=np.uint8)
    red_upper1 = np.array([12, 255, 255], dtype=np.uint8)

    # Upper red range (around 180 degrees, wrap-around)
    red_lower2 = np.array([165, 80, 80], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    # Orange range (between red and yellow)
    orange_lower = np.array([5, 100, 100], dtype=np.uint8)
    orange_upper = np.array([30, 255, 255], dtype=np.uint8)


    # 3) Create individual masks for red and orange.
    #    cv2.inRange(hsv, lower, upper) produces a binary image:
    #    - 255 where the pixel is within the range
    #    - 0 otherwise

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)


    # 4) Clean up noise separately for red and orange masks.
    kernel = np.ones((5, 5), np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)

    # 5) Combine red + orange into a single "heat" mask for debugging display.
    heat_mask = cv2.bitwise_or(red_mask, orange_mask)
    
    # 6)Make a copy of the original frame to draw on
    annotated_frame = frame.copy()

    # 7) Find and draw RED blobs ("windows").
    red_contours, _ = cv2.findContours(
        red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in red_contours:
        area = cv2.contourArea(contour)
        if area < 500:  # ignore small specks
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Region of interest from the RED mask in this bounding box.
        roi_mask = red_mask[y:y + h, x:x + w]
        total_pixels = roi_mask.size
        hot_pixels = np.count_nonzero(roi_mask)

        # Percentage of pixels in this box that are red.
        heat_percent = (hot_pixels / total_pixels) * 100.0

        # Draw RED box.
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            (0, 0, 25),   # red box (BGR)
            2,
        )

        # Draw black text with percentage.
        text = f"Red {heat_percent:.1f}%"
        cv2.putText(
            annotated_frame,
            text,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),    # black text
            2,
        )

    # 8) Find and draw ORANGE blobs ("windows").
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

        # Percentage of pixels in this box that are red.
        heat_percent = (hot_pixels / total_pixels) * 100.0

        # Draw ORANGE-ish box (border color only; the interior is your original image).
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + w, y + h),
            (0, 10, 50),   # red box (BGR)
            2,
        )

        text = f"Orange {heat_percent:.1f}%"
        cv2.putText(
            annotated_frame,
            text,
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),      # black text
            2,
        )
        
    return annotated_frame, heat_mask



def main():
    """
    Main entry point.

    If USE_WEBCAM is True:
        - Capture frames from the default camera (or DroidCam).
        - Process each frame in real time.

    If USE_WEBCAM is False:
        - Load a single test image from disk.
        - Process it once and wait for a key press.
    """

    if USE_WEBCAM:
        # ===========================
        # LIVE MODE (webcam / phone)
        # ===========================
        cap = cv2.VideoCapture(0)  # try 0, 1, or 2 depending on your setup

        if not cap.isOpened():
            print("Could not open webcam. Try a different index (1 or 2).")
            return

        print("Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                break

            annotated_frame, heat_mask = process_frame(frame)

            # Show both original+annotations and the binary mask
            cv2.imshow("Heat Detection (Simulated OAK)", annotated_frame)
            cv2.imshow("Heat Mask", heat_mask)

            # Wait 1 ms for a key press; if 'q' is pressed, exit loop.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        # ===========================
        # SINGLE IMAGE MODE
        # ===========================
        frame = cv2.imread(TEST_IMAGE_PATH)

        if frame is None:
            print(f"Could not read image: {TEST_IMAGE_PATH}")
            print("Make sure the file exists in this folder.")
            return

        annotated_frame, heat_mask = process_frame(frame)

        cv2.imshow("Heat Detection (Image)", annotated_frame)
        cv2.imshow("Heat Mask", heat_mask)

        print("Press any key in the image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
