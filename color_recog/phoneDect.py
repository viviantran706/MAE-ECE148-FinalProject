#!/usr/bin/env python3
import cv2
import numpy as np

# -------------------------------
# Fire score -> temperature params
# -------------------------------
T_MIN = 20.0    # ambient-ish (°C)
T_MAX = 900.0   # fully developed room fire (°C, approximate)

W_RED = 1.0
W_ORANGE = 0.5  # orange "softens" the temperature

BLACK_PERSON_THRESH = 0.03  # 3% black pixels => person present


# ---------- temperature from red + orange only ----------

def temperature_from_counts(red_count, orange_count, black_count, total_pixels):
    """
    Compute temperature from red & orange only, ignoring black for temp.

    - effective_area = total_pixels - black_count
    - fire_coverage = (red+orange) / effective_area
    - mix_red, mix_orange are fractions among fire pixels only
    - fire_score = fire_coverage * (W_RED * mix_red + W_ORANGE * mix_orange)
    - T = T_MIN + fire_score * (T_MAX - T_MIN)
    """
    effective_area = max(1, total_pixels - black_count)

    total_fire = red_count + orange_count
    if total_fire <= 0:
        return T_MIN  # no fire

    fire_coverage = total_fire / effective_area  # 0..1
    mix_red = red_count / total_fire
    mix_orange = orange_count / total_fire

    mix_score = W_RED * mix_red + W_ORANGE * mix_orange  # 0.5..1-ish
    fire_score = np.clip(fire_coverage * mix_score, 0.0, 1.0)

    return float(T_MIN + fire_score * (T_MAX - T_MIN))


# ---------- color masks ----------

def get_color_masks(frame_bgr: np.ndarray):
    """
    Return red_mask, orange_mask, black_mask (0/255 uint8).
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # --- Red (two lobes) ---
    red_lower1 = np.array([0, 130, 80], dtype=np.uint8)
    red_upper1 = np.array([5, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 130, 80], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)

    # --- Orange ---
    orange_lower = np.array([10, 130, 130], dtype=np.uint8)
    orange_upper = np.array([26, 255, 255], dtype=np.uint8)

    # --- Black / dark (low value) ---
    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([180, 255, 60], dtype=np.uint8)

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_mask  = cv2.bitwise_or(red_mask1, red_mask2)

    orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)
    black_mask  = cv2.inRange(hsv, black_lower, black_upper)

    kernel = np.ones((3, 3), np.uint8)

    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.dilate(red_mask, kernel, iterations=1)

    orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
    orange_mask = cv2.dilate(orange_mask, kernel, iterations=1)

    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.dilate(black_mask, kernel, iterations=1)

    return red_mask, orange_mask, black_mask


# ---------- detect windows from fire mask ----------

def detect_windows_from_firemask(red_mask: np.ndarray, orange_mask: np.ndarray):
    """
    Use union of red+orange as 'fire mask' and contours to find each window.
    No fixed limit on number of windows.
    """
    fire_mask = cv2.bitwise_or(red_mask, orange_mask)
    kernel = np.ones((5, 5), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 2000:  # ignore tiny blobs
            continue
        aspect = w / float(h)
        if aspect < 0.4 or aspect > 3.0:  # weird skinny shapes
            continue
        rects.append((x, y, w, h))

    if not rects:
        return []

    # Sort *roughly* row-major first
    rects = sorted(rects, key=lambda r: (r[1] + r[3] / 2.0,
                                         r[0] + r[2] / 2.0))
    return rects


def group_rects_into_grid(rects):
    """
    Given a list of rects (x,y,w,h) sorted by y then x,
    group them into rows based on vertical proximity, and
    sort each row by x-center. Returns:

      rows: list of lists of rects, e.g. [[r00,r01,...], [r10,r11,...], ...]
    """
    if not rects:
        return []

    # Compute centers & heights
    centers_y = [y + h / 2.0 for (_, y, _, h) in rects]
    heights   = [h for (_, _, _, h) in rects]
    median_h  = np.median(heights)
    row_thresh = 0.6 * median_h  # same row if within 60% of typical height

    rows = []
    current_row = [rects[0]]
    current_y   = centers_y[0]

    for i in range(1, len(rects)):
        r = rects[i]
        cy = centers_y[i]
        if abs(cy - current_y) <= row_thresh:
            current_row.append(r)
        else:
            # finish old row, start new
            # sort row by x-center
            current_row.sort(key=lambda rr: rr[0] + rr[2] / 2.0)
            rows.append(current_row)
            current_row = [r]
            current_y = cy

    # add last row
    current_row.sort(key=lambda rr: rr[0] + rr[2] / 2.0)
    rows.append(current_row)

    return rows


# ---------- per-frame processing ----------

def process_frame(frame_bgr: np.ndarray):
    """
    Detect each window individually, group them in rows by geometry,
    then compute:
      - temps (flat, one per window, row-major order)
      - persons (flat)
      - temp_matrix (rows x max_cols)
      - person_matrix (same shape)
    Returns:
      annotated_frame, temps, persons, windows_ordered, temp_matrix, person_matrix
    """
    red_mask, orange_mask, black_mask = get_color_masks(frame_bgr)
    annotated = frame_bgr.copy()

    rects = detect_windows_from_firemask(red_mask, orange_mask)
    rows = group_rects_into_grid(rects)  # list of rows, each row is list of rects

    # Flatten rows into row-major window list
    windows = [r for row in rows for r in row]
    n = len(windows)

    temps = np.zeros(n, dtype=np.float32)
    persons = np.zeros(n, dtype=bool)

    num_rows = len(rows)
    num_cols = max(len(row) for row in rows) if rows else 0

    temp_matrix = np.full((num_rows, num_cols), np.nan, dtype=np.float32)
    person_matrix = np.zeros((num_rows, num_cols), dtype=int)

    idx = 0
    for r_idx, row in enumerate(rows):
        for c_idx, (x, y, w, h) in enumerate(row):
            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)

            red_win = red_mask[y:y + h, x:x + w]
            orange_win = orange_mask[y:y + h, x:x + w]
            black_win = black_mask[y:y + h, x:x + w]

            total_pixels = red_win.size
            red_count    = int(np.count_nonzero(red_win))
            orange_count = int(np.count_nonzero(orange_win))
            black_count  = int(np.count_nonzero(black_win))

            # temperature from red+orange only
            T = temperature_from_counts(red_count, orange_count, black_count, total_pixels)

            # person only from black
            person = (black_count / max(1, total_pixels)) >= BLACK_PERSON_THRESH

            temps[idx] = T
            persons[idx] = person

            temp_matrix[r_idx, c_idx] = T
            person_matrix[r_idx, c_idx] = 1 if person else 0

            # annotate
            cx = x + w // 2
            cy = y + h // 2
            label = f"{int(T)}C"
            if person:
                label += " P"
            cv2.putText(
                annotated,
                label,
                (cx - 40, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            idx += 1

    return annotated, temps, persons, windows, temp_matrix, person_matrix


# ---------- main loop ----------

def main():
    cap_index = 0  # change to 1 or 2 if needed
    cap = cv2.VideoCapture(cap_index)

    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam on index {cap_index}. Try 1 or 2.")
        return

    print("[INFO] Webcam mode. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame.")
            break

        annotated, temps, persons, windows, temp_matrix, person_matrix = process_frame(frame)

        n = len(windows)

        # ---- Table: window i -> (temp, person) ----
        print("Window index -> (Temp, Person):")
        for i in range(n):
            print(f"  Window {i+1}: ({temps[i]:.1f} C, {bool(persons[i])})")

        # ---- Adaptive matrix output ----
        print("Temperature matrix (C):")
        print(np.array_str(temp_matrix, precision=1))
        print("Person matrix (1=True):")
        print(person_matrix)
        print("-" * 40)

        cv2.imshow("Per-window Fire Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
