# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import os


def estimate_distortion(image_path):
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if image is None:
        print("Error: Unable to load image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        print("No lines found. Unable to estimate distortion.")
        return

    total_deviation = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
        deviation = min(abs(angle % 90), 90 - abs(angle % 90))
        total_deviation += deviation

    avg_deviation = total_deviation / len(lines)

    print(f"Average line deviation: {avg_deviation:.2f}°")

    if avg_deviation > 1.5:
        x1, y1, x2, y2 = lines[0][0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)

        distortion_type = "Pincushion (positive distortion)" if slope > 0 else "Barrel (negative distortion)"
        print(f"Distortion type: {distortion_type}")
    else:
        print("No significant distortion detected.")


if __name__ == "__main__":
    image_path = input("Enter the path to the image: ").strip()

    if not os.path.exists(image_path):
        print("File not found.")
    else:
        estimate_distortion(image_path)