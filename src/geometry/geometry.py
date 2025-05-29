# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

def estimate_distortion(image_path):
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

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
        return [{'data': 'No lines found. Unable to estimate distortion', 'cmap': None, 'type':'text'}]

    total_deviation = 0.0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi
        deviation = min(abs(angle % 90), 90 - abs(angle % 90))
        total_deviation += deviation

    avg_deviation = total_deviation / len(lines)

    result = [
       {'title': 'Average line deviation', 'data': f"{avg_deviation:.2f}", 'cmap': None, 'type':'text'}, 
    ]

    if avg_deviation > 1.5:
        x1, y1, x2, y2 = lines[0][0]
        slope = (y2 - y1) / (x2 - x1 + 1e-6)

        distortion_type = "Pincushion (positive distortion)" if slope > 0 else "Barrel (negative distortion)"
        result.append({'title': 'Distortion type', 'data': f"{distortion_type}", 'cmap': None, 'type':'text'})
    else:
        result.append({'data': 'No significant distortion detected.', 'cmap': None, 'type':'text'})
        
    return result