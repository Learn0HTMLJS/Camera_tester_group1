
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

def detect_chromatic_aberration(image_path, window_size=15, threshold=0.7):
    # Загрузка изображения
    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    if image is None:
        print("Error: Unable to load image.")
        return

    # Разделение каналов
    b, g, r = cv2.split(image)

    # Функция для вычисления "разницы" между двумя каналами
    def channel_diff(ch1, ch2, shift=1):
        h, w = ch1.shape
        ch1_cropped = ch1[shift:h-shift, shift:w-shift]
        ch2_shifted = ch2[shift:h-shift, shift:w-shift]
        diff = cv2.absdiff(ch1_cropped, ch2_shifted)
        return np.mean(diff)

    # Анализ пар каналов
    rg_diff = channel_diff(r, g)
    bg_diff = channel_diff(b, g)

    print(f"Red-Green difference: {rg_diff:.2f}")
    print(f"Blue-Green difference: {bg_diff:.2f}")

    total_diff = (rg_diff + bg_diff) / 2

    if total_diff > threshold:
        print(f"Chromatic aberration detected. Intensity: {total_diff:.2f}")
    else:
        print(f"No significant chromatic aberration. Intensity: {total_diff:.2f}")


if __name__ == "__main__":
    image_path = input("Enter the path to the image: ").strip()

    if not os.path.exists(image_path):
        print("File not found.")
    else:
        detect_chromatic_aberration(image_path)