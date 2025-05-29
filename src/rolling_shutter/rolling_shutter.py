import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_rolling_shutter_pattern(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)

    # Переводим изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Обнаружение границ с помощью Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Используем преобразование Хафа для поиска линий
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

    # Создаем копию изображения для визуализации
    line_image = np.copy(image) * 0

    # Визуализируем найденные линии
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Накладываем линии на оригинальное изображение
    combined = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    
    return {'title': 'Обнаруженные линии (возможный эффект rolling shutter)', 'data': cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), 'cmap': None, 'type': 'flat'}
