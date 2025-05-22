import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def calculate_mtf(image_path, edge_angle='vertical'):
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено!")

    # Выбор области с краем
    height, width = img.shape
    if edge_angle == 'vertical':
        # Вертикальный край (берем центральную область)
        x_start = width // 2 - 50
        x_end = width // 2 + 50
        roi = img[:, x_start:x_end]
        esf = roi.mean(axis=1)  # Усреднение по горизонтали
    else:
        # Горизонтальный край
        y_start = height // 2 - 50
        y_end = height // 2 + 50
        roi = img[y_start:y_end, :]
        esf = roi.mean(axis=0)  # Усреднение по вертикали

    # Нормализация ESF
    esf = (esf - esf.min()) / (esf.max() - esf.min() + 1e-10)

    # Расчет LSF (производная ESF)
    lsf = np.gradient(esf)

    # Фурье-преобразование LSF -> MTF
    n = len(lsf)
    mtf = np.abs(fft(lsf))[:n//2]
    mtf = mtf / (mtf.max() + 1e-10)  # Нормировка

    # Частотная ось (в циклах/пиксель)
    freq = np.fft.fftfreq(n, d=1.0)[:n//2]

    # Визуализация
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(roi, cmap='gray')
    plt.title("Область с краем")

    plt.subplot(2, 2, 2)
    plt.plot(esf)
    plt.title("Функция распределения края (ESF)")

    plt.subplot(2, 2, 3)
    plt.plot(lsf)
    plt.title("Функция распределения линии (LSF)")

    plt.subplot(2, 2, 4)
    plt.plot(freq, mtf)
    plt.xlabel("Пространственная частота (циклы/пиксель)")
    plt.ylabel("MTF")
    plt.title("Функция передачи модуляции (MTF)")
    plt.grid()

    plt.tight_layout()
    plt.show()

# Пример использования
# Загрузка изображения (в оттенках серого)
image_path = 'edge_test.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    # Генерация тестового изображения с краем
    test_img = np.zeros((256, 256), dtype=np.uint8)
    test_img[:, 128:] = 255  # Резкий переход чёрный → белый
    cv2.imwrite("edge_test.png", test_img)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Изображение не найдено!")

calculate_mtf(image_path, edge_angle='vertical')