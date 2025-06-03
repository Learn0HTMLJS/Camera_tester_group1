import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def calculate_mtf(image_path, edge_angle='vertical'):
    # Загрузка изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
    
    return [
        {'title':'Область с краем', 'data': roi, 'cmap': 'gray', 'type':'flat'},
        {'title':'Функция распределения края (ESF)', 'data': {'x': {'region': esf}}, 'cmap': None, 'type':'graph', 'space': 1, 'grid': False},
        {'title':'Функция распределения линии (LSF)', 'data': {'x': {'region': lsf}}, 'cmap': None, 'type':'graph', 'space': 1, 'grid': False},
        {'title':'Функция распределения линии (MTF)', 'data': {'x': {'label': 'Пространственная частота (циклы/пиксель)', 'region': freq}, 'y':{'label': 'MTF', 'region': mtf}}, 'cmap': None, 'type':'graph', 'space': 2, 'grid': True},
    ]