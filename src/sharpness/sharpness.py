import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def evaluate_mtf(mtf_values, freq):
    """Оценка качества резкости по MTF"""
    # Критерии оценки
    mtf50 = freq[np.where(mtf_values >= 0.5)[0][-1]] if np.any(mtf_values >= 0.5) else 0
    mtf30 = freq[np.where(mtf_values >= 0.3)[0][-1]] if np.any(mtf_values >= 0.3) else 0
    mtf10 = freq[np.where(mtf_values >= 0.1)[0][-1]] if np.any(mtf_values >= 0.1) else 0
    
    print("\nОценка резкости:")
    print(f"MTF50: {mtf50:.3f} циклов/пиксель (частота, где контраст падает до 50%)")
    print(f"MTF30: {mtf30:.3f} циклов/пиксель (частота, где контраст падает до 30%)")
    print(f"MTF10: {mtf10:.3f} циклов/пиксель (предел разрешения, 10% контраста)")
    
    # Качественная оценка
    if mtf50 > 0.15:
        return ("Отличная резкость!\n Система сохраняет высокий\n контраст на мелких деталях.")
    elif mtf50 > 0.1:
        return ("Хорошая резкость.\n Приемлемое качество для\n большинства задач.")
    else:
        return ("Низкая резкость.\n Возможны проблемы с фокусировкой\n или качеством оптики.")

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

        # Фильтрация высоких частот
    window = np.hanning(len(mtf))
    mtf_smoothed = mtf * window

    # Оценка результатов
    talmud = evaluate_mtf(mtf_smoothed, freq)
    
    return [
        #{'title':'Область с краем', 'data': roi, 'cmap': 'gray', 'type':'flat'},
        {'title':'Функция распределения края (ESF)', 'data': {'x': {'region': esf}}, 'cmap': None, 'type':'graph', 'space': 1, 'grid': False},
        {'title':'Функция распределения линии (LSF)', 'data': {'x': {'region': lsf}}, 'cmap': None, 'type':'graph', 'space': 1, 'grid': False},
        {'title':'Функция передачи модуляции (MTF)', 'data': {'x': {'label': 'Пространственная частота (циклы/пиксель)', 'region': freq}, 'y':{'label': 'MTF', 'region': mtf}}, 'cmap': None, 'type':'graph', 'space': 2, 'grid': True},
        {'title':'Качественная оценка', 'data': talmud, 'cmap': None, 'type':'text'}
    ]