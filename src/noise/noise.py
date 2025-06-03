import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def interpret_noise_level(noise_type, value):
    """Интерпретирует уровень шума в понятной форме."""
    if noise_type == "impulse":
        if value < 0.01:
            return "Очень слабый импульсный шум (почти нет)"
        elif 0.01 <= value < 0.05:
            return "Слабый импульсный шум"
        elif 0.05 <= value < 0.1:
            return "Умеренный импульсный шум"
        else:
            return "Сильный импульсный шум (требует очистки)"
    
    elif noise_type == "gaussian":
        if value < 5:
            return "Очень слабый гауссов шум (почти нет)"
        elif 5 <= value < 15:
            return "Слабый гауссов шум"
        elif 15 <= value < 30:
            return "Умеренный гауссов шум"
        else:
            return "Сильный гауссов шум (требует очистки)"
    
    return ""

def detect_noise_type(image):
    """Определяет тип шума на изображении."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    
    if hist[0] > 0.01 or hist[-1] > 0.01:
        return "impulse"
    
    if np.std(hist) < 0.005:
        return "gaussian"
    
    return "mixed"

def estimate_impulse_noise(image):
    """Оценка импульсного шума (соль-перец)."""
    filtered = cv2.medianBlur(image, 3)
    diff = cv2.absdiff(image, filtered)
    noise_level = np.mean(diff > 40)
    return noise_level

def estimate_gaussian_noise(image):
    """Оценка гауссова шума."""
    filtered = gaussian_filter(image, sigma=1)
    noise_level = np.std(image - filtered)
    return noise_level

def estimate_mixed_noise(image):
    """Оценка смешанного шума."""
    impulse_part = estimate_impulse_noise(image)
    gaussian_part = estimate_gaussian_noise(image)
    return impulse_part, gaussian_part

def evaluate_noise(image_path):
    """Основная функция оценки с интерпретацией."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    noise_type = detect_noise_type(image)
    
    if noise_type == 'impulse':
        noise = 'импульсный (соль-перец)'
    elif noise_type == 'gaussian':
        noise = 'Гаусов'
    else:
        noise = 'Смешанный'
    
    if noise_type == "impulse":
        filtered = cv2.medianBlur(image, 3)
        level = estimate_impulse_noise(image)
        interpretation = interpret_noise_level("impulse", level)
        filter_title = "После медианного фильтра"
        hist_title = "Гистограмма (пики по краям)"
        noise_level = f"{level:.4f} ({interpretation})"
    
    elif noise_type == "gaussian":
        filtered = gaussian_filter(image, sigma=1)
        level = estimate_gaussian_noise(image)
        interpretation = interpret_noise_level("gaussian", level)
        filter_title = "После гауссова фильтра"
        hist_title = "Размытая гистограмма"
        noise_level = f"{level:.2f} ({interpretation})"
    
    else:
        impulse, gaussian = estimate_mixed_noise(image)
    
    if noise_type == 'impulse' or noise_type == 'gaussian':
        return [
            {'title': filter_title, 'data': filtered, 'cmap': 'gray', 'type': 'flat'},
            {'title': hist_title, 'data': image.ravel(), 'cmap': 'gray', 'type': 'hist'},
            {'title': 'Тип шума', 'data': noise, 'cmap': None, 'type': 'text'},
            {'title': 'Уровень шума (σ)', 'data': noise_level, 'cmap': None, 'type': 'text'},
        ]
    else:
        return [
            {'title': 'Медианный фильтр', 'data': cv2.medianBlur(image, 3), 'cmap': 'gray', 'type': 'flat'},
            {'title': 'Гауссов фильтр', 'data': gaussian_filter(image, sigma=1), 'cmap': 'grap', 'type': 'flat'},
            {'title': 'Импульсная составляющая', 'data': f"{impulse:.4f} ({interpret_noise_level('impulse', impulse)})", 'cmap': None, 'type': 'text'},
            {'title': 'Гауссова составляющая (σ)', 'data': f"{gaussian:.2f} ({interpret_noise_level('gaussian', gaussian)})", 'cmap': None, 'type': 'text'},
            {'title': 'Общая оценка', 'data': f"{'Преобладает импульсный' if impulse > 0.03 else 'Преобладает гауссов' if gaussian > 15 else 'Сбалансированный смешанный'} шум", 'cmap': None, 'type': 'text'},
        ]
