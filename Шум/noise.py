import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
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
    if image is None:
        raise ValueError("Не удалось загрузить изображение")
    
    noise_type = detect_noise_type(image)
    
    print("\n=== Результаты анализа шума ===")
    print(f"Изображение: {image_path}")
    print(f"Тип шума: {'импульсный (соль-перец)' if noise_type == 'impulse' else 'гауссов' if noise_type == 'gaussian' else 'смешанный'}")
    
    if noise_type == "impulse":
        level = estimate_impulse_noise(image)
        interpretation = interpret_noise_level("impulse", level)
        print(f"Уровень шума: {level:.4f} ({interpretation})")
    
    elif noise_type == "gaussian":
        level = estimate_gaussian_noise(image)
        interpretation = interpret_noise_level("gaussian", level)
        print(f"Уровень шума (σ): {level:.2f} ({interpretation})")
    
    else:
        impulse, gaussian = estimate_mixed_noise(image)
        print("\nСмешанный шум состоит из:")
        print(f"- Импульсная составляющая: {impulse:.4f} ({interpret_noise_level('impulse', impulse)})")
        print(f"- Гауссова составляющая (σ): {gaussian:.2f} ({interpret_noise_level('gaussian', gaussian)})")
        print(f"\nОбщая оценка: {'Преобладает импульсный' if impulse > 0.03 else 'Преобладает гауссов' if gaussian > 15 else 'Сбалансированный смешанный'} шум")
    
    # Визуализация
    plt.figure(figsize=(12, 4))
    plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title("Исходное изображение")
    
    if noise_type == "impulse":
        filtered = cv2.medianBlur(image, 3)
        plt.subplot(132), plt.imshow(filtered, cmap='gray'), plt.title("После медианного фильтра")
        plt.subplot(133), plt.hist(image.ravel(), 256, [0,256]), plt.title("Гистограмма (пики по краям)")
    
    elif noise_type == "gaussian":
        filtered = gaussian_filter(image, sigma=1)
        plt.subplot(132), plt.imshow(filtered, cmap='gray'), plt.title("После гауссова фильтра")
        plt.subplot(133), plt.hist(image.ravel(), 256, [0,256]), plt.title("Размытая гистограмма")
    
    else:
        plt.subplot(132), plt.imshow(cv2.medianBlur(image, 3), cmap='gray'), plt.title("Медианный фильтр")
        plt.subplot(133), plt.imshow(gaussian_filter(image, sigma=1), cmap='gray'), plt.title("Гауссов фильтр")
    
    plt.tight_layout()
    plt.show()

# Пример использования
print("Анализ изображений с разными типами шума:")
evaluate_noise(".\img\lamp.jpg")
#evaluate_noise("S:\code\img\women.jpg")
