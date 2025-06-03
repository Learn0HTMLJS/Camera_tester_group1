import cv2
import numpy as np
from colour_checker_detection import detect_colour_checkers_segmentation
from colour_checker_detection import plot_image
import matplotlib.pyplot as plt

def find_colorchecker_lib(image_path, visualize=False):
    """
    Находит ColorChecker на изображении с помощью специализированных библиотек
    
    Параметры:
        image_path: путь к изображению
        visualize: флаг визуализации процесса
    
    Возвращает:
        список обнаруженных ColorChecker'ов или None если не найдены
    """
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Используем алгоритм сегментации для поиска
        checkers = detect_colour_checkers_segmentation(image_rgb)
        
        if visualize:
            # Визуализация результатов
            plt.figure(figsize=(12, 6))
            plot_image(image_rgb)
            plt.title("Original Image")
            
            for i, (_, _, swatches) in enumerate(checkers):
                plt.figure(figsize=(12, 6))
                plot_image(swatches)
                plt.title(f"Detected ColorChecker {i+1}")
            
            plt.show()
        
        return checkers
    
    except Exception as e:
        print(f"Ошибка при обнаружении ColorChecker: {e}")
        return None

# Пример использования
detected = find_colorchecker_lib("C:\\Users\\User\\Downloads\\Screenshot 2025-06-03 at 14-23-53 PowerPoint Presentation - Color_Checker.pdf.png", visualize=True)

if detected:
    print(f"Найдено {len(detected)} ColorChecker'ов")
    for i, (checker, segments, swatches) in enumerate(detected):
        print(f"\nColorChecker {i+1}:")
        print(f"Координаты: {segments}")
        print(f"Цвета: {checker.shape}")  # Массив 4x6x3 (RGB значения)
else:
    print("ColorChecker не обнаружен")