import cv2
import numpy as np
from skimage import io, color
from skimage.color import deltaE_ciede2000, rgb2lab
import matplotlib.pyplot as plt
from collections import defaultdict
# Эталонные значения ColorChecker Classic (sRGB значения)
# Патчи: Темные, светлые, основные цвета, пастельные и т. д.
reference_colors = {
    "Dark Skin": (115, 82, 68),
    "Light Skin": (194, 150, 130),
    "Blue Sky": (98, 122, 157),
    "Foliage": (87, 108, 67),
    "Blue Flower": (133, 128, 177),
    "Bluish Green": (103, 189, 170),
    "Orange": (214, 126, 44),
    "Purplish Blue": (80, 91, 166),
    "Moderate Red": (193, 90, 99),
    "Purple": (94, 60, 108),
    "Yellow Green": (157, 188, 64),
    "Orange Yellow": (224, 163, 46),
    "Blue": (56, 61, 150),
    "Green": (70, 148, 73),
    "Red": (175, 54, 60),
    "Yellow": (231, 199, 31),
    "Magenta": (187, 86, 149),
    "Cyan": (8, 133, 161),
    "White": (243, 243, 242),
    "Neutral 8": (200, 200, 200),
    "Neutral 6.5": (160, 160, 160),
    "Neutral 5": (122, 122, 121),
    "Neutral 3.5": (85, 85, 85),
    "Black": (52, 52, 52),
}

def detect_colorchecker(image):
    # Загрузка изображения
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Конвертация в LAB (лучше для анализа цветов)
    image_lab = color.rgb2lab(image_rgb)
    
    # Количество цветов в ColorChecker Classic (24)
    num_colors = 24
    
    # Кластеризация K-means для выделения основных цветов
    pixels = image_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    # Конвертация центров кластеров обратно в RGB
    centers_lab = centers.reshape(-1, 1, 3).astype(np.float32)
    centers_rgb = color.lab2rgb(centers_lab) * 255
    centers_rgb = centers_rgb.reshape(-1, 3).astype(np.uint8)
    
    # Группировка похожих цветов (если нужно)
    color_dict = defaultdict(list)
    for color_rgb in centers_rgb:
        key = tuple(color_rgb)
        color_dict[key].append(key)
    
    # Вывод уникальных цветов
    unique_colors = np.array(list(color_dict.keys()))
      
    return unique_colors

def calculate_deltaE(reference_rgb, measured_rgb):
    """Вычисляет Delta E между эталонным и измеренным цветом."""
    # Конвертация RGB -> LAB (для расчёта Delta E 2000)
    reference_lab = rgb2lab(np.uint8([[reference_rgb]]))[0][0]
    measured_lab = rgb2lab(np.uint8([[measured_rgb]]))[0][0]
    return deltaE_ciede2000(reference_lab, measured_lab)

def analyze_color_checker(image_path):
    """Анализирует цветопередачу камеры по фотографии ColorChecker."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV читает в BGR
    
    # Извлекаем цвета с таблицы
    measured_colors = detect_colorchecker(image)
    
    # Сравниваем с эталоном
    delta_errors = []
    for (name, ref_rgb), measured_rgb in zip(reference_colors.items(), measured_colors):
        delta_e = calculate_deltaE(ref_rgb, measured_rgb)
        delta_errors.append(delta_e)
        print(f"{name:15} | Эталон: {ref_rgb} | Камера: {measured_rgb} | ΔE: {delta_e:.2f}")

    plt.bar(range(len(delta_errors)), delta_errors)
    plt.axhline(y=2, color='r', linestyle='--', label="ΔE < 2 (Хорошо)")
    plt.axhline(y=5, color='orange', linestyle='--', label="ΔE < 5 (Приемлемо)")
    plt.title("Ошибки цветопередачи (Delta E)")
    plt.xlabel("Патч ColorChecker")
    plt.ylabel("Delta E (CIEDE2000)")
    plt.legend()
    plt.show()

    # Результат
    return [
        {
            'title' : 'Ошибки цветопередачи (Delta E 2000)', 
            'data' : [range(len(delta_errors)), delta_errors],
            #'axhline' : {{'y':2}, {'color':'r'}, {'linestyle':'--'}, {"label":"ΔE < 2 (Хорошо)"}},
            #'axhline' : {{'y':5}, {'color':'orange'}, {'linestyle':'--'}, {"label":"ΔE < 5 (Приемлемо)"}},
            'xlabel' : 'Патч ColorChecker',
            'ylabel' : 'Delta E (CIEDE2000)'
        }
    ]
    #plt.figure(figsize=(12, 6))

# Пример использования
#analyze_color_checker("/home/chyn9/Политех/2 семестр/Техническое зрение/Camera_tester_group1/resources/img/color/fake-сolor-сhecker.webp")