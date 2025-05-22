import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
import matplotlib.pyplot as plt

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

def extract_color_patches(image, rows=4, cols=6, patch_size=100):
    """Вырезает цветовые патчи с фотографии ColorChecker."""
    patches = []
    h, w = image.shape[:2]
    cell_h, cell_w = h // rows, w // cols
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            patch = image[y1:y2, x1:x2]
            avg_color = np.mean(patch, axis=(0, 1)).astype(int)
            patches.append(avg_color)
    
    return patches

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
    measured_colors = extract_color_patches(image)
    
    # Сравниваем с эталоном
    delta_errors = []
    for (name, ref_rgb), measured_rgb in zip(reference_colors.items(), measured_colors):
        delta_e = calculate_deltaE(ref_rgb, measured_rgb)
        delta_errors.append(delta_e)
        print(f"{name:15} | Эталон: {ref_rgb} | Камера: {measured_rgb} | ΔE: {delta_e:.2f}")
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(delta_errors)), delta_errors)
    plt.axhline(y=2, color='r', linestyle='--', label="ΔE < 2 (Хорошо)")
    plt.axhline(y=5, color='orange', linestyle='--', label="ΔE < 5 (Приемлемо)")
    plt.title("Ошибки цветопередачи (Delta E)")
    plt.xlabel("Патч ColorChecker")
    plt.ylabel("Delta E (CIEDE2000)")
    plt.legend()
    plt.show()

# Пример использования
analyze_color_checker("d:\\OtherFiles\\Computer_vision\\Camera_tester_group1\\fakeColorChecker.webp")