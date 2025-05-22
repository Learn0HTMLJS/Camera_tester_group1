import cv2
import numpy as np
import matplotlib.pyplot as plt

def estimate_vignetting(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Преобразование в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    height, width = gray.shape
    center_x, center_y = width // 2, height // 2

    # Создание сетки координат
    Y, X = np.indices((height, width))
    R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    R_max = np.sqrt(center_x**2 + center_y**2)

    # Нормализация радиуса [0..1]
    R_normalized = R / R_max

    # Бины радиусов и средняя яркость в каждом бине
    bins = np.linspace(0, 1, 50)
    bin_indices = np.digitize(R_normalized, bins)
    mean_brightness = [np.mean(gray[bin_indices == i]) for i in range(1, len(bins))]

    # Нормализуем яркость: делаем яркость в центре равной 1
    center_brightness = mean_brightness[0]
    normalized_brightness = np.array(mean_brightness) / center_brightness

    # Метрика виньетирования: разница между центром и краем
    vignetting_level = 1 - normalized_brightness[-1]

    # Визуализация
    plt.figure(figsize=(8, 5))
    plt.plot(bins[1:], normalized_brightness, 'o-', label='Relative Brightness')
    plt.axhline(1, color='r', linestyle='--', label='Center Brightness')
    plt.xlabel('Normalized Distance from Center')
    plt.ylabel('Relative Brightness')
    plt.title(f'Radial Brightness Profile\nVignetting Level: {vignetting_level:.2f}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Возвращаем результат
    return {
        "vignetting_level": vignetting_level,
        "normalized_brightness_profile": normalized_brightness,
        "radius_bins": bins[1:]
    }

# Пример использования:
if __name__ == "__main__":
    result = estimate_vignetting("./img/test1.jpg")
    print(f"Уровень виньетирования: {result['vignetting_level']:.2f}")