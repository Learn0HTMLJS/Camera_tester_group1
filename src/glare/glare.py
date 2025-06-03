import cv2
import numpy as np

def detect_glare_with_otsu(image_path, detail_thresh=5):
    # Загрузка изображения и перевод в оттенки серого
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Оператор Лапласа для карты детализации
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    detail_map = np.abs(laplacian)

    # Порог по Отсу для выделения ярких областей
    _, brightness_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Маска ярких областей по Отсу
    _, bright_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Преобразуем bright_mask в булеву маску
    bright_mask_bool = bright_mask.astype(bool)

    # Маска низкой детализации
    low_detail_mask = detail_map < detail_thresh

    # Маска бликов — яркие + низкая детализация
    glare_mask = np.logical_and(bright_mask_bool, low_detail_mask)

    # Процент площади бликов
    glare_ratio = np.sum(glare_mask) / glare_mask.size * 100

    # Интерпретация
    if glare_ratio < 1:
        interpretation = "Бликов практически нет."
    elif glare_ratio < 5:
        interpretation = "Незначительные блики."
    elif glare_ratio < 15:
        interpretation = "Умеренные блики."
    else:
        interpretation = "Сильная засветка – блики критичны."

    screen_res = 1280, 720  # половина HD

    # Масштабирование маски бликов (glare_mask — 2D массив)
    # Убедись, что glare_mask типа uint8 (0 и 1), умножаем на 255 для видимости
    resized_mask = cv2.resize(glare_mask.astype(np.uint8) * 255, screen_res, interpolation=cv2.INTER_NEAREST)

    return [
      {'title': 'Маска бликов', 'data': resized_mask, 'cmap': None, 'type': 'flat'},
      {'title': 'Порог яркости (Отсу):', 'data': _, 'cmap': None, 'type': 'text'},
      {'title': 'Процент бликов:', 'data': f"Процент бликов: {glare_ratio:.2f}%", 'cmap': None, 'type': 'text'},
      {'title': 'Интерпретация', 'data': interpretation, 'cmap': None, 'type': 'text'},
    ]