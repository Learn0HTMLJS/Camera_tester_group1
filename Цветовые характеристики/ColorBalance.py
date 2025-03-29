import cv2
import numpy as np

def find_gray_card(img_rgb, threshold=20, min_area_ratio=0.001):
    """
    Автоматически находит серую карту на изображении.
    
    Параметры:
        img_rgb (numpy.ndarray): RGB-изображение (8-bit).
        threshold (int): Допустимое отклонение от идеального серого.
        min_area_ratio (float): Минимальная относительная площадь серой карты.
    
    Возвращает:
        tuple: Координаты области серой карты (x, y, w, h) или None.
    """
    # Конвертируем в HSV для устойчивого поиска серого
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Маска серого цвета (Saturation < 50, Value 50-200)
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
    
    # Улучшаем маску морфологическими операциями
    kernel = np.ones((5,5), np.uint8)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
    gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
    
    # Ищем контуры
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Фильтруем по площади и форме
    min_area = min_area_ratio * img_rgb.shape[0] * img_rgb.shape[1]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            aspect_ratio = max(w,h)/min(w,h)
            if aspect_ratio < 1.5:  # Игнорируем вытянутые области
                return cv2.boundingRect(cnt)
    return None

def auto_white_balance_jpeg(img_path, target_gray=(128, 128, 128)):
    """
    Корректирует баланс белого JPEG-изображения по найденной серой карте.
    
    Параметры:
        img_path (str): Путь к JPEG-файлу.
        target_gray (tuple): Целевой цвет серой карты (sRGB, по умолчанию 128,128,128).
    
    Возвращает:
        tuple: (изображение с выделенной картой, сбалансированное изображение)
    """
    # Загружаем изображение
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError("Не удалось загрузить изображение!")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Ищем серую карту
    gray_card_rect = find_gray_card(img_rgb)
    if gray_card_rect is None:
        raise ValueError("Серая карта не найдена!")
    
    x,y,w,h = gray_card_rect
    gray_region = img_rgb[y:y+h, x:x+w]
    
    # Среднее значение в области карты
    mean_rgb = np.mean(gray_region, axis=(0,1))
    
    # Коэффициенты коррекции (для sRGB)
    scale = np.array(target_gray) / mean_rgb
    
    # Применяем баланс белого
    balanced = np.clip(img_rgb * scale, 0, 255).astype(np.uint8)
    balanced_bgr = cv2.cvtColor(balanced, cv2.COLOR_RGB2BGR)
    
    # Рисуем прямоугольник для отладки
    debug_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(debug_img, (x,y), (x+w,y+h), (0,255,0), 2)
    
    return debug_img, balanced_bgr

# Пример использования
if __name__ == "__main__":
    try:
        debug_img, result_img = auto_white_balance_jpeg("photo_with_gray_card.jpg")
        
        cv2.imshow("Detected Gray Card", debug_img)
        cv2.imshow("White Balanced Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Сохранение результатов
        cv2.imwrite("debug_output.jpg", debug_img)
        cv2.imwrite("balanced_result.jpg", result_img)
    except ValueError as e:
        print(f"Ошибка: {e}")