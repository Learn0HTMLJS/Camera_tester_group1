# КОД ДЛЯ ОЦЕНКИ РЕЗКОСТИ с помощью оператора Лапласа. OpenCV
import cv2

def estimate_sharpness(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Изображение не загружено. Проверьте путь.")
    
    # Применение оператора Лапласа и вычисление дисперсии
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = laplacian.var()
    
    return sharpness