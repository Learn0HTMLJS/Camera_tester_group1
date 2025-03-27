# КОД ДЛЯ ОЦЕНКИ РЕЗКОСТИ с помощью оператора Лапласа. OpenCV
import cv2
import numpy as np
from inspect import getsourcefile
import os

def estimate_sharpness(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Изображение не загружено. Проверьте путь.")
    
    # Применение оператора Лапласа и вычисление дисперсии
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sharpness = laplacian.var()
    
    return sharpness
###
# Пример использования
#image_path = os.path.join(os.path.dirname(__file__), 'my_photo.jpg')
image_path = 'C:\my_photo.jpg'
sharpness_score = estimate_sharpness(image_path)
print(f"Оценка резкости: {sharpness_score}")

# Интерпретация результата (эмпирические значения)
if sharpness_score < 50:
    print("Изображение размытое.")
elif 50 <= sharpness_score < 150:
    print("Изображение средней резкости.")
else:
    print("Изображение очень резкое.")