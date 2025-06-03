import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_moire_pattern(image_path, threshold=30, kernel_size=5):
    """
    Обнаружение муара на изображении.
    
    Параметры:
        image_path (str): Путь к изображению
        threshold (int): Порог для обнаружения муара (по умолчанию 30)
        kernel_size (int): Размер ядра для морфологических операций (по умолчанию 5)
    """
    # Загрузка изображения в оттенках серого
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Применение быстрого преобразования Фурье (FFT)
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Вычисление амплитудного спектра
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
    
    # Создание маски для высокочастотных компонентов
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols, 2), np.uint8)
    r = min(rows, cols) // 4  # Радиус центральной области, которую мы сохраняем
    cv2.circle(mask, (ccol, crow), r, (0,0,0), -1)
    
    # Применение маски и обратное преобразование Фурье
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
    
    # Нормализация и пороговая обработка
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    _, thresholded = cv2.threshold(img_back, threshold, 255, cv2.THRESH_BINARY)
    
    # Морфологические операции для улучшения результата
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    
    # Расчет процента площади с муаром
    moire_area = np.sum(morphed == 255)
    total_area = rows * cols
    moire_percentage = (moire_area / total_area) * 100
    
    # Наложение муара на исходное изображение
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    morphed_color = cv2.cvtColor(morphed, cv2.COLOR_GRAY2BGR)
    morphed_color[:,:,0] = 0  # Убираем красный канал
    morphed_color[:,:,1] = 0  # Убираем зеленый канал
    overlay = cv2.addWeighted(img_color, 1, morphed_color, 0.5, 0)
    
    return [
        {'title': 'Амплитудный спектр (Фурье)', 'data': magnitude_spectrum, 'cmap': 'gray', 'type': 'flat'},
        {'title': 'Маска для Фурье', 'data': mask[:, :, 0], 'cmap': 'gray', 'type': 'flat'},
        {'title': 'Обратное Фурье с маской', 'data': img_back, 'cmap': 'gray', 'type': 'flat'},
        {'title': f'Обнаруженный муар ({moire_percentage:.2f}%)', 'data': morphed, 'cmap': 'gray', 'type': 'flat'},
        {'title': 'Муар на исходном изображении', 'data': cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 'cmap': None, 'type': 'flat'},
    ]