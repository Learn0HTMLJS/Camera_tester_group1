import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

def plot_color_spectrum(image_path, n_colors=10):
    # Загрузка изображения
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка загрузки изображения")
        return
    
    # Конвертация из BGR в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Подготовка данных для кластеризации
    pixels = img_rgb.reshape((-1, 3))
    
    # Кластеризация цветов с помощью K-means
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    
    # Получение центров кластеров (доминирующих цветов)
    colors = kmeans.cluster_centers_
    colors = np.array(colors, dtype=np.uint8)
    
    # Визуализация в 3D пространстве RGB
    fig = plt.figure(figsize=(12, 6))
    
    # 3D график     (Вызывает зависание)
    #ax1 = fig.add_subplot(121, projection='3d')
    #r, g, b = cv2.split(img_rgb)
    #ax1.scatter(r.flatten(), g.flatten(), b.flatten(), c=pixels/255.0, marker='.', s=1)
    #ax1.set_xlabel('Red')
    #ax1.set_ylabel('Green')
    #ax1.set_zlabel('Blue')
    #ax1.set_title('Цветовой спектр изображения')
    
    # 2D график доминирующих цветов
    ax2 = fig.add_subplot(122)
    color_swatches = np.zeros((100, 100 * n_colors, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        color_swatches[:, i*100:(i+1)*100] = color
    ax2.imshow(color_swatches)
    ax2.axis('off')
    ax2.set_title(f'Доминирующие цвета ({n_colors})')
    
    plt.tight_layout()
    plt.show()

    # Возвращаем доминирующие цвета
    return colors

# Пример использования
image_path = 'C:\\me.jpg'  # Укажите путь к вашему изображению
dominant_colors = plot_color_spectrum(image_path, n_colors=8)
print("Доминирующие цвета (RGB):")
print(dominant_colors)