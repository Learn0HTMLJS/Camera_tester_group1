### **OpenCV: Библиотека компьютерного зрения**  

**OpenCV** (Open Source Computer Vision Library) — это открытая библиотека для обработки изображений и компьютерного зрения, разработанная Intel в 1999 году. Она поддерживает Python, C++, Java и другие языки, а также работает на Windows, Linux, macOS, Android и iOS.  

## **Основные возможности OpenCV**  
1. **Чтение и запись изображений/видео**  
   - Поддержка форматов: JPEG, PNG, TIFF, WebP, MP4, AVI и др.  
   - Работа с камерами (включая USB и IP-камеры).  

2. **Обработка изображений**  
   - Фильтрация (размытие, шумоподавление, детекция краёв).  
   - Морфологические операции (эрозия, дилатация).  
   - Цветовые преобразования (RGB ↔ HSV, Grayscale).  

3. **Компьютерное зрение**  
   - Детекция объектов (Haar Cascades, HOG, YOLO, SSD).  
   - Распознавание лиц и жестов.  
   - Оптический поток (трекинг движения).  
   - Калибровка камеры и 3D-реконструкция.  

4. **Машинное обучение**  
   - Встроенные алгоритмы (k-NN, SVM, Random Forest).  
   - Интеграция с TensorFlow, PyTorch.  

---

## **Пример кода на Python**  
### **1. Загрузка и отображение изображения**  
```python
import cv2

# Загрузка изображения
image = cv2.imread("image.jpg")

# Отображение
cv2.imshow("Image", image)
cv2.waitKey(0)  # Ожидание нажатия клавиши
cv2.destroyAllWindows()
```

### **2. Детекция границ (Canny Edge Detection)**  
```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)  # Пороги градиента
cv2.imshow("Edges", edges)
cv2.waitKey(0)
```

### **3. Распознавание лиц (Haar Cascades)**  
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("Faces", image)
cv2.waitKey(0)
```

---

## **Установка OpenCV**  
```bash
pip install opencv-python          # Основной пакет
pip install opencv-contrib-python  # Дополнительные модули (SIFT, SURF)
```

---

## **Полезные ссылки**  
- **[Официальный сайт OpenCV](https://opencv.org/)**  
- **[Документация OpenCV](https://docs.opencv.org/4.x/)**  
- **[Репозиторий на GitHub](https://github.com/opencv/opencv)**  
- **[Учебник OpenCV на Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)**  

OpenCV широко используется в робототехнике, дополненной реальности (AR), медицинской визуализации и автономных автомобилях. 🚀