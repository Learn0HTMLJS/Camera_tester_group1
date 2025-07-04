## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Camera%20with%20Flash.png" alt="Camera with Flash" width="40" height="40" /> <img src="https://readme-typing-svg.herokuapp.com/?font=Fira+Code&size=24&pause=1000&vCenter=true&width=435&height=25&lines=%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7+%D0%BA%D0%B0%D0%BC%D0%B5%D1%80%D1%8B" alt="Typing SVG" />

###  Инструмент для тестирования характеристик камеры на основе изображений.


## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Bar%20Chart.png" alt="Bar Chart" width="25" height="25" /> Какие характеристики тестируются?
- **Резкость**  
- **Шумы**
- **Виньетирование**
- **Геометрическое искажение**
- **Хроматическая аберрация**
- **Временной параллакс**
- **Муар**
- **Светосила**
- **Блики**
- **Инфокрасная подсветка**
<!-- - **Цветопередача** -->
---

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Hammer%20and%20Wrench.png" alt="Hammer and Wrench" width="25" height="25" /> Зависимости

| Библиотека       | Версия  | Назначение                                              | Ссылка                                    |
|------------------|---------|---------------------------------------------------------|-------------------------------------------|
| OpenCV           | 4.11.0  | Обработка изображений и анализ                          | [https://opencv.org/](https://opencv.org) |
| NumPy            | 2.2.0   | Матричные операции и вычисления                         | [https://numpy.org/](https://numpy.org)   |
| Matplotlib       | 3.10.3  | Визуализации данных двумерной и трёхмерной графикой     | [https://matplotlib.org/](https://matplotlib.org)   |


---

## ✅ Как использовать

Предварительно отключите все AI помошники!
0. Установите зависимости из файла requirements.txt (с помощью комманды pip install)
1. Загрузите обычное изображение (можно использовать обычные фотографии с бытовыми сценами)
2. Загрузите изображение резкого чёрно-белого края (например, лист бумаги на контрастном фоне)
3. Запустите main.py
4. Для пролистывания характеристик используйте кнопки Предыдущий и Следующий

---

## 📌 Поддерживаемые форматы изображений

Программы поддерживают все распространённые форматы изображений (JPEG, PNG, BMP и т.д.), поддерживаемые библиотекой OpenCV.

---

## 👨‍💻 Разработчики
* [Панкин Максим](https://github.com/9chyn9)
* [Свинцицкий Роман](https://github.com/Learn0HTMLJS)
* [Архипов Кирилл](ссылка)
* [Васинкина Диана](ссылка)
* [Ашымбеков Тариэл](ссылка)
* [Воробьёва Кристина](ссылка)
---

## 🧪 Результаты апробации работы

[Таблица с результатами](https://docs.google.com/spreadsheets/d/1HhOulAVGFpgDN7Yy8NfflmScbsZdS6lwfDRl9N9adDo/edit?gid=0#gid=0)

---

## 📸 Ограничения и требования к подготовке фото-материалов

Для корректного анализа характеристик камеры с помощью программы необходимо соблюдать определённые условия при подготовке изображений. Эти ограничения обеспечивают сопоставимость результатов и точность вычислений.

### 1. **Единый формат и разрешение**
- Все фотографии должны быть сохранены в одном графическом формате (JPEG, PNG, BMP и т.д., поддерживаемые OpenCV).
- Файлы должны иметь одинаковое разрешение (например, 1920×1080 пикселей), чтобы избежать искажений, связанных с масштабированием.
- Не рекомендуется использовать автоматическое сжатие или изменение размера изображений после съемки.

### 2. **Фиксированная позиция камеры**
- Камера должна находиться в статичном положении на протяжении всей серии снимков.
- Для этого рекомендуется использовать штатив:
  - Высота камеры фиксируется заранее и не меняется между снимками.
  - Угол наклона и направление камеры также остаются неизменными.
- При отсутствии штатива — используйте устойчивую горизонтальную поверхность и ориентируйте камеру строго перпендикулярно объекту съемки.

### 3. **Один и тот же объект съемки**
- Все тестовые фотографии делаются с одного и того же объекта или сцены.
- Объект должен быть статичным, без движений или изменений внешнего вида в процессе съемки.
- Это позволяет сравнивать характеристики разных камер на основе одинаковой исходной информации.

### 4. **Контрастное изображение с чёткой границей**
- В обязательном порядке делается одно изображение с резким контрастным переходом:
  - Например: половина кадра — черная, вторая — белая, разделенные четкой прямой линией.
  - Такое изображение используется для анализа резкости, хроматической аберрации и геометрических искажений.
  - Рекомендуется использовать плоский объект (лист бумаги на светлом фоне) и обеспечить равномерное освещение.

### 5. **Схожие настройки камер**
- При использовании нескольких устройств (например, разных смартфонов):
  - Настройки камеры (экспозиция, ISO, баланс белого, фокусное расстояние) должны быть максимально приближены друг к другу.
  - Отключите автоматические режимы коррекции (AI, HDR, автофокус, шумодавление и т.д.), если это возможно.
  - Если камера не позволяет фиксировать параметры, сделайте предварительный замер и зафиксируйте настройки вручную.

### 6. **Избегание потери качества при передаче изображений**
- Передача файлов должна осуществляться напрямую с устройства, на котором была сделана фотография, на компьютер с программой.
- Избегайте преобразования изображений через облачные сервисы, мессенджеры или социальные сети, так как они могут:
  - Сжимать изображения.
  - Автоматически корректировать цвет и яркость.
  - Менять формат файла.
- Используйте USB-подключение или облачные хранилища без автоматического изменения содержимого (Google Drive, Dropbox и др.).

---

## ✅ Пример идеального набора фото-материалов
Папка `test_images/` может содержать следующие файлы:

| Название файла | Описание |
|----------------|----------|
| sharp_edge.jpg | Четкая черно-белая граница для анализа резкости и искажений |
| wide_building.jpg | Широкоугольный снимок здания для оценки дисторсии |
| standard_building.jpg | Обычный снимок того же здания для сравнения |
| high_contrast_aberration.jpg | Сцена с контрастными краями для обнаружения хроматической аберрации |
| no_aberration.jpg | Эталонное изображение без заметных аберраций |

---

## 🚫 Возможные ошибки при нарушении условий
- Разное разрешение → некорректный расчет резкости и шума.
- Подвижная камера → трудности в сравнении характеристик.
- Разные объекты → невозможно провести объективный анализ.
- Плохой контраст → программа не сможет точно определить границы и искажения.
- Автокоррекция и компрессия → искажение реальных характеристик камеры.

## 📸 Рекомендации при неидеальных условиях съемки

В реальных условиях часто невозможно полностью соблюсти все идеальные требования к подготовке фото-материалов. Ниже приведены рекомендации по использованию программы в таких ситуациях, а также описание возможных ограничений точности анализа.

### 1. **Разный формат или разрешение**
- Если изображения сделаны в разных форматах или имеют разное разрешение:
  - Программа всё равно сможет обработать файлы, но сравнение характеристик будет менее точным.
  - Перед запуском рекомендуется вручную привести изображения к одному разрешению и формату (например, PNG).
  - Избегайте потери качества при конвертации (не используйте сжатие JPEG).

### 2. **Подвижная камера или разные ракурсы**
- Если камера перемещалась или объект снимался под разными углами:
  - Анализ геометрической дисторсии и хроматической аберрации может быть затруднён.
  - Для резкости и шума результаты остаются относительно корректными, если объекты на изображениях схожи.
  - Рекомендуется указать в отчёте условия съёмки для правильной интерпретации данных.

### 3. **Разные объекты съемки**
- При тестировании камер на разных объектах:
  - Сравнение становится качественным, а не количественным.
  - Это допустимо при грубой оценке, но не подходит для точного анализа характеристик.
  - Используйте максимально схожие сцены (например, аналогичные контрастные границы, освещение и фон).

### 4. **Отсутствие контрастного изображения**
- Если нет изображения с чёткой границей (черно-белый край):
  - Некоторые метрики (резкость, хроматическая аберрация) будут недоступны или неточны.
  - В качестве замены можно использовать любое изображение с резкими краями (например, тень на стене, край стола и т.п.), но это снижает надёжность анализа.

### 5. **Разные настройки камер**
- Если настройки камер (ISO, баланс белого, автофокус и т.д.) различаются:
  - Это влияет на уровень шума, цветопередачу и резкость.
  - Возможна автоматическая коррекция изображения самой камерой, что скрывает реальные характеристики.
  - Учитывайте эти факторы при анализе и сравнивайте результаты с осторожностью.

### 6. **Изменение изображений после съемки**
- Если фотографии передавались через мессенджеры, облачные сервисы или социальные сети:
  - Возможно сжатие, изменение цвета, размера или формата.
  - Это может сильно исказить результаты анализа.
  - Если оригинал недоступен, укажите в отчёте, что изображения были изменены, чтобы сохранить прозрачность исследования.

---

## ⚠️ Возможные последствия неидеальных условий

| Условие | Влияние на анализ |
|--------|-------------------|
| Разное разрешение | Неточности в резкости, шуме, геометрии |
| Разные ракурсы | Затруднение анализа дисторсии и аберраций |
| Разные объекты | Невозможность точного сравнения |
| Отсутствие контраста | Ограничение в определении резкости и аберраций |
| Автокоррекция и компрессия | Потеря реальных характеристик камеры |

---

## ✅ Рекомендуемый минимум для работы программы
Даже в неидеальных условиях вы можете получить полезные данные, если обеспечите:
- Хотя бы одно изображение с резким краем для базового анализа.
- Сходные сцены и условия освещения между снимками.
- Минимальную обработку изображений до анализа.
- Честное описание условий съемки при оформлении отчета.

Такой подход позволяет проводить исследование даже без идеального оборудования и контролируемых условий.

---

### 🚀 Возможные улучшения системы
1. **База данных с результатами тестирования**
2. **Меню с выбором параметра, вместо последовательного списка и причие улучшения интерфейса**
3. **Возможность создания веб версии и клиент-серверной архитектуры**
4. **Интерпритация выводимых значений (на данный момент не все результаты имеют пояснение)**
5. **Более подробная инструкция пользователя**

---

## <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Scroll.png" alt="Scroll" width="25" height="25" /> Лицензия
Проект распространяется под [MIT License](https://choosealicense.com/licenses/mit/)

---

<!-- ### **1. Общие методы оценки качества изображения**  
- **Imatest® Software: Principles** (2005) – подробное описание алгоритмов оценки резкости (MTF), дисторсии, виньетирования и шумов.  
  - [Imatest LLC. (2005). *Imatest Technical Documentation*](https://www.imatest.com/docs/)  

- **ISO 12233:2017** – международный стандарт для оценки разрешения и пространственной частотной характеристики.  
  - [ISO. (2017). *Photography — Electronic still picture imaging — Resolution and spatial frequency responses*](https://www.iso.org/standard/71181.html)  

### **2. Оценка шумов и динамического диапазона**  
- **Tzannes, A. P., & Mooney, J. M. (1995).** *Measurement of the modulation transfer function of infrared cameras.* Optical Engineering, 34(6), 1808-1817.  
  - DOI: [10.1117/12.203098](https://doi.org/10.1117/12.203098)  

- **Ponomarenko, N. et al. (2008).** *On between-coefficient contrast masking of DCT basis functions.* Proc. of the 3rd Int. Workshop on Video Processing and Quality Metrics.  
  - [PDF доступен здесь](https://www.researchgate.net/publication/228928390)  

### **3. Сравнение сенсоров и алгоритмов обработки**  
- **Krylov, V. A., & Nelson, J. D. B. (2018).** *Statistical Analysis of Noise in Modern CMOS Image Sensors.* Sensors, 18(9), 3083.  
  - DOI: [10.3390/s18093083](https://doi.org/10.3390/s18093083)  

- **Foi, A. et al. (2008).** *Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data.* IEEE Transactions on Image Processing, 17(10), 1737-1754.  
  - DOI: [10.1109/TIP.2008.2001399](https://doi.org/10.1109/TIP.2008.2001399)  

### **4. Глубинные методы оценки качества**  
- **Bosse, S. et al. (2017).** *Deep Neural Networks for No-Reference and Full-Reference Image Quality Assessment.* IEEE Transactions on Image Processing, 27(1), 206-219.  
  - DOI: [10.1109/TIP.2017.2760518](https://doi.org/10.1109/TIP.2017.2760518)  

### **5. Специальные методы для мультиспектральных и ИК-камер**  
- **Holst, G. C. (2000).** *Testing and Evaluation of Infrared Imaging Systems.* SPIE Press.  
  - [Книга на SPIE](https://spie.org/publications/book/2018196)
Вот еще 5 научных статей, посвященных оценке технических характеристик камер, включая анализ сенсоров, алгоритмы обработки изображений и объективные метрики качества:  

### **6. Анализ разрешения и MTF (Modulation Transfer Function)**  
- **Boreman, G. D. (2001).** *Modulation Transfer Function in Optical and Electro-Optical Systems.* SPIE Press.  
  - DOI: [10.1117/3.419857](https://doi.org/10.1117/3.419857)  
  - Классическая работа по теории и практике измерения MTF для оптических и цифровых систем.  

### **7. Оценка цветопередачи и калибровки камер**  
- **Cheung, V., Westland, S., & Connah, D. (2004).** *A comparative study of the characterisation of colour cameras by means of neural networks and polynomial transforms.* Coloration Technology, 120(1), 19-25.  
  - DOI: [10.1111/j.1478-4408.2004.tb00202.x](https://doi.org/10.1111/j.1478-4408.2004.tb00202.x)  
  - Сравнение методов калибровки цветовых характеристик камер.  

### **8. Шумы и динамический диапазон в CMOS-сенсорах**  
- **Yadav, G., & Kumar, M. (2019).** *Noise Analysis and Modeling for CMOS Image Sensors.* IEEE Sensors Journal, 19(15), 6141-6150.  
  - DOI: [10.1109/JSEN.2019.2913602](https://doi.org/10.1109/JSEN.2019.2913602)  
  - Подробный анализ источников шума и методов их подавления.  

### **9. Автоматизированные системы тестирования камер**  
- **Wueller, D., & Kretzschmar, R. (2010).** *A new approach to camera testing for the photography industry.* IS&T/SPIE Electronic Imaging.  
  - DOI: [10.1117/12.838620](https://doi.org/10.1117/12.838620)  
  - Методы автоматизированного тестирования камер для промышленных применений.  

### **10. Оценка качества изображения в условиях низкой освещенности**  
- **Hasinoff, S. W., et al. (2016).** *Burst photography for high dynamic range and low-light imaging on mobile cameras.* ACM Transactions on Graphics, 35(6), 1-12.  
  - DOI: [10.1145/2980179.2980254](https://doi.org/10.1145/2980179.2980254)  
  - Анализ методов улучшения изображений при слабом освещении.   -->
