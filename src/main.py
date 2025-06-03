from chrome_abb.chrome_abb import detect_chromatic_aberration
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec
from tkinter import Tk, filedialog

from geometry.geometry import estimate_distortion
from glare.glare import detect_glare_with_otsu
from moire.moire import detect_moire_pattern
from noise.noise import evaluate_noise
from rolling_shutter.rolling_shutter import detect_rolling_shutter_pattern
from sharpness.sharpness import calculate_mtf

# Загрузка изображения
Tk().withdraw()
file_path = filedialog.askopenfilename(
    title="Выберите изображение",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
)
if not file_path:
    print("Изображение не выбрано!")
    exit()
sharpness_file_path = filedialog.askopenfilename(
    title="Выберите изображение для определения резкости",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
)
if not sharpness_file_path:
    print("Изображение не выбрано!")
    exit()

# --- Создание views ---
img_bgr = cv2.imread(file_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

"""
    Параметры:
        title (str): Заголовок окна
        object (object): {
            title (str): Заголовок области
            cmap (str): 
        }
        type (str: "plot" | "multiplot" | "text"):
"""
views = [
    {'title': 'Оригинальное изображение', 'object': {'data': img_rgb, 'cmap': 'gray'}, 'type': 'plot'},
    {'title': 'Резкость', 'object': lambda: calculate_mtf(sharpness_file_path), 'type': 'multiplot'},
    {'title': 'Эффект "Муар"', 'object': lambda: detect_moire_pattern(file_path), 'type': 'multiplot'},
    {'title': 'Временной параллакс', 'object': lambda: detect_rolling_shutter_pattern(file_path), 'type': 'plot'},
    {'title': 'Блики', 'object': lambda: detect_glare_with_otsu(file_path), 'type': 'multiplot'},
    {'title': 'Шум', 'object': lambda: evaluate_noise(file_path), 'type': 'multiplot'},
    {'title': 'Хром-аберрация', 'object': lambda: detect_chromatic_aberration(file_path), 'type': 'multiplot'},
    {'title': 'Искажение', 'object': lambda: estimate_distortion(file_path), 'type': 'multiplot'},
]

# --- Настройка графика ---
fig, ax = plt.subplots(figsize=(10, 8))
subplot_axes = []
plt.subplots_adjust(bottom=0.2)
current_index = [0]

# --- Функция отображения текущего view ---
def show_view(index):
    global ax, subplot_axes # Чтобы удалить предыдущую ось
    view = views[index]

    # Удаляем текущую ось (если есть), чтобы очистить место
    if ax in fig.axes:
        ax.remove()
        fig.suptitle('')
        
    # Удаляем все subplot-оси, если они были
    if subplot_axes:
        for a in subplot_axes:
            if a in fig.axes:
                a.remove()
        subplot_axes.clear()
        fig.suptitle('')

    if view.get('type') == 'multiplot':
        subplot_axes = []
        data_list = view['object']()

        n = len(data_list)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        gs = GridSpec(rows, cols, figure=fig)
        gs.update(bottom=0.2)  # Оставляем место для кнопок

        for i, element in enumerate(data_list):
            ax_sub = fig.add_subplot(gs[i])
            subplot_axes.append(ax_sub)

            if element.get('type') == 'text':
                ax_sub.set_facecolor('gray')
                ax_sub.text(0.5, 0.5, element['data'], fontsize=12, ha='center', bbox=dict(facecolor='none', edgecolor='black', boxstyle='round,pad=1'))
                if element.get('title'):
                    ax_sub.set_title(element['title'])
                ax_sub.axis('off')
            elif element.get('type') == 'hist':
                ax_sub.hist(element['data'], 256, [0,256])
                ax_sub.set_title(element['title'])
                ax_sub.axis('off')
            elif element.get('type') == 'graph':
                if element['space'] == 2:
                    x = element['data']['x']
                    y = element['data']['y']
                    ax_sub.plot(x['region'], y['region'])
                    
                    if x.get('label'):
                        ax_sub.set(xlabel = x['label'])
                    if y.get('label'):
                        ax_sub.set(ylabel = y['label'])
                
                elif element['space'] == 1:
                    x = element['data']['x']
                    ax_sub.plot(x['region'])
                    
                    if x.get('label'):
                        ax_sub.set(xlabel = x['label'])
                    
                ax_sub.set_title(element['title'])
                if element['grid'] == True:
                    ax_sub.grid()
            elif element.get('type') == 'flat':
                ax_sub.imshow(element['data'], cmap=element.get('cmap'))
                ax_sub.set_title(element['title'])
                ax_sub.axis('off')
            elif element.get('type') == 'bar':
                ax_sub.bar(element['data'][0], element['data'][1])
                ax_sub.set_title(element['title'])
                ax_sub.set_xlabel(element['xlabel'])
                ax_sub.set_ylabel(element['ylabel'])

        fig.suptitle(view.get('title', 'Multiplot View'), fontsize=14)

    else:
        gs = GridSpec(1, 1, figure=fig)
        gs.update(bottom=0.2)  # Оставляем место для кнопок
        
        ax = fig.add_subplot(gs[0])
        data = view['object']

        if callable(data):
            displayed_data = data()
        else:
            displayed_data = data
            
        if view.get('type') == 'text':
            ax.text(0.5, 0.5, f"{displayed_data}", fontsize=12, ha='center')
        elif view.get('type') == 'plot':
            ax.imshow(displayed_data['data'], cmap=displayed_data['cmap'])
            if displayed_data.get('title'):
                ax.set_title(displayed_data['title'])

        ax.axis('off')

    fig.suptitle(view.get('title', 'Multiplot View'), fontsize=14)
    fig.canvas.draw_idle()


# --- Callback кнопок ---
def next_view(event):
    current_index[0] = (current_index[0] + 1) % len(views)
    show_view(current_index[0])


def prev_view(event):
    current_index[0] = (current_index[0] - 1) % len(views)
    show_view(current_index[0])


# --- Инициализация кнопок ---
ax_prev = plt.axes([0.7, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])

btn_prev = Button(ax_prev, 'Предыдущий')
btn_next = Button(ax_next, 'Следующий')

btn_prev.on_clicked(prev_view)
btn_next.on_clicked(next_view)

# --- Показываем первый view ---
show_view(current_index[0])
plt.show()