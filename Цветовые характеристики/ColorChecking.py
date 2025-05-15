import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from colormath.color_diff import delta_e_cie1976, delta_e_cie1994
from colormath.color_objects import LabColor
import matplotlib.pyplot as plt

# Загрузка тестовых данных (пример: CSV с эталонными и камерными значениями)
data = pd.read_csv('color_checker_data.csv')  # Замените на свои данные

# Предположим, что у нас есть предсказанные CIELAB значения из модели камеры
# В реальности вам нужно преобразовать Camera_RGB -> CIELAB (через калибровочную модель)
predicted_L = data['Predicted_L']  # Пример: получено из нейронной сети/полинома
predicted_a = data['Predicted_a']
predicted_b = data['Predicted_b']

# Эталонные значения (известные Lab для ColorChecker)
reference_L = data['Reference_L']
reference_a = data['Reference_a']
reference_b = data['Reference_b']

# Расчет ошибок
def calculate_metrics(ref_L, ref_a, ref_b, pred_L, pred_a, pred_b):
    # RMSE для L, a, b
    rmse_L = np.sqrt(mean_squared_error(ref_L, pred_L))
    rmse_a = np.sqrt(mean_squared_error(ref_a, pred_a))
    rmse_b = np.sqrt(mean_squared_error(ref_b, pred_b))
    rmse_total = np.sqrt(mean_squared_error(
        np.vstack([ref_L, ref_a, ref_b]).T,
        np.vstack([pred_L, pred_a, pred_b]).T
    ))

    # ΔE (CIE76 и CIE94)
    delta_e_76 = []
    delta_e_94 = []
    for i in range(len(ref_L)):
        ref_color = LabColor(ref_L[i], ref_a[i], ref_b[i])
        pred_color = LabColor(pred_L[i], pred_a[i], pred_b[i])
        delta_e_76.append(delta_e_cie1976(ref_color, pred_color))
        delta_e_94.append(delta_e_cie1994(ref_color, pred_color))

    # R² коэффициент
    r2 = r2_score(np.vstack([ref_L, ref_a, ref_b]).T,
                  np.vstack([pred_L, pred_a, pred_b]).T)

    return {
        'RMSE_L': rmse_L,
        'RMSE_a': rmse_a,
        'RMSE_b': rmse_b,
        'RMSE_Total': rmse_total,
        'DeltaE_76_mean': np.mean(delta_e_76),
        'DeltaE_94_mean': np.mean(delta_e_94),
        'R2_Score': r2
    }

metrics = calculate_metrics(reference_L, reference_a, reference_b,
                           predicted_L, predicted_a, predicted_b)

# Вывод результатов
print("Метрики ошибок цветопередачи:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(reference_L, predicted_L, label='L*', c='blue')
plt.scatter(reference_a, predicted_a, label='a*', c='red')
plt.scatter(reference_b, predicted_b, label='b*', c='green')
plt.plot([0, 100], [0, 100], 'k--', label='Идеальная точность')
plt.xlabel('Эталонные значения')
plt.ylabel('Предсказанные значения')
plt.title('Сравнение эталонных и предсказанных цветов')
plt.legend()
plt.grid()
plt.show()