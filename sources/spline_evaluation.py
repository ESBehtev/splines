
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Определяем контрольные точки для B-сплайна
ctr = np.array([(3, 1), (2.5, 4), (0, 1), (-2.5, 4),
                (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])

# Разделяем контрольные точки на координаты x и y
x = ctr[:, 0]
y = ctr[:, 1]

# Раскомментируйте обе линии для получения замкнутой кривой
# x = np.append(x, [x[0]])  # Добавляем первую точку в конец массива x
# y = np.append(y, [y[0]])  # Добавляем первую точку в конец массива y

l = len(x)  # Получаем количество контрольных точек

# Создаем параметр t для интерполяции
t = np.linspace(0, 1, l-2, endpoint=True)  # Генерируем равномерные значения от 0 до 1
t = np.append([0, 0, 0], t)  # Добавляем начальные значения для кубической интерполяции
t = np.append(t, [1, 1, 1])  # Добавляем конечные значения для кубической интерполяции

# Создаем B-сплайн с заданными параметрами
tck = [t, [x, y], 3]  # tck: узлы, контрольные точки и степень сплайна (3 - кубический)
u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)  # Генерируем значения параметра u для оценки кривой
out = interpolate.splev(u3, tck)  # Оцениваем B-сплайн

# Рисуем контрольный многоугольник
plt.plot(x, y, 'k--', label='Контрольный многоугольник', marker='o', markerfacecolor='red')
# plt.plot(x, y, 'ro', label='Только контрольные точки')  # Рисуем только контрольные точки (раскомментировать при необходимости)

# Рисуем B-сплайн
plt.plot(out[0], out[1], 'b', linewidth=2.0, label='Кривая B-сплайна')
plt.legend(loc='best')  # Добавляем легенду
plt.axis([min(x) - 1, max(x) + 1, min(y) - 1, max(y) + 1])  # Устанавливаем границы осей
plt.title('Оценка кубической кривой B-сплайна')  # Заголовок графика
plt.show()  # Показываем график


