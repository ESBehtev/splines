import numpy as np


# Универсальная функция генерации точек
def generate_noisy_data(func, noise_param=0.0, x_range=(0, 10), n_points=50, return_true_curve=True, dense_resolution=500):
	"""
	Генерация зашумлённых данных и (опционально) истинной кривой.

	Параметры:
		func         — математическая функция (например, np.sin)
		noise_param  — амплитуда шума
		x_range      — диапазон по оси X: (min, max)
		n_points     — количество точек
		return_true_curve — возвращать ли "истинную" функцию
		dense_resolution — сколько точек в плотной сетке

	Returns:
		x, y —  данные
		x_dense, y_true — (если return_true_curve=True) плотная сетка и истинная функция

	Пример использования:
		f = np.sin
		x_data, y_data, x_dense, y_true = generate_noisy_data(f, noise_param=0.1, x_range=(0, 10), n_points=10)

		# Построение графика
		plt.figure(figsize=(10, 6))
		plt.plot(x_dense, y_true, '--',color = 'green' ,label=f'Истинный {f.__name__}(x)')
		plt.scatter(x_data, y_data, color='purple', label='Исходные точки')

		plt.xlabel('x')
		plt.ylabel(f'{f.__name__}(x)')
		plt.legend()
		plt.grid(True)
		plt.show()
	"""
	# функция шума
	def w_noise(a, N):
		return a * np.random.normal(0, 1, N)

	x = np.sort(np.random.uniform(low=x_range[0], high=x_range[1], size=n_points))
	y = func(x)
	y += w_noise(noise_param, len(x))

	# Плотная сетка и истинная кривая
	if return_true_curve:
		x_dense = np.linspace(x_range[0], x_range[1], dense_resolution)
		y_true = func(x_dense)
		return x, y, x_dense, y_true
	else:
		return x, y

def factorial(n):
	if n <= 1:
		return 1

	return n * factorial(n - 1)