import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import BSpline


def b_spline_basis(degree, knots, x):
	basis = np.zeros((len(x), len(knots) - degree - 1))
	for i in range(len(x)):
		basis[i, :] = BSpline(knots, np.eye(len(knots) - degree - 1), degree)(x[i])
	return basis


def p_spline(data, degree, knots, lambda_param):
	x = data[:, 0]
	y = data[:, 1]
	B = b_spline_basis(degree, knots, x)

	# Исправленная строка для создания матрицы D
	D = np.diff(np.eye(B.shape[1]), n=2, axis=0)

	def objective(theta):
		residuals = y - B @ theta
		penalty = lambda_param * theta.T @ D.T @ D @ theta
		return np.sum(residuals ** 2) + penalty

	theta_init = np.zeros(B.shape[1])
	result = minimize(objective, theta_init)
	theta_opt = result.x

	return theta_opt


# Пример использования
np.random.seed(0)
x = np.linspace(0, 10, 100)
y = np.sin(x)
y[50:70] = np.nan  # Добавляем разрыв в данных
data = np.column_stack((x, y))
data = data[~np.isnan(data).any(axis=1)]  # Удаляем строки с NaN

degree = 3
knots = np.linspace(0, 10, 15)
lambda_param = 0.1

theta_opt = p_spline(data, degree, knots, lambda_param)

# Визуализация
x_new = np.linspace(0, 10, 500)
B_new = b_spline_basis(degree, knots, x_new)
y_new = B_new @ theta_opt

plt.plot(data[:, 0], data[:, 1], 'ro', label='Data with Gaps')
plt.plot(x_new, y_new, 'b-', label='P-Spline')
plt.legend()
plt.show()