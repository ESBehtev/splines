import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
# Базовый класс Spline
class Spline:
    def __init__(self, knots, degree, coefficients=None, dimension=1):
        """
        Инициализация базового класса Spline.

        Параметры:
        - knots (array-like): Узлы сплайна.
        - degree (int): Степень сплайна.
        - coefficients (array-like, optional): Коэффициенты сплайна.
        - dimension (int): Размерность сплайна.
        """
        self.knots = np.array(knots)
        self.degree = degree
        self.dimension = dimension
        self.coefficients = np.array(coefficients) if coefficients is not None else None

    def evaluate(self, x):
        """
        Метод для вычисления значения сплайна в точке x.
        Должен быть переопределен в подклассе.
        """
        raise NotImplementedError("Этот метод должен быть переопределен в подклассе.")

    def plot_spline(self, x_range, num_points=100):
        """
        Построение графика сплайна в указанном диапазоне.

        Параметры:
        - x_range (tuple): Кортеж (min_x, max_x) для диапазона построения.
        - num_points (int): Количество точек для построения графика.
        """
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = self.evaluate(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
        if self.coefficients is not None:
            # Отображаем узлы сплайна
            plt.scatter(self.knots, self.evaluate(self.knots), color='red', label="Узлы сплайна")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Построение {self.__class__.__name__} сплайна")
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def create_p_spline(cls, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
        """
        Фабричный метод для создания объекта p_spline.

        Параметры:
        - x (array-like): Данные независимой переменной.
        - y (array-like): Данные зависимой переменной.
        - knots (array-like, optional): Узлы сплайна. Если не заданы, будут рассчитаны автоматически.
        - degree (int): Степень сплайна.
        - penalty_order (int): Порядок разностного штрафа.
        - lambda_ (float): Параметр сглаживания.
        - dimension (int): Размерность сплайна.

        Возвращает:
        - p_spline: Экземпляр подкласса p_spline.
        """
        return p_spline(x, y, knots, degree, penalty_order, lambda_, dimension)

# Подкласс p_spline
class p_spline(Spline):
    def __init__(self, x, y, knots=None, degree=3, penalty_order=2, lambda_=1.0, dimension=1):
        """
        Инициализация объекта p_spline.

        Параметры:
        - x (array-like): Данные независимой переменной.
        - y (array-like): Данные зависимой переменной.
        - knots (array-like, optional): Узлы сплайна.
        - degree (int): Степень сплайна.
        - penalty_order (int): Порядок разностного штрафа.
        - lambda_ (float): Параметр сглаживания.
        - dimension (int): Размерность сплайна.
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.penalty_order = penalty_order
        self.lambda_ = lambda_
        self.spline = None

        self.knots = knots
        self.degree = degree

        # Инициализируем граничные условия как None
        self.boundary_conditions = None

        # Выполняем подгонку сплайна к данным
        self.fit()

        # Инициализируем базовый класс с найденными параметрами
        super().__init__(knots=self.knots, degree=self.degree, coefficients=self.coefficients, dimension=dimension)

    def _difference_matrix(self, n_bases, d):
        """
        Создает разностную матрицу порядка d.

        Параметры:
        - n_bases (int): Количество базисных функций B-сплайна.
        - d (int): Порядок разности.

        Возвращает:
        - ndarray: Разностная матрица.
        """
        D = np.eye(n_bases)
        for _ in range(d):
            D = np.diff(D, n=1, axis=0)
        return D

    def set_boundary_conditions(self, bc_type, bc_values=None):
        """
        Задает граничные условия для сплайна.

        Параметры:
        - bc_type (str): Тип граничных условий ('natural', 'clamped').
        - bc_values (dict, optional): Значения производных для граничных условий.
            Для 'clamped' требуется {'left': value, 'right': value}.
            Для 'natural' не нужны дополнительные значения.
        """
        if bc_type not in ['natural', 'clamped']:
            raise ValueError("Поддерживаемые типы граничных условий: 'natural', 'clamped'.")

        if bc_type == 'clamped':
            if (bc_values is None or
                'left' not in bc_values or
                'right' not in bc_values):
                raise ValueError("Для 'clamped' граничных условий необходимо предоставить 'left' и 'right' значения производных.")

        self.boundary_conditions = {
            'type': bc_type,
            'values': bc_values
        }

        # После установки граничных условий необходимо повторно выполнить подгонку
        self.fit()

    def fit(self):
        """
        Аппроксимирует P-сплайн к предоставленным данным с использованием пенализованных наименьших квадратов
        и учитывая граничные условия, если они заданы.
        """
        if self.knots is None:
            num_internal_knots = max(int(len(self.x) / 4), 4)
            self.knots = np.linspace(min(self.x), max(self.x), num_internal_knots)
            self.knots = np.concatenate((
                [self.x[0]] * self.degree,
                self.knots,
                [self.x[-1]] * self.degree
            ))

        n = len(self.x)
        t = self.knots
        k = self.degree
        n_bases = len(t) - k - 1

        # Создаем базисную матрицу B
        B = np.zeros((n, n_bases))
        for i in range(n_bases):
            c = np.zeros(n_bases)
            c[i] = 1
            spline = BSpline(t, c, k)
            B[:, i] = spline(self.x)

        # Создаем штрафную матрицу P
        D = self._difference_matrix(n_bases, self.penalty_order)
        P = self.lambda_ * D.T @ D

        # Основная система уравнений: (B^T B + P) c = B^T y
        BtB = B.T @ B
        Bty = B.T @ self.y
        A = BtB + P
        rhs = Bty.copy()

        # Обработка граничных условий
        if self.boundary_conditions is not None:
            bc_type = self.boundary_conditions['type']
            if bc_type == 'natural':
                # Вторая производная на концах равна нулю
                # Вычисляем вторые производные базисных функций на концах
                # Для каждого базисного сплайна вычисляем его вторую производную на границе
                B_der2_left = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[0])
                    for i in range(n_bases)
                ])
                B_der2_right = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(2)(self.x[-1])
                    for i in range(n_bases)
                ])

                # Добавляем эти условия в систему
                A = np.vstack([A, B_der2_left, B_der2_right])
                rhs = np.hstack([rhs, 0, 0])

            elif bc_type == 'clamped':
                # Первая производная на концах задана
                bc_values = self.boundary_conditions['values']
                # Вычисляем первые производные базисных функций на концах
                B_der1_left = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[0])
                    for i in range(n_bases)
                ])
                B_der1_right = np.array([
                    BSpline(t, np.eye(n_bases)[i], k).derivative(1)(self.x[-1])
                    for i in range(n_bases)
                ])

                # Добавляем эти условия в систему
                A = np.vstack([A, B_der1_left, B_der1_right])
                rhs = np.hstack([rhs, bc_values['left'], bc_values['right']])

        # Решаем систему уравнений с учетом граничных условий
        try:
            c = np.linalg.lstsq(A, rhs, rcond=None)[0]
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"Ошибка при решении системы уравнений: {e}")

        self.coefficients = c
        self.spline = BSpline(t, c, k)

    def evaluate(self, x):
        """
        Метод для вычисления значения сплайна в точке x.
        """
        if self.spline is None:
            raise ValueError("Сплайн еще не аппроксимирован.")
        return self.spline(x)

    def predict(self, x_new):
        """
        Предсказывает значения y для новых значений x с использованием аппроксимированного сплайна.

        Параметры:
        - x_new (array-like): Новые значения x.

        Возвращает:
        - ndarray: Предсказанные значения y.
        """
        return self.evaluate(x_new)

    def plot_spline(self, x_range=None, num_points=100):
        """
        Переопределение метода для построения графика сплайна вместе с исходными данными.

        Параметры:
        - x_range (tuple, optional): Диапазон (min_x, max_x) для построения графика. Если не задан, используется диапазон данных.
        - num_points (int): Количество точек для построения графика.
        """
        if x_range is None:
            x_range = (min(self.x), max(self.x))
        x_vals = np.linspace(x_range[0], x_range[1], num_points)
        y_vals = self.evaluate(x_vals)

        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label=f"{self.__class__.__name__} сплайн")
        plt.scatter(self.x, self.y, color='red', label="Данные")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Построение {self.__class__.__name__} сплайна")
        plt.legend()
        plt.grid(True)
        plt.show()

    def method_specific_to_p_spline(self):
        """
        Пример метода, специфичного для p_spline.
        """
        #print("Это метод, специфичный для p_spline.")

def plot_p_spline(start=0,stop=10,num=100, boundary_conditions=None):
    # Генерация данных
    np.random.seed()  # Для воспроизводимости
    x_data = np.linspace(start, stop, num)
    y_data = np.sin(x_data) + np.random.normal(0, 0.2, size=len(x_data))

    # Создание объекта p_spline через фабричный метод базового класса Spline
    spline_P = Spline.create_p_spline(
        x=x_data,
        y=y_data,
        degree=3,
        penalty_order=2,
        lambda_=1.0
    )

    # Использование методов базового класса Spline
    x_new = np.linspace(0, 10, 200)
    y_new = spline_P.evaluate(x_new)

    # Построение графика сплайна без граничных условий

    # 1-natural
    # 2-clamped
    # Установка граничных условий, если они заданы
    if boundary_conditions == 1:
        spline_P.set_boundary_conditions(bc_type='natural')
        spline_P.plot_spline(x_range=(start, stop), num_points=200)
        print("Сплайн с граничными условиями 'natural':")
    elif boundary_conditions == 2:
        clamped_values = {'left': 1.0, 'right': -1.0}  # Пример значений производных
        spline_P.set_boundary_conditions(bc_type='clamped', bc_values=clamped_values)
        spline_P.plot_spline(x_range=(start, stop), num_points=200)
        print("Сплайн с граничными условиями 'natural':")
    else:
        spline_P.plot_spline(x_range=(start, stop), num_points=200)
        print("Сплайн без граничных условий:")

    # Использование специфичного метода p_spline
    spline_P.method_specific_to_p_spline()



if __name__ == "__main__":
    plot_p_spline()







