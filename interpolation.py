import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

class Interp1d(object):
    """
    Класс, превращающий набор точек (x,f) в непрерывную функцию f(x), путем 
    линейной интерполяции между этими точками. Будет использоваться для 
    аэродинамических и массо-тяговременных характеристик ракеты типа P(t), m(t), и т.д.
    """
    def __init__(self, mass_x, mass_f):
        """
        Конструктор класса 
        Arguments: mass_x {list} -- абсциссы интерполируемой функции
                   mass_f {list} -- ординаты интерполируемой функции
        """
        if mass_x.shape == mass_f.shape:
            self.mx = np.array(mass_x)
            self.mf = np.array(mass_f)
        else:
            raise AttributeError(f'Данные разных размерностей: x{mass_x.shape};  f{mass_f.shape}')
            
    def __call__(self, x):
        """
        Метод получения интерполированных данных: object(x) -> y
        Arguments: x {float} -- абсцисса точки
        Returns:   y {float} -- ордината точки
        """
        return np.interp(x, self.mx, self.mf)
    
    def plot(self, ylabel='y', xlabel='x',):
        """
        Визуализация интерполируемых данных
        """
        fig = plt.figure(dpi=100)
        plt.plot(self.mx, self.mf, 'k')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

class Interp2d(object):
    """
    Класс, превращающий набор точек (x,y,f) в непрерывную функцию f(x, y), путем 
    линейной интерполяции между этими точками.
    """
    def __init__(self, mass_x, mass_y, mass_f):
        """
        Конструктор класса 
        Arguments: mass_x {list} -- абсциссы интерполируемой функции, len = N
                   mass_y {list} -- ординаты интерполируемой функции, len = M
                   mass_f {list} -- матрица N x M со значениями функции в соответствующих точках 
        """
        if mass_x.size * mass_y.size == mass_f.size:
            self.mx = np.array(mass_x)
            self.my = np.array(mass_y) 
            self.mf = np.array(mass_f)
            self.func_inter = interp.RectBivariateSpline(self.mx, self.my, self.mf, kx=1, ky=1)
        else:
            raise AttributeError(f'Данные разных размеростей: x{mass_x.shape}; y{mass_y.shape}; f{mass_f.shape}')
        
    def plot(self, ylabel='y', xlabel='x', zlabel='z'):
        """
        Визуализация интерполируемых данных
        """
        fig = plt.figure(dpi=150)
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(self.mx, self.my)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self(X[i,j], Y[i,j])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r, linewidth=0, antialiased=False)
        fig.colorbar(surf, label=zlabel, shrink=0.5, aspect=5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot2d(self):
        for y in self.my:
            f = [self(x, y) for x in self.mx]
            plt.plot(self.mx, f, label=f'{y}')
        plt.grid()
        plt.legend()
    plt.show()

    def __call__(self, x, y):
        """
        Метод получения интерполированных данных
        Arguments: x {float} -- 1 абсцисса  точки
                   y {float} -- 2 абсцисса  точки
        Returns:   f {float} -- ордината точки
        """
        return self.func_inter(x, y)[0,0]

class InterpVec(object):
    """
    Класс служит для подномерной интерполяции векторов
    """
    def __init__(self, tups):
        """
        Конструктор
        Arguments: tups {list} -- список кортежей [(время, (x,y)), (время, (x,y)), ...]
                                              или [(время, [x,y]), (время, [x,y]), ...]
                                              или [(время, np.array([x,y])), (время, np.array([x,y])), ...]
        """
        if len(tups) == 1:
            result = np.array(tups[0][1])
            self.func_inter = lambda t: result
            return
        self.ts = list(map(lambda x: x[0], tups))
        self.vector_velocity = list(map(lambda x: x[1], tups))

        self.func_inter = interp.interp1d(self.ts, self.vector_velocity, axis=0, bounds_error=False, fill_value=(self.vector_velocity[0], self.vector_velocity[-1]))

    def __call__(self, t):
        """
        Возвращает интерполированное значение вектора
        arguments:t {float} -- время
        returns: {np.ndarray} - вектор
        """
        # if (max(self.ts) < t or min(self.ts) > t):
        #     raise AttributeError(f'Значение {t} выходит из диапазона [{min(self.ts)}, {max(self.ts)}]')

        return np.array(self.func_inter(t))

