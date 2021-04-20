#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate as interp
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


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
    
    def plot(self):
        """
        Визуализация интерполируемых данных
        """
        fig = plt.figure(dpi=100)
        plt.plot(self.mx, self.mf, 'k')
        plt.show()


# In[5]:


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
        
    def plot(self):
        """
        Визуализация интерполируемых данных
        """
        fig = plt.figure(dpi=100)
        ax = fig.gca(projection='3d')
        X, Y = np.meshgrid(self.mx, self.my)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i,j] = self(X[i,j], Y[i,j])
        surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlBu_r, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
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

