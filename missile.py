import random as rd
import numpy as np
import pandas as pd
from math import *
from aero_info import *
from interpolation import Interp1d, Interp2d, InterpVec

class Missile(object):
    
    def __init__(self, **kwargs):
        """
        Конструктор класса Missile:
         g          -- ускорение свободного падения [м / с^2] {default: 9.80665}
         dt         -- шаг интегрирования системы ОДУ движения ракеты [с] {default: 0.001}
         dny        -- запас по перегрузке [ед.] {default: 1}
         am         -- коэф-т, характеризующий быстроту реакции ракеты на манёвр цели
         S_m        -- площадь миделя [м^2] (к которой относятся АД коэффициенты)
         r_kill     -- радиус поражения боевой части ракеты [м]
         alpha_max  -- максимальный угол атаки [градусы]
         speed_change_alpha -- скорость изменения угола атаки [градусы / с]
         xi         -- коэффициент, характеризующий структуру подъёмной силы аэродинамической схемы ракеты
         Cx_itr     -- интерполятор определения коэффициента лобового сопротивления ракеты от угла атаки [градусы] и от числа Маха
         Cya_itr    -- интерполятор определения производной коэффициента подъемной силы ракеты по углам атаки от числа Маха
         m_itr      -- масса [кг] ракеты от времени [с]
         P_itr      -- тяга [Н] ракеты от времени [с]
         atm_itr    -- интерполятор параметров атмосферы
         t_0        -- начальное время интегрирования [c] {default: 0.0}
         V_0        -- начальная скорость полета ракеты (скорость выброса) [м / с] {default: 0.0}
         x_0        -- начальное положение ракеты по координате x [м] {default: 0.0}
         y_0        -- начальное положение ракеты по координате y [м] {default: 0.0}
         Q_0        -- начальное положение ракеты по углу склонения к горизонту в положительном направлении оси x [град] {default: 0.0}
         alpha_0    -- начальный угол атаки набегающего потока [град] {default: 0.0}
        """
        self.dt   = kwargs['dt']
        self.g    = kwargs['g']
        self.V_0  = kwargs['V_0']
        self.x_0  = kwargs['x_0']
        self.y_0  = kwargs['y_0']
        self.Q_0  = kwargs['Q_0']
        self.t_0  = kwargs['t_0']
        self.am   = kwargs['am']
        self.dny  = kwargs['dny']
        self.d    = kwargs['d']
        self.S_m  = kwargs['S_m']
        self.xi   = kwargs['xi']
        self.P_itr   = kwargs['P_itr']
        self.m_itr   = kwargs['m_itr']
        self.Cx_itr  = kwargs['Cx_itr']
        self.atm_itr = kwargs['atm_itr']
        self.Cya_itr = kwargs['Cya_itr']
        self.r_kill  = kwargs['r_kill']
        self.alpha_0 = kwargs['alpha_0']
        self.alpha_max          = kwargs['alpha_max']
        self.speed_change_alpha = kwargs['speed_change_alpha']
        self.alpha_targeting    = 0    
        
        self.a  = kwargs['a']
        self.x_ct_itr = kwargs['x_ct_itr']
        
        # Геометрия корпуса
        self.S_mid   = kwargs['S_m']
        self.S_dno   = kwargs['S_dno']
        self.F_f     = kwargs['F_f']
        self.W_nos   = kwargs['W_nos']
        self.L_korp  = kwargs['L_korp']
        self.L_cil   = kwargs['L_cil']
        self.L_nos   = kwargs['L_nos']
        self.L_korm  = kwargs['L_korm']
        self.L_kon1  = kwargs['L_kon1']
        self.L_kon2  = kwargs['L_kon2']
        self.d_korm  = kwargs['d_korm']
        self.d_kon1  = kwargs['d_kon1']
        self.S_kon1  = kwargs['S_kon1']
        self.betta_kon1  = kwargs['betta_kon1']
        self.betta_kon2  = kwargs['betta_kon2']
        self.class_korp  = kwargs['class_korp']
        self.lambd_korp  = kwargs['lambd_korp']
        self.lambd_nos   = kwargs['lambd_nos']
        self.lambd_cil   = kwargs['lambd_cil']
        self.lambd_korm  = kwargs['lambd_korm']
        self.nu_korm     = kwargs['nu_korm']
        self.class_korp  = kwargs['class_korp']
        
        # Геометрия рулей (оперения)
        self.S_oper       = kwargs['S_oper']
        self.S_k_oper     = kwargs['S_k_oper']
        self.c_oper       = kwargs['c_oper']
        self.L_oper       = kwargs['L_oper']
        self.L_k_oper     = kwargs['L_k_oper']
        self.L_hv_oper    = kwargs['L_hv_oper']
        self.D_oper       = kwargs['D_oper']
        self.a_oper       = kwargs['a_oper']
        self.b_0_oper     = kwargs['b_0_oper']
        self.b_a_oper     = kwargs['b_a_oper']
        self.b_b_oper     = kwargs['b_b_oper']
        self.x_b_oper     = kwargs['x_b_oper']
        self.x_b_a_oper   = kwargs['x_b_a_oper']
        self.z_a_oper     = kwargs['z_a_oper']
        self.khi_pk_oper  = kwargs['khi_pk_oper']
        self.khi_rul      = kwargs['khi_rul']
        self.class_oper   = kwargs['class_oper']
        self.K_oper       = kwargs['K_oper']
        self.tg_khi_pk_oper  = kwargs['tg_khi_pk_oper']
        self.tg_khi_05_oper  = kwargs['tg_khi_05_oper']
        self.lambd_k_oper    = kwargs['lambd_k_oper']
        self.lambd_oper      = kwargs['lambd_oper']
        self.nu_oper         = kwargs['nu_oper']
        self.nu_k_oper       = kwargs['nu_k_oper']
        self.b_k_oper        = kwargs['b_k_oper']
        self.nu_oper         = kwargs['nu_oper']
        
    @classmethod
    def get_missile(cls, dict_opts):
        """
        Классовый метод создания ракеты со всеми необходимыми аэродинамическими, массо- и тяговременными характеристиками
        arguments: dict_opts {dict} -- словарь с параметрами проектируемой ракеты
        returns:   экземпляр класса Missile {cls}
        """
        
        @np.vectorize
        def get_m(t):
            if t < t_marsh:
                return m_0 - G_marsh * t
            else:
                return m_0 - w_marsh
            
        @np.vectorize
        def get_P(t):
            if t < t_marsh:
                return P_marsh
            else:
                return 0
        
        @np.vectorize
        def get_x_ct(t):
            dx_ct = abs(x_ct_0 - x_ct_marsh) / t_marsh
            if t < t_marsh:
                return x_ct_0 - dx_ct * t
            else:
                return x_ct_marsh
        
        dt      = dict_opts.get('dt', 0.001)
        g       = dict_opts.get('g', 9.80665)
        V_0     = dict_opts['init_conditions'].get('V_0', 0.0)
        x_0     = dict_opts['init_conditions'].get('x_0', 0.0)
        y_0     = dict_opts['init_conditions'].get('y_0', 0.0)
        Q_0     = dict_opts['init_conditions'].get('Q_0', 0.0)
        alpha_0 = dict_opts['init_conditions'].get('alpha_0', 0.0)
        t_0     = dict_opts['init_conditions'].get('t_0', 0.0)
        am      = dict_opts['am']
        dny     = dict_opts.get('dny', 1)
        d       = dict_opts['d']
        m_0     = dict_opts['m_0']
        t_marsh = dict_opts['t_marsh']
        w_marsh = dict_opts['w_marsh']
        P_marsh = dict_opts['P_marsh']
        I       = dict_opts['I']
        r_kill  = dict_opts['r_kill']
        xi      = dict_opts['xi']
        alpha_max = dict_opts['alpha_max']
        speed_change_alpha = dict_opts.get('speed_change_alpha')
        
        a  = dict_opts['a']
        x_ct_0       = dict_opts['x_ct_0']
        x_ct_marsh   = dict_opts['x_ct_marsh']

        L_korp       = dict_opts['L_korp']
        L_cil        = dict_opts['L_cil']
        L_korm       = dict_opts['L_korm']
        L_kon1       = dict_opts['L_kon1']
        L_kon2       = dict_opts['L_kon2']
        d_korm       = dict_opts['d_korm']
        betta_kon2   = dict_opts['betta_kon2']
        class_korp   = dict_opts['class_korp']

        S_oper       = dict_opts['S_oper']
        c_oper       = dict_opts['c_oper']
        L_oper       = dict_opts['L_oper']
        b_0_oper     = dict_opts['b_0_oper']
        x_b_oper     = dict_opts['x_b_oper']
        khi_pk_oper  = dict_opts['khi_pk_oper']
        khi_rul      = dict_opts['khi_rul']
        class_oper   = dict_opts['class_oper']
        
        
        # вычисление геометрии корпуса
        d_kon1 = d - 2 * np.tan(np.radians(betta_kon2)) * L_kon2
        L_nos = L_kon1
        d_kon1 = d - 2 * np.tan(np.radians(betta_kon2)) * L_kon2
        betta_kon1 = (d_kon1 / 2) / L_kon1
        S_kon1 = np.pi * d_kon1**2 / 4
        S_mid   = np.pi * d ** 2 / 4
        S_dno = np.pi * d_korm**2 / 4
        F_f = (np.pi * d_kon1/2 * np.sqrt((d_kon1/2)**2 + L_kon1**2)) + (np.pi * (d_kon1/2 + d/2) * np.sqrt((d/2 - d_kon1/2)**2 + L_kon2**2)) + (np.pi * d * L_cil) + (np.pi * (d_korm/2 + d/2) * np.sqrt((d/2 - d_korm/2)**2 + L_korm**2))
        W_nos = 1/3 * L_kon1 * S_mid + 1/3 * np.pi * L_kon2 * ((d_kon1/2)**2 + (d_kon1/2)*(d/2) + (d/2)**2)
        lambd_korp = L_korp / d
        lambd_nos = L_nos / d
        lambd_cil = L_cil / d
        lambd_korm = L_korm / d
        nu_korm = d_korm / d
        
        
        # вычисление геометрии оперения
        D_oper = d / L_oper
        L_k_oper = L_oper - d
        tg_khi_pk_oper = np.tan(np.radians(khi_pk_oper))
        lambd_oper = L_oper**2 / S_oper
        nu_oper = (S_oper / (L_oper * b_0_oper) - 0.5) ** (-1) / 2
        b_k_oper = b_0_oper / nu_oper
        b_a_oper = 4 / 3 * S_oper / L_oper * (1 - (nu_oper / (nu_oper + 1) ** 2))
        b_b_oper = b_0_oper * (1 - (nu_oper - 1) / nu_oper * d / L_oper)
        z_a_oper = L_oper / 6 * ((nu_oper + 2) / (nu_oper + 1))
        S_k_oper = S_oper * (1 - ((nu_oper - 1) / (nu_oper + 1)) * d / L_oper) * (1 - d / L_oper)
        nu_k_oper = nu_oper - d / L_oper * (nu_oper - 1)
        lambd_k_oper = lambd_oper * ((1 - d / L_oper) / (1 - ((nu_oper - 1) / (nu_oper + 1) * d / L_oper)))
        tg_khi_05_oper = tg_khi_pk_oper - 2 / lambd_oper * (nu_k_oper - 1) / (nu_k_oper + 1)
        a_oper = 2/3 * b_b_oper
        K_oper = 1 / (1 - a_oper / b_a_oper)
        L_hv_oper = L_korp - x_b_oper - b_b_oper
        if tg_khi_pk_oper == 0:
            x_b_a_oper = x_b_oper
        else:
            x_b_a_oper = x_b_oper + (z_a_oper - d / 2) * tg_khi_pk_oper
        
        
        G_marsh = w_marsh / t_marsh
          
        ts    = np.linspace(0, t_marsh, 100)
        m_itr = Interp1d(ts, get_m(ts))
        P_itr = Interp1d(ts, get_P(ts))
        x_ct_itr = Interp1d(ts, get_x_ct(ts))
        
        df1 = pd.read_csv('data_constants/cya_from_mach.csv', sep = ";")
        df2 = pd.read_csv('data_constants/cx_from_mach_and_alpha.csv', sep = ";", index_col=0)
        arr_alpha = np.array(df2.index)
        arr_mach = df1['Mach']
        arr_cya = df1['Cya']
        arr_cx = df2.to_numpy()

        Cx_itr  = Interp2d(arr_alpha, arr_mach, arr_cx)
        Cya_itr = Interp1d(arr_mach, arr_cya)

        missile = cls(
            dt       = dt,
            g        = g,
            V_0      = V_0,
            x_0      = x_0,
            y_0      = y_0,
            Q_0      = Q_0,
            t_0      = t_0,
            am       = am,
            dny      = dny,
            S_m      = S_mid,
            alpha_0  = alpha_0,
            m_itr    = m_itr,
            P_itr    = P_itr,
            x_ct_itr = x_ct_itr,
            alpha_max = alpha_max,
            speed_change_alpha = speed_change_alpha,
            xi       = xi,  
            Cx_itr   = Cx_itr,
            atm_itr  = table_atm,
            Cya_itr  = Cya_itr,
            r_kill   = r_kill,
            a      = a,
            d      = d,
            d_kon1 = d_kon1,
            d_korm = d_korm,
            L_korp = L_korp,
            L_cil  = L_cil,
            L_nos  = L_nos,
            L_korm = L_korm,
            L_kon1 = L_kon1,
            L_kon2 = L_kon2,
            betta_kon1 = betta_kon1,
            betta_kon2 = betta_kon2,
            S_kon1 = S_kon1,
            S_mid  = S_mid,
            S_dno  = S_dno,
            F_f    = F_f,
            W_nos  = W_nos,
            class_korp = class_korp,
            class_oper = class_oper,
            lambd_korp = lambd_korp,
            lambd_nos  = lambd_nos,
            lambd_cil  = lambd_cil,
            lambd_korm = lambd_korm,
            nu_korm  = nu_korm,
            S_oper   = S_oper,
            S_k_oper = S_k_oper,
            c_oper   = c_oper,
            D_oper   = D_oper,
            L_oper   = L_oper,
            L_k_oper = L_k_oper,
            tg_khi_pk_oper = tg_khi_pk_oper,
            khi_pk_oper    = khi_pk_oper,
            khi_rul        = khi_rul,
            tg_khi_05_oper = tg_khi_05_oper,
            lambd_k_oper   = lambd_k_oper,
            lambd_oper     = lambd_oper,
            nu_oper    = nu_oper,
            nu_k_oper  = nu_k_oper,
            b_k_oper   = b_k_oper,
            b_a_oper   = b_a_oper,
            b_b_oper   = b_b_oper,
            b_0_oper   = b_0_oper,
            x_b_oper   = x_b_oper,
            z_a_oper   = z_a_oper,
            a_oper     = a_oper,
            K_oper     = K_oper,
            L_hv_oper  = L_hv_oper,
            x_b_a_oper = x_b_a_oper,
        )
        
        return missile

    def get_standart_parameters_of_missile(self):
        """
        Метод, возвращающий начальное состояние ракеты
        returns: {np.ndarray} -- [v,   x, y,       Q,   alpha, t]
                                 [м/с, м, м, радианы, градусы, с]
        """
        return np.array([self.V_0, self.x_0, self.y_0, np.radians(self.Q_0), self.alpha_0, self.t_0])     
        
    def set_init_cond(self, parameters_of_missile=None):
        """
        Метод, задающий начальные параметры (положение, скорость, углы ...)
        arguments: parameters_of_missile {list / np.ndarray}
                   [v,   x, y,       Q,   alpha, t]
                   [м/с, м, м, радианы, градусы, c]
        """
        if parameters_of_missile is None:
            parameters_of_missile = self.get_standart_parameters_of_missile()
        self.state   = np.array(parameters_of_missile)
        self.state_0 = np.array(parameters_of_missile)

    def reset(self):
        """
        Метод, устанавливающий ракету в начальное состояние state_0
        """
        self.set_state(self.state_0)

    def get_state(self):
        """
        Метод получения вектора со всеми параметрами системы 
        returns: {np.ndarray} -- [v,   x, y, Q,       alpha,   t]
                                 [м/с, м, м, радианы, градусы, с]
        """
        return self.state
    
    def get_state_0(self):
        """
        Метод получения вектора со всеми параметрами системы в начальном состоянии
        returns: {np.ndarray} -- [v,   x, y, Q,       alpha,   t]
                                 [м/с, м, м, радианы, градусы, с]
        """
        return self.state_0

    def set_state(self, state):
        """
        Метод задания нового (может полностью отличающегося от текущего) состояния ракеты
        arguments: state {np.ndarray} -- [v,   x, y, Q,       alpha,   t]
                                         [м/с, м, м, радианы, градусы, с]
        """
        self.state = np.array(state)

    @property
    def pos(self):
        """
        Свойство, возвращающее текущее положение ц.м. ракеты
        returns: np.array([x, y])
        """
        return np.array([self.state[1], self.state[2]])

    @property
    def vel(self):
        """
        Свойство, возвращающее текущий вектор скорости ракеты
        returns: np.array([Vx, Vy])
        """
        v = self.state[0]
        Q = self.state[3]
        return np.array([v * np.cos(Q), v * np.sin(Q)])

    @property
    def x_axis(self):
        """
        Свойство, возвращающее текущий нормированный вектор центральной оси ракеты
        returns: np.array([Axi_x, Axi_y])
        """
        Q = self.Q
        alpha = np.radians(self.alpha)
        return np.array([np.cos(Q  + alpha), np.sin(Q + alpha)])

    @property
    def v(self):
        return self.state[0]

    @property
    def x(self):
        return self.state[1]

    @property
    def y(self):
        return self.state[2]

    @property
    def Q(self):
        return self.state[3]

    @property
    def alpha(self):
        return self.state[4]
    
    @property
    def t(self):
        return self.state[5]

    @property
    def M(self):
        return self.v / self.atm_itr(self.y, 4)

    @property
    def Cya(self):
        return self.Cya_itr(self.M)  

    @property
    def Cx(self):
        return self.Cx_itr(self.alpha, self.M)       
    
    def step(self, action, tau):
        """
        Метод моделирования динамики ракеты за шаг по времени tau. На протяжении tau управляющее воздействие на ракету постоянно (action)
        Меняет внутреннее состояние ракеты state на момент окончания шага
        arguments: action {int} -- управляющее воздействие на протяжении шага
                   tau {float}  -- длина шага по времени (не путать с шагом интегрирования)
        """
        self.alpha_targeting = self.alpha_max * action

        y = self.validate_y(self.state[:-1])
        t = self.state[-1]  
        t_end = t + tau

        flag = True
        while flag:
            if t_end - t > self.dt:
                dt = self.dt 
            else:
                dt = t_end - t
                flag = False
            k1 = self.f_system(t, y)
            k2 = self.f_system(t + 0.5 * dt, self.validate_y(y + 0.5 * dt * k1))
            k3 = self.f_system(t + 0.5 * dt, self.validate_y(y + 0.5 * dt * k2))
            k4 = self.f_system(t + dt, self.validate_y(y + dt * k3))
            t += dt
            y  = self.validate_y(y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
        self.set_state(np.append(y, t))
        
    def f_system(self, t, y):
        """
        Функция правой части системы ОДУ динамики ракеты
        arguments: t {float}      -- время
                   y {np.ndarray} -- вектор состояния системы 
                                     [v,   x, y, Q,       alpha   ]
                                     [м/с, м, м, радианы, градусы ]                        
        returns: {np.ndarray}     -- dy/dt
                                     [dv,    dx,  dy,  dQ,        dalpha   ]
                                     [м/с^2, м/c, м/c, радианы/c, градусы/c]
        """
        v, x, y, Q, alpha = y
        P   = self.P_itr(t)
        m   = self.m_itr(t)
        rho = self.atm_itr(y, 3)
        a   = self.atm_itr(y, 4)
        M   = v / a
        Cya = self.Cya_itr(M) 
        Cx  = self.Cx_itr(alpha, M)

        alpha_diff = self.alpha_targeting - alpha
        if abs(alpha_diff) < 1e-6:
            dalpha = 0
        elif alpha_diff > 0:
            dalpha = self.speed_change_alpha
        else:
            dalpha = -self.speed_change_alpha

        return np.array([
            (P * np.cos(np.radians(alpha)) - rho * v ** 2 / 2 * self.S_m * Cx - m * self.g * np.sin(Q)) / m,
            v * np.cos(Q),
            v * np.sin(Q),
            (alpha * ( Cya * rho * v ** 2 / 2 * self.S_m * (1 + self.xi) + P / 57.3) / ( m * self.g ) - np.cos(Q)) * self.g / v,
            dalpha
            ], copy = False) 

    def validate_y(self, y):
        """
        Проверка значений углов вектора y
        """
        if y[4] > self.alpha_max:
            y[4] = self.alpha_max
        elif y[4] < -self.alpha_max:
            y[4] = -self.alpha_max
        elif abs(y[4] - self.alpha_targeting) < 1e-4:
            y[4] = self.alpha_targeting
            
        if y[3] < -180 or y[3] > 180:
            y[3] = y[3] % (2 * pi)
            y[3] = (y[3] + 2 * pi) % (2 * pi)
            if y[3] > pi:
                y[3] -= 2*pi
        return y

    def get_action_proportional_guidance(self, target):
        """
        Метод, возвращающий аналог action'a, соответствующий методу пропорциональной навигации
        arguments: target {object} -- ссылка на цель. Обязательно должен иметь два свойства: pos->np.ndarray и vel->np.ndarray. 
                                      pos возвращает координату цели, vel -- скорость
        returns: {float}           -- [-1; 1] action. Если умножить его на self.alpha_max, 
                                      то получится потребный угол атаки для обеспечения метода параллельного сближения
        """
        am  = self.am
        dny = self.dny

        xc, yc = target.pos
        Qc     = target.Q
        vc     = target.v
    
        v, x, y, Q, alpha, t = self.state
        P   = self.P_itr(t)
        m   = self.m_itr(t)
        ro  = self.atm_itr(y, 3)
        a   = self.atm_itr(y, 4)
        M   = v / a
        Cya = self.Cya_itr(M)

        vis = target.pos - self.pos
        fi  = np.arctan2(vis[1], vis[0])
        r   = np.linalg.norm(vis)
        
        vel_c_otn     = target.vel - self.vel
        vis1          = vis / r
        vel_c_otn_tau = vis1 * np.dot(vis1, vel_c_otn)
        vel_c_otn_n   = vel_c_otn - vel_c_otn_tau

        dfi_dt = copysign(np.linalg.norm(vel_c_otn_n) / r, np.cross(vis1, vel_c_otn_n))

        dQ_dt = am * dfi_dt
        nya = v * dQ_dt / self.g + np.cos(Q) + dny
        alpha_req = (nya *  m * self.g) / (Cya * ro * v ** 2 / 2 * self.S_m * (1 + self.xi) + P / 57.3)

        return alpha_req / self.alpha_max

    def get_action_chaise_guidance(self, target, t_corr=1/30, dny=1):
        """
        Метод, возвращающий аналог action'a, соответствующий идельному методу чистой погони
        arguments: target {object} -- ссылка на цель. Обязательно должен иметь два свойства: pos->np.ndarray и vel->np.ndarray. 
                                      Эти свойства аналогичны свойствам этого класса: pos возвращает координату цели, vel - скорость
        returns: {float}           -- [-1; 1] аналог action'a, только не int, а float. Если умножить его на self.alphamax, то получится
                                      потребный угол атаки для обеспечения метода параллельного сближения
        """

        xc, yc = target.pos
        Qc = target.Q
        vc = target.v

        v, x, y, Q, alpha, t = self.state
        P   = self.P_itr(t)
        m   = self.m_itr(t)
        rho = self.atm_itr(y, 3)
        a   = self.atm_itr(y, 4)
        M   = v / a
        Cya = self.Cya_itr(M)

        vis = target.pos + vc * t_corr - self.pos
        fi2 = np.arctan2(vis[1], vis[0])
        fi1 = Q

        dQ_dt = (fi2 - fi1) / t_corr
        nya = v * dQ_dt / self.g + np.cos(Q) + dny
        alpha_req = (nya * m * self.g) / (Cya * rho * v ** 2 / 2 * self.S_m * (1 + self.xi) + P / 57.3)

        return alpha_req / self.alpha_max
       
    def get_parameters_of_missile_to_meeting_target(self, trg_pos, trg_vel, missile_vel_abs, missile_pos=None):
        """
        Метод, возвращающий состояние ракеты, которая нацелена на мгновенную точку встречи с целью
        arguments: trg_vel {tuple/list/np.ndarray} -- вектор скорости цели
                   trg_pos {tuple/list/np.ndarray} -- положение цели
                   my_vel_abs {float}              -- средняя скорость ракеты
        keyword arguments: my_pos {tuple/list/np.ndarray} -- начальное положение ракеты, если не указано, то (0,0) (default: {None})
        returns: [np.ndarray] -- [v, x, y, Q, alpha, t]
        """
        trg_vel = np.array(trg_vel)
        trg_pos = np.array(trg_pos)
        missile_pos = np.array(missile_pos) if missile_pos else np.array([0, 0])
        suc, meeting_point = self.get_instant_meeting_point(trg_pos, trg_vel, missile_vel_abs, missile_pos)
        vis = meeting_point - missile_pos
        Q = np.arctan2(vis[1], vis[0])
        return np.array([self.V_0, missile_pos[0], missile_pos[1], Q, self.alpha_0, self.t_0])
    
    @staticmethod
    def get_instant_meeting_point(trg_pos, trg_vel, my_vel_abs, my_pos):
        """
        Метод нахождения мгновенной точки встречи ракеты с целью (с координатой trg_pos и скоростью trg_vel)
        arguments: trg_pos {tuple/np.ndarray} -- координата цели
                   trg_vel {tuple/np.ndarray} -- скорость цели
                   my_vel_abs {float}         -- скорость ракеты
                   my_pos {tuple/np.ndarray}  -- положение ракеты
        retuns: (bool, np.ndarray) - (успех/неуспех, координата точки)
        """
        trg_pos = np.array(trg_pos)
        trg_vel = np.array(trg_vel)
        my_pos  = np.array(my_pos) 

        vis  = trg_pos - my_pos
        vis1 = vis / np.linalg.norm(vis)

        trg_vel_tau = np.dot(trg_vel, vis1) * vis1
        trg_vel_n   = trg_vel - trg_vel_tau
        trg_vel_n1  = trg_vel_n / np.linalg.norm(trg_vel_n)

        if np.linalg.norm(trg_vel_n) > my_vel_abs:
            return False, trg_pos

        my_vel_n   = trg_vel_n
        my_vel_tau = vis1 * sqrt(my_vel_abs**2 - np.linalg.norm(my_vel_n)**2)

        vel_close = my_vel_tau - trg_vel_tau
        if np.dot(vis1, vel_close) <= 0:
            return False, trg_pos

        t = np.linalg.norm(vis) / np.linalg.norm(vel_close)
        
        return True, trg_pos + trg_vel * t
    
    
    def get_aero_constants(self, state):
        """
        Метод, расчитывающий аэродинамические коэффициенты ракеты в текущем состоянии state
        arguments: state {np.ndarray} -- состояние ракеты; 
                                         [v,   x, y, Q,       alpha,   t]
                                         [м/с, м, м, радианы, градусы, с]
        returns: {dict}               -- словарь с АД коэф-тами 
        """ 
        v, x, y, alpha, t = state[0], state[1], state[2], state[4], state[5],
        Mach = v / self.atm_itr(y, 4)
        nyu = self.atm_itr(y, 6)
        x_ct = self.x_ct_itr(t)
        Re_korp_f = v * self.L_korp / nyu
        Re_korp_t = table_4_5(Mach, Re_korp_f, self.class_korp, self.L_korp)
    
        # Коэф-т подъемной силы корпуса
        if Mach <= 1:
            Cy_alpha_nos = 2 / 57.3 * (1 + 0.27 * Mach**2)
        else:
            Cy_alpha_nos = 2 / 57.3 * (np.cos(np.radians(self.betta_kon1))**2 * self.S_kon1 / self.S_mid
                                   + np.cos(np.radians(self.betta_kon2))**2 * (1 - self.S_kon1 / self.S_mid))
        Cy_alpha_korm = - 2 / 57.3 * (1 - self.nu_korm ** 2) * self.a
        Cy_alpha_korp = Cy_alpha_nos + Cy_alpha_korm
    
    
        # Коэф-т подъемной силы оперения по углу атаки
        K_t_oper = table_3_21(Mach, self.lambd_nos)
        Cy_alpha_k_oper = Cy_alpha_iz_kr(Mach * np.sqrt(K_t_oper), self.lambd_oper, self.c_oper, self.tg_khi_05_oper)
        k_aa_oper = (1 + 0.41 * self.D_oper)**2 * ((1 + 3 * self.D_oper - 1 / self.nu_k_oper * self.D_oper * (1 - self.D_oper)) / (1 + self.D_oper)**2)
        K_aa_oper = 1 + 3 * self.D_oper - (self.D_oper * (1 - self.D_oper)) / self.nu_k_oper
        Cy_alpha_oper = Cy_alpha_k_oper * K_aa_oper
    
    
        # Коэф-т подъемной силы оперения (рулей) по углу их отклонения
        K_delt_0_oper = k_aa_oper
        k_delt_0_oper = k_aa_oper ** 2 / K_aa_oper
        if Mach <= 1:
            k_shch = 0.825
        elif 1 < Mach <= 1.4:
            k_shch = 0.85 + 0.15 * (Mach - 1) / 0.4
        else:
            k_shch = 0.975
        n_eff = k_shch * np.cos(np.radians(self.khi_rul))
        Сy_delt_oper = Cy_alpha_k_oper * K_delt_0_oper * n_eff
    
    
        # Коэф-т подъемной силы ракеты
        Cy_alpha = Cy_alpha_korp * (self.S_mid / self.S_mid)  + Cy_alpha_oper * (self.S_oper / self.S_mid) * K_t_oper    
      
    
        # Сопротивление корпуса
        x_t = Re_korp_t * nyu / v
        x_t_ = x_t / self.L_korp
        Cx_f_ = table_4_2(Re_korp_f, x_t_) / 2
        nu_m = table_4_3(Mach, x_t_)
        nu_c = 1 + 1 / self.lambd_nos
        Cx_tr = Cx_f_ * (self.F_f / self.S_mid) * nu_m * nu_c
    
        if Mach > 1:
            p_kon1_ = (0.0016 + 0.002 / Mach**2) * self.betta_kon1**1.7
            p_kon2_ = (0.0016 + 0.002 / Mach**2) * self.betta_kon2**1.7
            Cx_nos = p_kon1_ * (self.S_kon1 / self.S_mid) + p_kon2_ * (1 - (self.S_kon1 / self.S_mid))
        else:
            Cx_nos = table_4_11(Mach, self.lambd_nos)

        Cx_korm = table_4_24(Mach, self.nu_korm, self.lambd_korm)
    
        if self.P_itr(t) == 0:
            p_dno_ = table_p_dno_(Mach, oper=True)
            K_nu = table_k_nu(self.nu_korm, self.lambd_korm, Mach)
            Cx_dno = p_dno_ * K_nu * (self.S_dno / self.S_mid)
        else:
            Cx_dno = 0
        Cx_0_korp = Cx_tr + Cx_nos + Cx_korm + Cx_dno
    
        if Mach < 1:
            phi = -0.2
        else:
            phi = 0.7
        Cx_ind_korp = Cy_alpha_korp * alpha**2 * ((1 + phi) / 57.3)
    
        Cx_korp = Cx_0_korp + Cx_ind_korp
    
    
        # Сопротивление оперения
        Re_oper_f = v * self.b_a_oper / nyu
        Re_oper_t = table_4_5(Mach, Re_oper_f, self.class_oper, self.b_a_oper)
        x_t_oper = Re_oper_t / Re_oper_f
        C_f_oper = table_4_2(Re_oper_f, x_t_oper)
        nu_c_oper = table_4_28(x_t_oper, self.c_oper)
        Cx_oper_prof = C_f_oper * nu_c_oper
    
        if Mach < 1.1:
            Cx_oper_voln = table_4_30(Mach, self.nu_k_oper, self.lambd_oper, self.tg_khi_05_oper, self.c_oper)
        else:
            phi = table_4_32(Mach, self.tg_khi_05_oper)
            Cx_oper_voln = (table_4_30(Mach, self.nu_k_oper, self.lambd_oper, self.tg_khi_05_oper, self.c_oper)) * (1 + phi * (self.K_oper - 1))
        
        Cx_0_oper = Cx_oper_prof + Cx_oper_voln
    
        if Mach * np.cos(np.radians(self.khi_pk_oper)) > 1:
            Cx_ind_oper = (Cy_alpha_oper * alpha) * np.tan(np.radians(alpha))
        else:
            Cx_ind_oper = 0.38 * (Cy_alpha_oper * alpha)**2 / (self.lambd_oper - 0.8 * (Cy_alpha_oper * alpha) * (self.lambd_oper - 1)) *            ((self.lambd_oper / np.cos(np.radians(self.khi_pk_oper)) + 4) / (self.lambd_oper + 4))
        
        Cx_oper = Cx_0_oper + Cx_ind_oper
    
        Cx_0 = 1.05 * (Cx_0_korp * (self.S_mid / self.S_mid) + Cx_0_oper * K_t_oper * (self.S_oper / self.S_mid))
        Cx_ind = Cx_ind_korp * (self.S_mid / self.S_mid) + Cx_ind_oper * (self.S_oper / self.S_mid) * K_t_oper
        Cx = Cx_0 + Cx_ind
    
    
        # Центр давления корпуса
        delta_x_f = F_iz_korp(Mach, self.lambd_nos, self.lambd_korp, self.L_nos)
        x_fa_nos_cil = self.L_nos - self.W_nos / self.S_mid + delta_x_f
        x_fa_korm = self.L_korp - 0.5 * self.L_korm   
        x_fa_korp = 1 / Cy_alpha_korp * (Cy_alpha_nos * x_fa_nos_cil + Cy_alpha_korm * x_fa_korm)
    
    
        # Фокус оперения по углу атаки
        x_f_iz_oper_ = F_iz_kr(Mach, self.lambd_k_oper, self.tg_khi_05_oper, self.nu_k_oper)
        x_f_iz_oper = self.x_b_a_oper + self.b_a_oper * x_f_iz_oper_
        f1 = table_5_11(self.D_oper, self.L_k_oper)
        x_f_delt_oper = x_f_iz_oper - self.tg_khi_05_oper * f1
        if Mach > 1:
            b__b_oper = self.b_b_oper / (np.pi / 2 * self.d * np.sqrt(Mach ** 2 - 1))
            L__hv_oper = self.L_hv_oper / (np.pi * self.d * np.sqrt(Mach ** 2 - 1))
            c_const_oper = (4 + 1 / self.nu_k_oper) * (1 + 8 * self.D_oper ** 2)
            F_1_oper = 1 - 1 / (c_const_oper * b__b_oper ** 2) * (1 - np.exp(-c_const_oper * b__b_oper ** 2))
            F_oper = 1 - np.sqrt(np.pi) / (2 * b__b_oper * np.sqrt(c_const_oper)) * (table_int_ver((b__b_oper + L__hv_oper) *                    np.sqrt(2 * c_const_oper)) - table_int_ver(L__hv_oper * np.sqrt(2 * c_const_oper)))
            x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
            x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_ * F_oper * F_1_oper
            x_fa_oper = 1 / K_aa_oper * (x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)
        else:
            x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
            x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_
            x_fa_oper = 1 / K_aa_oper * (x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)

        
        # Фокус оперения по углу отклонения
        x_fd_oper = 1 / K_delt_0_oper * (k_delt_0_oper * x_f_iz_oper + (K_delt_0_oper - k_delt_0_oper) * x_f_ind_oper)
    
    
        # Фокус ракеты
        x_fa = 1 / Cy_alpha * ((Cy_alpha_korp * (self.S_mid / self.S_mid) * x_fa_korp) + Cy_alpha_oper * (self.S_oper / self.S_mid) * x_fa_oper * K_t_oper)
    
    
        # Демпфирующие моменты АД поверхностей
        x_c_ob = self.L_korp * ((2 * (self.lambd_nos + self.lambd_cil)**2 - self.lambd_nos**2) / (4 * (self.lambd_nos + self.lambd_cil) * (self.lambd_nos + self.lambd_cil - 2 / 3 * self.lambd_nos)))
        m_z_wz_korp = - 2 * (1 - x_ct / self.L_korp + (x_ct / self.L_korp) ** 2 - x_c_ob / self.L_korp)

        x_ct_oper_ = (x_ct - self.x_b_a_oper) / self.b_a_oper
        m_z_wz_oper = -57.3 * (Cy_alpha_oper * (x_ct_oper_ - 1 / 2)**2 * K_aa_oper)

        m_z_wz = m_z_wz_korp * (self.S_mid / self.S_mid) * (self.L_korp / self.L_korp)**2 + m_z_wz_oper * (self.S_oper / self.S_mid) * (self.b_a_oper / self.L_korp) * np.sqrt(K_t_oper)
    

        # Балансировочная зависимость
        M_z_delt_oper = Сy_delt_oper * (x_ct - x_fd_oper) / self.L_korp
        M_z_alpha = Cy_alpha * (x_ct - x_fa) / self.L_korp
        ballans_relation = - (M_z_alpha / M_z_delt_oper)

    
        # Запас статической устойчивости
        m_z_cy = (x_ct - x_fa) / self.L_korp
    
    
        return {
            't': t,
            'x': x,
            'y': y,
            'alpha': alpha,
            'Mach': Mach,
            'Cy_alpha': Cy_alpha,
            'Cy_alpha_korp': Cy_alpha_korp,
            'Cy_alpha_oper': Cy_alpha_oper,
            'Cx': Cx,
            'Cx_0': Cx_0,
            'Cx_0_korp': Cx_0_korp,
            'Cx_0_oper': Cx_0_oper,
            'Cx_ind': Cx_ind,
            'Cx_ind_korp': Cx_ind_korp,
            'Cx_ind_oper': Cx_ind_oper,
            'x_fa': x_fa,
            'x_fa_korp': x_fa_korp,
            'x_fa_oper': x_fa_oper,
            'x_fd_oper': x_fd_oper,
            'm_z_cy': m_z_cy,
            'm_z_wz': m_z_wz,
            'm_z_wz_korp': m_z_wz_korp,
            'm_z_wz_oper': m_z_wz_oper,
            'ballans_relation': ballans_relation,
            'M_z_alpha': M_z_alpha,
            'M_z_delt_oper': M_z_delt_oper
        }
    
    def get_summary(self):
        """
        Метод возвращающий словарь с основными текущими параметрами и характеристиками ракеты в данный момент
        returns: {dict}
        """
        return { 
            't': self.t,
            'v': self.v,
            'x': self.x,
            'y': self.y,
            'Q': np.degrees(self.Q),
            'm': self.m_itr(self.t),
            'P': self.P_itr(self.t),
            'alpha': self.alpha,
            'alpha_targeting': self. alpha_targeting,
            'Cx': self.Cx_itr(self.alpha, self.M), 
            'Cya': self.Cya_itr(self.M)
        } 