from aero_info import *
from interpolation import Interp1d


class Aerodynamic(object):

    @classmethod
    def get_aero_models(cls, opts):

        @np.vectorize
        def get_x_ct(t):
            dx_ct = abs(opts['x_ct_0'] - opts['x_ct_marsh']) / opts['t_marsh']
            if t < opts['t_marsh']:
                return x_ct_0 - dx_ct * t
            else:
                return x_ct_marsh

        d = opts['d']
        a = opts['a']
        x_ct_0 = opts['x_ct_0']
        x_ct_marsh = opts['x_ct_marsh']

        L_korp = opts['L_korp']
        L_nos = opts['L_korp']
        L_cil = opts['L_cil']
        L_korm = opts['L_korm']
        d_korm = opts['d_korm']
        class_korp = opts['class_korp']
        betta_kon = opts['betta_kon']

        S_oper = opts['S_oper']
        c_oper = opts['c_oper']
        L_oper = opts['L_oper']
        b_0_oper = opts['b_0_oper']
        x_b_oper = opts['x_b_oper']
        khi_pk_oper = opts['khi_pk_oper']
        khi_rul = opts['khi_rul']
        class_oper = opts['class_oper']

        S_kr = opts['S_kr']
        c_kr = opts['c_kr']
        L_kr = opts['L_kr']
        b_0_kr = opts['b_0_kr']
        x_b_kr = opts['x_b_kr']
        khi_pk_kr = opts['khi_pk_kr']
        class_kr = opts['class_kr']

        # вычисление геометрии корпуса
        S_mid = np.pi * d ** 2 / 4
        S_dno = np.pi * d_korm ** 2 / 4
        F_f = np.pi * d/2 * (d/2 + np.sqrt((d/2)**2 + (L_nos)**2)) + (np.pi * (d_korm / 2 + d / 2) * np.sqrt((d / 2 - d_korm / 2) ** 2 + L_korm ** 2))
        W_nos = 1 / 3 * L_nos * S_mid
        lambd_korp = L_korp / d
        lambd_nos = L_nos / d
        lambd_cil = L_cil / d
        lambd_korm = L_korm / d
        nu_korm = d_korm / d

        # вычисление геометрии оперения
        D_oper = d / L_oper
        L_k_oper = L_oper - d
        tg_khi_pk_oper = np.tan(np.radians(khi_pk_oper))
        lambd_oper = L_oper ** 2 / S_oper
        nu_oper = (S_oper / (L_oper * b_0_oper) - 0.5) ** (-1) / 2
        b_k_oper = b_0_oper / nu_oper
        b_a_oper = 4 / 3 * S_oper / L_oper * (1 - (nu_oper / (nu_oper + 1) ** 2))
        b_b_oper = b_0_oper * (1 - (nu_oper - 1) / nu_oper * d / L_oper)
        z_a_oper = L_oper / 6 * ((nu_oper + 2) / (nu_oper + 1))
        S_k_oper = S_oper * (1 - ((nu_oper - 1) / (nu_oper + 1)) * d / L_oper) * (1 - d / L_oper)
        nu_k_oper = nu_oper - d / L_oper * (nu_oper - 1)
        lambd_k_oper = lambd_oper * ((1 - d / L_oper) / (1 - ((nu_oper - 1) / (nu_oper + 1) * d / L_oper)))
        tg_khi_05_oper = tg_khi_pk_oper - 2 / lambd_oper * (nu_k_oper - 1) / (nu_k_oper + 1)
        a_oper = 2 / 3 * b_b_oper
        K_oper = 1 / (1 - a_oper / b_a_oper)
        L_hv_oper = L_korp - x_b_oper - b_b_oper
        if tg_khi_pk_oper == 0:
            x_b_a_oper = x_b_oper
        else:
            x_b_a_oper = x_b_oper + (z_a_oper - d / 2) * tg_khi_pk_oper

        # вычисление геометрии крыльев
        D_kr = d / L_kr
        L_k_kr = L_kr - d
        tg_khi_pk_kr = np.tan(np.radians(khi_pk_kr))
        lambd_kr = L_kr ** 2 / S_kr
        nu_kr = (S_kr / (L_kr * b_0_kr) - 0.5) ** (-1) / 2
        b_k_kr = b_0_kr / nu_kr
        b_a_kr = 4 / 3 * S_kr / L_kr * (1 - (nu_kr / (nu_kr + 1) ** 2))
        b_b_kr = b_0_kr * (1 - (nu_kr - 1) / nu_kr * d / L_kr)
        z_a_kr = L_kr / 6 * ((nu_kr + 2) / (nu_kr + 1))
        S_k_kr = S_kr * (1 - ((nu_kr - 1) / (nu_kr + 1)) * d / L_kr) * (1 - d / L_kr)
        nu_k_kr = nu_kr - d / L_kr * (nu_kr - 1)
        lambd_k_kr = lambd_kr * ((1 - d / L_kr) / (1 - ((nu_kr - 1) / (nu_kr + 1) * d / L_kr)))
        tg_khi_05_kr = tg_khi_pk_kr - 2 / lambd_kr * (nu_k_kr - 1) / (nu_k_kr + 1)
        a_kr = 2 / 3 * b_b_kr
        K_kr = 1 / (1 - a_kr / b_a_kr)
        L_hv_kr = L_korp - x_b_kr - b_b_kr
        if tg_khi_pk_kr == 0:
            x_b_a_kr = x_b_kr
        else:
            x_b_a_kr = x_b_kr + (z_a_kr - d / 2) * tg_khi_pk_kr
        x_otn_1_2 = x_b_kr - x_b_oper - b_b_oper

        ts = np.linspace(0, opts['t_marsh'], 100)
        x_ct_itr = Interp1d(ts, get_x_ct(ts))

        @np.vectorize
        def get_P(t):
            if t < opts['t_marsh']:
                return opts['t_marsh']
            else:
                return 0

        ts = np.linspace(0, opts['t_marsh'], 100)
        P_itr = Interp1d(ts, get_P(ts))

        return cls(d=d,
                   d_korm=d_korm,
                   L_korp=L_korp,
                   L_cil=L_cil,
                   L_nos=L_nos,
                   L_korm=L_korm,
                   S_mid=S_mid,
                   S_dno=S_dno,
                   F_f=F_f,
                   W_nos=W_nos,
                   class_korp=class_korp,
                   class_oper=class_oper,
                   lambd_korp=lambd_korp,
                   lambd_nos=lambd_nos,
                   lambd_cil=lambd_cil,
                   lambd_korm=lambd_korm,
                   nu_korm=nu_korm,
                   betta_kon=betta_kon,
                   S_oper=S_oper,
                   S_k_oper=S_k_oper,
                   c_oper=c_oper,
                   D_oper=D_oper,
                   L_oper=L_oper,
                   L_k_oper=L_k_oper,
                   tg_khi_pk_oper=tg_khi_pk_oper,
                   khi_pk_oper=khi_pk_oper,
                   khi_rul=khi_rul,
                   tg_khi_05_oper=tg_khi_05_oper,
                   lambd_k_oper=lambd_k_oper,
                   lambd_oper=lambd_oper,
                   nu_oper=nu_oper,
                   nu_k_oper=nu_k_oper,
                   b_k_oper=b_k_oper,
                   b_a_oper=b_a_oper,
                   b_b_oper=b_b_oper,
                   b_0_oper=b_0_oper,
                   x_b_oper=x_b_oper,
                   z_a_oper=z_a_oper,
                   a_oper=a_oper,
                   K_oper=K_oper,
                   L_hv_oper=L_hv_oper,
                   x_b_a_oper=x_b_a_oper,
                   S_kr=S_kr,
                   S_k_kr=S_k_kr,
                   c_kr=c_kr,
                   D_kr=D_kr,
                   L_kr=L_kr,
                   L_k_kr=L_k_kr,
                   tg_khi_pk_kr=tg_khi_pk_kr,
                   khi_pk_kr=khi_pk_kr,
                   tg_khi_05_kr=tg_khi_05_kr,
                   lambd_k_kr=lambd_k_kr,
                   lambd_kr=lambd_kr,
                   nu_kr=nu_kr,
                   nu_k_kr=nu_k_kr,
                   b_k_kr=b_k_kr,
                   b_a_kr=b_a_kr,
                   b_b_kr=b_b_kr,
                   b_0_kr=b_0_kr,
                   x_b_kr=x_b_kr,
                   z_a_kr=z_a_kr,
                   a_kr=a_kr,
                   K_kr=K_kr,
                   L_hv_kr=L_hv_kr,
                   x_b_a_kr=x_b_a_kr,
                   class_kr=class_kr,
                   x_otn_1_2=x_otn_1_2,
                   a=a,
                   x_ct_itr=x_ct_itr,
                   P_itr=P_itr)

    def __init__(self, **kwargs):

        self.x_ct_itr = kwargs['x_ct_itr']
        self.P_itr = kwargs['P_itr']

        # Геометрия корпуса
        self.d = kwargs['d']
        self.S_mid = kwargs['S_mid']
        self.S_dno = kwargs['S_dno']
        self.F_f = kwargs['F_f']
        self.W_nos = kwargs['W_nos']
        self.L_korp = kwargs['L_korp']
        self.L_cil = kwargs['L_cil']
        self.L_nos = kwargs['L_nos']
        self.L_korm = kwargs['L_korm']
        self.d_korm = kwargs['d_korm']
        self.class_korp = kwargs['class_korp']
        self.lambd_korp = kwargs['lambd_korp']
        self.lambd_nos = kwargs['lambd_nos']
        self.lambd_cil = kwargs['lambd_cil']
        self.lambd_korm = kwargs['lambd_korm']
        self.nu_korm = kwargs['nu_korm']
        self.class_korp = kwargs['class_korp']
        self.betta_kon = kwargs['betta_kon']

        # Геометрия рулей (оперения)
        self.S_1 = kwargs['S_oper']
        self.S_k_1 = kwargs['S_k_oper']
        self.c_1 = kwargs['c_oper']
        self.L_1 = kwargs['L_oper']
        self.L_k_1 = kwargs['L_k_oper']
        self.L_hv_1 = kwargs['L_hv_oper']
        self.D_1 = kwargs['D_oper']
        self.a_1 = kwargs['a_oper']
        self.b_0_1 = kwargs['b_0_oper']
        self.b_a_1 = kwargs['b_a_oper']
        self.b_b_1 = kwargs['b_b_oper']
        self.x_b_1 = kwargs['x_b_oper']
        self.x_b_a_1 = kwargs['x_b_a_oper']
        self.z_a_1 = kwargs['z_a_oper']
        self.khi_pk_1 = kwargs['khi_pk_oper']
        self.khi_1 = kwargs['khi_rul']
        self.class_1 = kwargs['class_oper']
        self.K_oper = kwargs['K_oper']
        self.tg_khi_pk_1 = kwargs['tg_khi_pk_oper']
        self.tg_khi_05_1 = kwargs['tg_khi_05_oper']
        self.lambd_k_1 = kwargs['lambd_k_oper']
        self.lambd_1 = kwargs['lambd_oper']
        self.nu_1 = kwargs['nu_oper']
        self.nu_k_1 = kwargs['nu_k_oper']
        self.b_k_1 = kwargs['b_k_oper']
        self.nu_1 = kwargs['nu_oper']

        # Геометрия крыльев
        self.x_otn_1_2 = kwargs['x_otn_1_2']
        self.S_2 = kwargs['S_kr']
        self.S_k_2 = kwargs['S_k_kr']
        self.c_2 = kwargs['c_kr']
        self.L_2 = kwargs['L_kr']
        self.L_k_2 = kwargs['L_k_kr']
        self.L_hv_2 = kwargs['L_hv_kr']
        self.D_2 = kwargs['D_kr']
        self.a_2 = kwargs['a_kr']
        self.b_0_2 = kwargs['b_0_kr']
        self.b_a_2 = kwargs['b_a_kr']
        self.b_b_2 = kwargs['b_b_kr']
        self.x_b_2 = kwargs['x_b_kr']
        self.x_b_a_2 = kwargs['x_b_a_kr']
        self.z_a_2 = kwargs['z_a_kr']
        self.khi_pk_2 = kwargs['khi_pk_kr']
        self.khi_2 = kwargs['khi_rul']
        self.class_2 = kwargs['class_kr']
        self.K_kr = kwargs['K_kr']
        self.tg_khi_pk_2 = kwargs['tg_khi_pk_kr']
        self.tg_khi_05_2 = kwargs['tg_khi_05_kr']
        self.lambd_k_2 = kwargs['lambd_k_kr']
        self.lambd_2 = kwargs['lambd_kr']
        self.nu_2 = kwargs['nu_kr']
        self.nu_k_2 = kwargs['nu_k_kr']
        self.b_k_2 = kwargs['b_k_kr']
        self.nu_2 = kwargs['nu_kr']

    def get_aero_const(self, RESULT):
        res = []
        for i in range(len(RESULT['t'])):
            res.append(self.get_aero_const_state((RESULT['missile']['v'][i], RESULT['missile']['y'][i], RESULT['missile']['alpha'][i], RESULT['t'][i])))
        return res

    def get_aero_const_state(self, state):
        """
        Метод, рассчитывающий аэродинамические коэффициенты ракеты в текущем состоянии state
        arguments: state {np.ndarray} -- состояние ракеты;
                                         [v,   x, y, Q,       alpha,   t]
                                         [м/с, м, м, радианы, градусы, с]
        returns: {dict} -- словарь с АД коэффициентами
        """
        v, y, alpha, t = state
        Mach = v / table_atm(y, 4)
        nyu = table_atm(y, 6)
        x_ct = self.x_ct_itr(t)

        # Коэф-т подъемной силы корпуса
        Cy1_alpha_nos = table_3_3(Mach, self.lambd_nos, self.lambd_cil) # Оживало+цилиндр
        Cy1_alpha_korm = - 2 / 57.3 * (1 - self.nu_korm ** 2) * 0.2
        Cy1_alpha_korp = Cy1_alpha_nos + Cy1_alpha_korm

        # Коэф-т подъемной силы передних поверхностей по углу атаки
        K_t_1 = table_3_21(Mach, self.lambd_nos)
        Mach_1 = Mach * np.sqrt(K_t_1)
        Cy1_alpha_k_1 = Cy_alpha_iz_kr(Mach_1, self.lambd_1, self.c_1, self.tg_khi_05_1)
        k_aa_1 = (1 + 0.41 * self.D_1) ** 2 * (
                (1 + 3 * self.D_1 - 1 / self.nu_k_1 * self.D_1 * (1 - self.D_1)) / (
                1 + self.D_1) ** 2)
        K_aa_1 = 1 + 3 * self.D_1 - (self.D_1 * (1 - self.D_1)) / self.nu_k_1
        L_1 = self.x_b_1 + (self.b_b_1 / 2)
        delta_ps = 0.093 * L_1 / self.d * (1 + 0.4 * Mach_1 + 0.147 * Mach_1**2 - 0.006 * Mach**3) / ((v * L_1 / nyu) ** (1/5))
        khi_ps = (1 - delta_ps * 2*self.D_1**2 / (1 - self.D_1**2)) * (1 - delta_ps * self.D_1 * (self.nu_k_1 - 1) / (1 - self.D_1) / (self.nu_k_1 + 1))
        khi_mach = 0.95
        khi_nos = 0.6 + 0.4 * (1 - np.exp(-0.5 * L_1 / self.d))
        Cy1_alpha_1 = Cy1_alpha_k_1 * K_aa_1 * khi_ps * khi_mach * khi_nos

        # Коэф-т подъемной силы задних поверхностей по углу атаки
        K_t_2 = table_3_22(Mach, self.x_otn_1_2)
        Cy_2 = table_3_5(Mach * np.sqrt(K_t_2), self.L_2, self.c_2, self.tg_khi_05_2)
        K_aa_2 = 1 + 3 * self.D_2 - (self.D_2 * (1 - self.D_2)) / self.nu_k_2
        eps_sr_alf = 0
        Cy1_alpha_2 = (Cy_2 * K_aa_2) * (1 - eps_sr_alf)

        # Коэф-т подъемной силы ракеты
        S_korp = self.S_mid / self.S_mid
        S_1 = self.S_1 / self.S_mid
        S_2 = self.S_2 / self.S_mid
        Cy1_alpha = Cy1_alpha_korp * S_korp + Cy1_alpha_1 * S_1 * K_t_1 + Cy1_alpha_2 * S_2 * K_t_2

        # Коэф-т подъемной силы оперения (рулей) по углу их отклонения
        K_delt_0_1 = k_aa_1
        if Mach <= 1:
            k_shch = 0.825
        elif 1 < Mach <= 1.4:
            k_shch = 0.85 + 0.15 * (Mach - 1) / 0.4
        else:
            k_shch = 0.975
        n_eff = k_shch * np.cos(np.radians(self.khi_1))
        Cy_delt_1 = Cy1_alpha_k_1 * K_delt_0_1 * n_eff

        # Сопротивление корпуса
        Re_korp_f = v * self.L_korp / nyu
        Re_korp_t = table_4_5(Mach, Re_korp_f, self.class_korp, self.L_korp)
        x_t = Re_korp_t * nyu / v
        x_t_ = x_t / self.L_korp
        Cx_f_ = table_4_2(Re_korp_f, x_t_) / 2
        nu_m = table_4_3(Mach, x_t_)
        nu_c = 1 + 1 / self.lambd_nos

        Cx_tr = Cx_f_ * (self.F_f / self.S_mid) * nu_m * nu_c * S_korp
        Cx_nos = table_4_11(Mach, self.lambd_nos)
        Cx_korm = table_4_24(Mach, self.nu_korm, self.lambd_korm)

        if self.P_itr(t) == 0:
            p_dno_ = table_p_dno_(Mach, oper=True)
            K_nu = table_k_nu(self.nu_korm, self.lambd_korm, Mach)
            Cx_dno = p_dno_ * K_nu * (self.S_dno / self.S_mid)
        else:
            Cx_dno = 0

        Cx_0_korp = Cx_tr + Cx_nos + Cx_korm + Cx_dno

        phi = -0.2 if Mach < 1 else 0.7
        Cx_ind_korp = Cy1_alpha_korp * alpha ** 2 * ((1 + phi) / 57.3)

        Cx_korp = Cx_0_korp + Cx_ind_korp

        # Сопротивление оперения
        Re_oper_f = v * self.b_a_1 / nyu
        Re_oper_t = table_4_5(Mach, Re_oper_f, self.class_1, self.b_a_1)
        x_t_oper = Re_oper_t / Re_oper_f
        C_f_oper = table_4_2(Re_oper_f, x_t_oper)
        nu_c_oper = table_4_28(x_t_oper, self.c_1)
        Cx_oper_prof = C_f_oper * nu_c_oper

        if Mach < 1.1:
            Cx_oper_voln = table_4_30(Mach, self.nu_k_1, self.lambd_1, self.tg_khi_05_1, self.c_1)
        else:
            phi = table_4_32(Mach, self.tg_khi_05_1)
            Cx_oper_voln = (table_4_30(Mach, self.nu_k_1, self.lambd_1, self.tg_khi_05_1, self.c_1)) * (1 + phi * (self.K_oper - 1))

        Cx_0_1 = Cx_oper_prof + Cx_oper_voln

        if Mach * np.cos(np.radians(self.khi_pk_1)) > 1:
            Cx_ind_1 = (Cy1_alpha_1 * alpha) * np.tan(np.radians(alpha))
        else:
            Cx_ind_1 = 0.38 * (Cy1_alpha_1 * alpha) ** 2 / (self.lambd_1 - 0.8 * (Cy1_alpha_1 * alpha) * (self.lambd_1 - 1)) * ((self.lambd_1 / np.cos(np.radians(self.khi_pk_1)) + 4) / (self.lambd_1 + 4))

        Cx_oper = Cx_0_1 + Cx_ind_1

        # Сопротивление крыльев
        Mach_2 = Mach * np.sqrt(K_t_2)
        Re_kr_f = v * self.b_a_2 / nyu
        Re_kr_t = table_4_5(Mach_2, Re_kr_f, self.class_2, self.b_a_2)
        x_t_kr = Re_kr_t / Re_kr_f
        C_f_kr = table_4_2(Re_kr_f, x_t_kr)
        nu_c_kr = table_4_28(x_t_kr, self.c_2)
        Cx_kr_prof = C_f_kr * nu_c_kr

        if Mach_2 < 1.1:
            Cx_kr_voln = table_4_30(Mach_2, self.nu_k_2, self.lambd_2, self.tg_khi_05_2, self.c_2)
        else:
            phi = table_4_32(Mach_2, self.tg_khi_05_2)
            Cx_kr_voln = (table_4_30(Mach_2, self.nu_k_2, self.lambd_2, self.tg_khi_05_2, self.c_2)) * (1 + phi * (self.K_kr - 1))

        Cx_0_2 = Cx_kr_prof + Cx_kr_voln

        if Mach_2 * np.cos(np.radians(self.khi_pk_2)) > 1:
            Cx_ind_2 = (Cy1_alpha_2 * alpha) * np.tan(np.radians(alpha))
        else:
            Cx_ind_2 = 0.38 * (Cy1_alpha_2 * alpha) ** 2 / (self.lambd_2 - 0.8 * (Cy1_alpha_2 * alpha) * (self.lambd_2 - 1)) * ((self.lambd_2 / np.cos(np.radians(self.khi_pk_2)) + 4) / (self.lambd_2 + 4))

        Cx_kr = Cx_0_2 + Cx_ind_2
        Cx_0 = 1.05 * (Cx_0_korp * S_korp + Cx_0_1 * K_t_1 * 2*S_1 + Cx_0_2 * K_t_2 * 2*S_2)
        Cx_ind = Cx_ind_korp * S_korp + Cx_ind_1 * S_1 * K_t_1 + Cx_ind_2 * S_2 * K_t_2
        Cx = Cx_0 + Cx_ind

        # # Центр давления корпуса
        # delta_x_f = F_iz_korp(Mach, self.lambd_nos, self.lambd_cil, self.L_nos)
        # x_fa_nos_cil = self.L_nos - self.W_nos / self.S_mid + delta_x_f
        # x_fa_korm = self.L_korp - 0.5 * self.L_korm
        # x_fa_korp = 1 / Cy_alpha_korp * (Cy_alpha_nos * x_fa_nos_cil + Cy_alpha_korm * x_fa_korm)
        #
        # # Фокус оперения по углу атаки
        # x_f_iz_oper_ = F_iz_kr(Mach, self.lambd_k_oper, self.tg_khi_05_oper, self.nu_k_oper)
        # x_f_iz_oper = self.x_b_a_oper + self.b_a_oper * x_f_iz_oper_
        # f1 = table_5_11(self.D_oper, self.L_k_oper)
        # x_f_delt_oper = x_f_iz_oper - self.tg_khi_05_oper * f1
        # if Mach > 1:
        #     b__b_oper = self.b_b_oper / (np.pi / 2 * self.d * np.sqrt(Mach ** 2 - 1))
        #     L__hv_oper = self.L_hv_oper / (np.pi * self.d * np.sqrt(Mach ** 2 - 1))
        #     c_const_oper = (4 + 1 / self.nu_k_oper) * (1 + 8 * self.D_oper ** 2)
        #     F_1_oper = 1 - 1 / (c_const_oper * b__b_oper ** 2) * (1 - np.exp(-c_const_oper * b__b_oper ** 2))
        #     F_oper = (1 - np.sqrt(np.pi) / (2 * b__b_oper * np.sqrt(c_const_oper)) *\
        #              (table_int_ver((b__b_oper + L__hv_oper) * np.sqrt(2 * c_const_oper)) -\
        #             table_int_ver(L__hv_oper * np.sqrt(2 * c_const_oper))))
        #     x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
        #     x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_ * F_oper * F_1_oper
        #     x_fa_oper = 1 / K_aa_oper * (x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)
        # else:
        #     x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
        #     x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_
        #     x_fa_oper = 1 / K_aa_oper * (
        #             x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)
        #
        # # Фокус оперения по углу отклонения
        # x_fd_oper = 1 / K_delt_0_oper * (k_delt_0_oper * x_f_iz_oper + (K_delt_0_oper - k_delt_0_oper) * x_f_ind_oper)
        #
        # # Фокус ракеты
        # x_fa = 1 / Cy_alpha * ((Cy_alpha_korp * (self.S_mid / self.S_mid) * x_fa_korp) + Cy_alpha_oper * (
        #         self.S_oper / self.S_mid) * x_fa_oper * K_t_oper)
        #
        # # Демпфирующие моменты АД поверхностей
        # x_c_ob = self.L_korp * ((2 * (self.lambd_nos + self.lambd_cil) ** 2 - self.lambd_nos ** 2) / (
        #         4 * (self.lambd_nos + self.lambd_cil) * (self.lambd_nos + self.lambd_cil - 2 / 3 * self.lambd_nos)))
        # m_z_wz_korp = - 2 * (1 - x_ct / self.L_korp + (x_ct / self.L_korp) ** 2 - x_c_ob / self.L_korp)
        #
        # x_ct_oper_ = (x_ct - self.x_b_a_oper) / self.b_a_oper
        #
        # mz_wz_cya_iz_kr = table_5_15(self.nu_oper, self.lambd_oper, self.tg_khi_05_oper, Mach)
        # B1 = table_5_16(self.lambd_oper, self.tg_khi_05_oper, Mach)
        # m_z_wz_oper = (mz_wz_cya_iz_kr - B1 * (1 / 2 - x_ct_oper_) - 57.3 * (
        #         1 / 2 - x_ct_oper_) ** 2) * K_aa_oper * Cy_alpha_k_oper
        #
        # m_z_wz = m_z_wz_korp * (self.S_mid / self.S_mid) * (self.L_korp / self.L_korp) ** 2 + m_z_wz_oper * (
        #         self.S_oper / self.S_mid) * (self.b_a_oper / self.L_korp) ** 2 * np.sqrt(K_t_oper)
        #
        # # Балансировочная зависимость
        # M_z_delt = Cy_delt_oper * (x_ct - x_fd_oper) / self.L_korp
        # M_z_alpha = Cy_alpha * (x_ct - x_fa) / self.L_korp
        # ballans_relation = - (M_z_alpha / M_z_delt)
        #
        # # Запас статической устойчивости
        # m_z_cy = (x_ct - x_fa) / self.L_korp
        #
        return {
            't': t,
            'y': y,
            'alpha': alpha,
            'Mach': Mach,
            'Cy_alpha': Cy1_alpha,
            'Cy_alpha_korp': Cy1_alpha_korp,
            'Cy_alpha_oper': Cy1_alpha_1,
            'Cy_alpha_kr': Cy1_alpha_2,
            'Cy_delt_oper': Cy_delt_1,
            'Cx': Cx,
            'Cx_0': Cx_0,
            'Cx_0_korp': Cx_0_korp,
            'Cx_0_oper': Cx_0_1,
            'Cx_0_kr': Cx_0_2,
            'Cx_ind': Cx_ind,
            'Cx_ind_korp': Cx_ind_korp,
            'Cx_ind_oper': Cx_ind_1,
            'Cx_ind_kr': Cx_ind_2,
            'Cx_kr': Cx_kr,
            'Cx_oper': Cx_oper,
            'Cx_korp': Cx_korp
         }
    #     'x_fa': x_fa,
    #     'x_fa_korp': x_fa_korp,
    #     'x_fa_oper': x_fa_oper,
    #     'x_fd_oper': x_fd_oper,
    #     'm_z_cy': m_z_cy,
    #     'm_z_wz': m_z_wz,
    #     'm_z_wz_korp': m_z_wz_korp,
    #     'm_z_wz_oper': m_z_wz_oper,
    #     'ballans_relation': ballans_relation,
    #     'M_z_alpha': M_z_alpha,
    #     'M_z_delt': M_z_delt