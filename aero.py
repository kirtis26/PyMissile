from aero_info import *
from interpolation import Interp1d


class Aerodynamic(object):

    @classmethod
    def get_parameters(cls, missile, opts):

        @np.vectorize
        def get_x_ct(t):
            dx_ct = abs(missile.x_ct_0 - missile.x_ct_marsh) / missile.t_marsh
            if t < missile.t_marsh:
                return x_ct_0 - dx_ct * t
            else:
                return x_ct_marsh

        d = missile.d
        a = opts['a']
        x_ct_0 = opts['x_ct_0']
        x_ct_marsh = opts['x_ct_marsh']

        L_korp = opts['L_korp']
        L_cil = opts['L_cil']
        L_korm = opts['L_korm']
        L_kon1 = opts['L_kon1']
        L_kon2 = opts['L_kon2']
        d_korm = opts['d_korm']
        betta_kon2 = opts['betta_kon2']
        class_korp = opts['class_korp']

        S_oper = opts['S_oper']
        c_oper = opts['c_oper']
        L_oper = opts['L_oper']
        b_0_oper = opts['b_0_oper']
        x_b_oper = opts['x_b_oper']
        khi_pk_oper = opts['khi_pk_oper']
        khi_rul = opts['khi_rul']
        class_oper = opts['class_oper']

        # вычисление геометрии корпуса
        L_nos = L_kon1 + L_kon2
        d_kon1 = d - 2 * np.tan(np.radians(betta_kon2)) * L_kon2
        betta_kon1 = (d_kon1 / 2) / L_kon1
        S_kon1 = np.pi * d_kon1 ** 2 / 4
        S_mid = np.pi * d ** 2 / 4
        S_dno = np.pi * d_korm ** 2 / 4
        F_f = (np.pi * d_kon1 / 2 * np.sqrt((d_kon1 / 2) ** 2 + L_kon1 ** 2)) + (
                np.pi * (d_kon1 / 2 + d / 2) * np.sqrt((d / 2 - d_kon1 / 2) ** 2 + L_kon2 ** 2)) + (
                      np.pi * d * L_cil) + (
                      np.pi * (d_korm / 2 + d / 2) * np.sqrt((d / 2 - d_korm / 2) ** 2 + L_korm ** 2))
        W_nos = 1 / 3 * L_kon1 * S_mid + 1 / 3 * np.pi * L_kon2 * (
                (d_kon1 / 2) ** 2 + (d_kon1 / 2) * (d / 2) + (d / 2) ** 2)
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

        ts = np.linspace(0, missile.t_marsh, 100)
        x_ct_itr = Interp1d(ts, get_x_ct(ts))

        return cls(state=missile.state,
                   d=d,
                   d_kon1=d_kon1,
                   d_korm=d_korm,
                   L_korp=L_korp,
                   L_cil=L_cil,
                   L_nos=L_nos,
                   L_korm=L_korm,
                   L_kon1=L_kon1,
                   L_kon2=L_kon2,
                   betta_kon1=betta_kon1,
                   betta_kon2=betta_kon2,
                   S_kon1=S_kon1,
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
                   a=a,
                   x_ct_itr=x_ct_itr)

    def __init__(self, state, **kwargs):

        self.state = state
        self.x_ct_itr = kwargs['x_ct_itr']

        # Геометрия корпуса
        self.d = kwargs['d']
        self.S_mid = kwargs['S_m']
        self.S_dno = kwargs['S_dno']
        self.F_f = kwargs['F_f']
        self.W_nos = kwargs['W_nos']
        self.L_korp = kwargs['L_korp']
        self.L_cil = kwargs['L_cil']
        self.L_nos = kwargs['L_nos']
        self.L_korm = kwargs['L_korm']
        self.L_kon1 = kwargs['L_kon1']
        self.L_kon2 = kwargs['L_kon2']
        self.d_korm = kwargs['d_korm']
        self.d_kon1 = kwargs['d_kon1']
        self.S_kon1 = kwargs['S_kon1']
        self.betta_kon1 = kwargs['betta_kon1']
        self.betta_kon2 = kwargs['betta_kon2']
        self.class_korp = kwargs['class_korp']
        self.lambd_korp = kwargs['lambd_korp']
        self.lambd_nos = kwargs['lambd_nos']
        self.lambd_cil = kwargs['lambd_cil']
        self.lambd_korm = kwargs['lambd_korm']
        self.nu_korm = kwargs['nu_korm']
        self.class_korp = kwargs['class_korp']

        # Геометрия рулей (оперения)
        self.S_oper = kwargs['S_oper']
        self.S_k_oper = kwargs['S_k_oper']
        self.c_oper = kwargs['c_oper']
        self.L_oper = kwargs['L_oper']
        self.L_k_oper = kwargs['L_k_oper']
        self.L_hv_oper = kwargs['L_hv_oper']
        self.D_oper = kwargs['D_oper']
        self.a_oper = kwargs['a_oper']
        self.b_0_oper = kwargs['b_0_oper']
        self.b_a_oper = kwargs['b_a_oper']
        self.b_b_oper = kwargs['b_b_oper']
        self.x_b_oper = kwargs['x_b_oper']
        self.x_b_a_oper = kwargs['x_b_a_oper']
        self.z_a_oper = kwargs['z_a_oper']
        self.khi_pk_oper = kwargs['khi_pk_oper']
        self.khi_rul = kwargs['khi_rul']
        self.class_oper = kwargs['class_oper']
        self.K_oper = kwargs['K_oper']
        self.tg_khi_pk_oper = kwargs['tg_khi_pk_oper']
        self.tg_khi_05_oper = kwargs['tg_khi_05_oper']
        self.lambd_k_oper = kwargs['lambd_k_oper']
        self.lambd_oper = kwargs['lambd_oper']
        self.nu_oper = kwargs['nu_oper']
        self.nu_k_oper = kwargs['nu_k_oper']
        self.b_k_oper = kwargs['b_k_oper']
        self.nu_oper = kwargs['nu_oper']

    def get_aero_constants(self):
        """
        Метод, рассчитывающий аэродинамические коэффициенты ракеты в текущем состоянии state
        arguments: state {np.ndarray} -- состояние ракеты;
                                         [v,   x, y, Q,       alpha,   t]
                                         [м/с, м, м, радианы, градусы, с]
        returns: {dict} -- словарь с АД коэф-коэффициентами
        """
        v, x, y, alpha, t = self.missile.state
        Mach = v / self.missile.atm_itr(y, 4)
        nyu = self.missile.atm_itr(y, 6)
        x_ct = self.x_ct_itr(t)
        Re_korp_f = v * self.L_korp / nyu
        Re_korp_t = table_4_5(Mach, Re_korp_f, self.class_korp, self.L_korp)

        # Коэф-т подъемной силы корпуса
        if Mach <= 1:
            Cy_alpha_nos = 2 / 57.3 * (1 + 0.27 * Mach ** 2)
        else:
            Cy_alpha_nos = 2 / 57.3 * (np.cos(np.radians(self.betta_kon1)) ** 2 * self.S_kon1 / self.S_mid
                                       + np.cos(np.radians(self.betta_kon2)) ** 2 * (1 - self.S_kon1 / self.S_mid))
        Cy_alpha_korm = - 2 / 57.3 * (1 - self.nu_korm ** 2) * self.a
        Cy_alpha_korp = Cy_alpha_nos + Cy_alpha_korm

        # Коэф-т подъемной силы оперения по углу атаки
        K_t_oper = table_3_21(Mach, self.lambd_nos)
        Cy_alpha_k_oper = Cy_alpha_iz_kr(Mach * np.sqrt(K_t_oper), self.lambd_oper, self.c_oper, self.tg_khi_05_oper)
        k_aa_oper = (1 + 0.41 * self.D_oper) ** 2 * (
                (1 + 3 * self.D_oper - 1 / self.nu_k_oper * self.D_oper * (1 - self.D_oper)) / (
                1 + self.D_oper) ** 2)
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
        Cy_delt_oper = Cy_alpha_k_oper * K_delt_0_oper * n_eff

        # Коэф-т подъемной силы ракеты
        Cy_alpha = Cy_alpha_korp * (self.S_mid / self.S_mid) + Cy_alpha_oper * (self.S_oper / self.S_mid) * K_t_oper

        # Сопротивление корпуса
        x_t = Re_korp_t * nyu / v
        x_t_ = x_t / self.L_korp
        Cx_f_ = table_4_2(Re_korp_f, x_t_) / 2
        nu_m = table_4_3(Mach, x_t_)
        nu_c = 1 + 1 / self.lambd_nos
        Cx_tr = Cx_f_ * (self.F_f / self.S_mid) * nu_m * nu_c

        if Mach > 1:
            p_kon1_ = (0.0016 + 0.002 / Mach ** 2) * self.betta_kon1 ** 1.7
            p_kon2_ = (0.0016 + 0.002 / Mach ** 2) * self.betta_kon2 ** 1.7
            Cx_nos = p_kon1_ * (self.S_kon1 / self.S_mid) + p_kon2_ * (1 - (self.S_kon1 / self.S_mid))
        else:
            Cx_nos = table_4_11(Mach, self.lambd_nos)

        Cx_korm = table_4_24(Mach, self.nu_korm, self.lambd_korm)

        if self.missile.P_itr(t) == 0:
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
        Cx_ind_korp = Cy_alpha_korp * alpha ** 2 * ((1 + phi) / 57.3)

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
            Cx_oper_voln = (table_4_30(Mach, self.nu_k_oper, self.lambd_oper, self.tg_khi_05_oper, self.c_oper)) * (
                    1 + phi * (self.K_oper - 1))

        Cx_0_oper = Cx_oper_prof + Cx_oper_voln

        if Mach * np.cos(np.radians(self.khi_pk_oper)) > 1:
            Cx_ind_oper = (Cy_alpha_oper * alpha) * np.tan(np.radians(alpha))
        else:
            Cx_ind_oper = 0.38 * (Cy_alpha_oper * alpha) ** 2 / (
                    self.lambd_oper - 0.8 * (Cy_alpha_oper * alpha) * (self.lambd_oper - 1)) * (
                                  (self.lambd_oper / np.cos(np.radians(self.khi_pk_oper)) + 4) / (
                                  self.lambd_oper + 4))

        Cx_oper = Cx_0_oper + Cx_ind_oper

        Cx_0 = 1.05 * (Cx_0_korp * (self.S_mid / self.S_mid) + Cx_0_oper * K_t_oper * (self.S_oper / self.S_mid))
        Cx_ind = Cx_ind_korp * (self.S_mid / self.S_mid) + Cx_ind_oper * (self.S_oper / self.S_mid) * K_t_oper
        Cx = Cx_0 + Cx_ind

        # Центр давления корпуса
        delta_x_f = F_iz_korp(Mach, self.lambd_nos, self.lambd_cil, self.L_nos)
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
            F_oper = (1 - np.sqrt(np.pi) / (2 * b__b_oper * np.sqrt(c_const_oper)) *\
                     (table_int_ver((b__b_oper + L__hv_oper) * np.sqrt(2 * c_const_oper)) -\
                    table_int_ver(L__hv_oper * np.sqrt(2 * c_const_oper))))
            x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
            x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_ * F_oper * F_1_oper
            x_fa_oper = 1 / K_aa_oper * (x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)
        else:
            x_f_b_oper_ = x_f_iz_oper_ + 0.02 * self.lambd_oper * self.tg_khi_05_oper
            x_f_ind_oper = self.x_b_oper + self.b_b_oper * x_f_b_oper_
            x_fa_oper = 1 / K_aa_oper * (
                    x_f_iz_oper + (k_aa_oper - 1) * x_f_delt_oper + (K_aa_oper - k_aa_oper) * x_f_ind_oper)

        # Фокус оперения по углу отклонения
        x_fd_oper = 1 / K_delt_0_oper * (k_delt_0_oper * x_f_iz_oper + (K_delt_0_oper - k_delt_0_oper) * x_f_ind_oper)

        # Фокус ракеты
        x_fa = 1 / Cy_alpha * ((Cy_alpha_korp * (self.S_mid / self.S_mid) * x_fa_korp) + Cy_alpha_oper * (
                self.S_oper / self.S_mid) * x_fa_oper * K_t_oper)

        # Демпфирующие моменты АД поверхностей
        x_c_ob = self.L_korp * ((2 * (self.lambd_nos + self.lambd_cil) ** 2 - self.lambd_nos ** 2) / (
                4 * (self.lambd_nos + self.lambd_cil) * (self.lambd_nos + self.lambd_cil - 2 / 3 * self.lambd_nos)))
        m_z_wz_korp = - 2 * (1 - x_ct / self.L_korp + (x_ct / self.L_korp) ** 2 - x_c_ob / self.L_korp)

        x_ct_oper_ = (x_ct - self.x_b_a_oper) / self.b_a_oper

        mz_wz_cya_iz_kr = table_5_15(self.nu_oper, self.lambd_oper, self.tg_khi_05_oper, Mach)
        B1 = table_5_16(self.lambd_oper, self.tg_khi_05_oper, Mach)
        m_z_wz_oper = (mz_wz_cya_iz_kr - B1 * (1 / 2 - x_ct_oper_) - 57.3 * (
                1 / 2 - x_ct_oper_) ** 2) * K_aa_oper * Cy_alpha_k_oper

        m_z_wz = m_z_wz_korp * (self.S_mid / self.S_mid) * (self.L_korp / self.L_korp) ** 2 + m_z_wz_oper * (
                self.S_oper / self.S_mid) * (self.b_a_oper / self.L_korp) ** 2 * np.sqrt(K_t_oper)

        # Балансировочная зависимость
        M_z_delt = Cy_delt_oper * (x_ct - x_fd_oper) / self.L_korp
        M_z_alpha = Cy_alpha * (x_ct - x_fa) / self.L_korp
        ballans_relation = - (M_z_alpha / M_z_delt)

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
            'M_z_delt': M_z_delt
        }