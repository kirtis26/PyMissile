import numpy as np
from missile import Missile
from target import Target
from interpolation import Interp1d
from math import *

class MissileGym(object):

    @classmethod
    def make_simple_scenario(cls, missile_opts, target_pos, target_vel):
        """
        Классовый метод создания простого сценария движения цели, в котором происходит инициилизация
        объектов Missile и Target, начальных параметров наведения ракеты на цель.
        arguments: missile_opts {dict} -- словарь с опциями ракеты
                   target_pos {tuple/list/np.ndarray} -- положение цели
                   target_vel {tuple/np.ndarray} -- скорость цели
        returns: {cls}
        """
        trg_pos = np.array(target_pos)
        trg_vel = np.array(target_vel)
        target  = Target.get_simple_target(trg_pos, trg_vel)
        missile_vel_abs = missile_opts['vel_abs']
        missile_pos = missile_opts['init_conditions'].get('pos_0', None)
        missile = Missile.get_missile(missile_opts)
        mparams = missile.get_parameters_of_missile_to_meeting_target(target.pos, target.vel, missile_vel_abs, missile_pos)
        missile.set_init_cond(parameters_of_missile=mparams)
        suc, meeting_point = missile.get_instant_meeting_point(target.pos, target.vel, missile_vel_abs, missile_pos if missile_pos is not None else (0,0))
        print(f'meet: {suc}; meeting point: {meeting_point}')
        return cls(missile=missile, target=target, t_max=missile_opts.get('t_max'), tau=missile_opts.get('tau', 1/30))

    def __init__(self, *args, **kwargs):
        self.point_solution = np.array([])
        self.missile = kwargs['missile']
        self.target  = kwargs['target']
        self.tau     = kwargs['tau'] 
        self.t_max   = kwargs['t_max']
        self._miss_state_len = self.missile.get_state().shape[0]
        self._trg_state_len = self.target.get_state().shape[0]
        self.prev_observation = self.get_current_observation()
 
    def reset(self):
        self.missile.reset()
        self.target.reset()
        self.prev_observation = self.get_current_observation()
        return self.get_observation()

    def get_observation(self):
        return np.concatenate([self.prev_observation, self.get_current_observation()])

    def step(self, action):
        """
        Основной метод. Сделать шаг по времени. Изменить внутреннее состояние и вернуть необходимые данные
        argument: action {float} -- управляющее действие на данном шаге
        """
        self.prev_observation = self.get_current_observation()
        mpos0, tpos0 = self.missile.pos, self.target.pos
        self.missile.step(action, self.tau)
        self.target.step(self.tau)
        obs = self.get_observation()
        mpos1, tpos1 = self.missile.pos, self.target.pos
        mvel1, tvel1 = self.missile.vel, self.target.vel
        done, info = self.get_info_about_step(mpos0, tpos0, mpos1, tpos1, tvel1, mvel1)
        return obs, done, info

    def step_with_guidance(self):
        """
        Метод, моделирующий шаг step по времени tau в зависимости от метода наведения.
        Пропорциональное сближение на активном участке и на пассивном, когда скорость ракеты больше скорости цели в 2 раза
        Метод чистой погони на пассивном участке, когда скорость ракеты менее чем в 2 раза превышает скорость цели, либо меньше её
        returns: {(np.array, bool, dict)} -- состояние окружения, флаг окончания моделирования, информация (причина, время, расстояние,...)
        """
        if self.missile.P_itr(self.missile.t) > 0:
            action_guidance = self.missile.get_action_proportional_guidance(self.target, a=10)
#             action_guidance = self.missile.get_action_alignment_guidance(self.target, tau=self.tau)
        elif (self.target.v / self.missile.v) <= 0.5:
            action_guidance = self.missile.get_action_proportional_guidance(self.target)
        elif (self.target.v / self.missile.v) >= 0.5:
            action_guidance = self.missile.get_action_chaise_guidance(self.target)
            
        obs, done, info = self.step(action_guidance)
        return obs, done, info

    def get_info_about_step(self, mpos0, tpos0, mpos1, tpos1, tvel1, mvel1):
        """
        Метод, проверяющий условия остановки шага по времени метода step
        arguments: mpos0, tpos0 -- положение ракеты и цели на текущем шаге по времени tau
                   mpos1, tpos1 -- положение ракеты и цели на следующем шаге по времени tau
                   mvel1, tvel1 -- скорость ракеты и цели на следующем шаге по времени tau
        returns: {(bool, dict)} -- флаг окончания моделирования, информация (причина, время, расстояние,...)
        """
        info = {}
        if mpos1[1] < 0:
            info['done_reason'] = 'missile fell'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mpos1 - tpos1)
            return True, info
        if self.is_hit(mpos0, tpos0, mpos1, tpos1):
            info['done_reason'] = 'target destroyed'
            info['t'] = self.missile.t
            return True, info
        if self.is_wrong_way(mpos1, mvel1, tpos1) and self.missile.P_itr(self.missile.t) == 0:
            info['done_reason'] = 'wrong way'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mpos1 - tpos1)
            return True, info
        if self.missile.t > self.t_max:
            info['done_reason'] = 'a long time to fly'
            info['t'] = self.missile.t
            return True, info
        if self.missile.t > 20 and self.missile.v < 340:
            info['done_reason'] = 'velocity is small'
            info['t'] = self.missile.t
            return True, info
        return False, {'done_reason': 'unknown'}

    def is_hit(self, mpos0, tpos0, mpos1, tpos1):
        r0 = np.linalg.norm(mpos0 - tpos0)
        r1 = np.linalg.norm(mpos1 - tpos1)
        r_kill = self.missile.r_kill
        if min(r1, r0) <= r_kill:
            return True
        return MissileGym._r1(mpos0, tpos0, mpos1, tpos1, r_kill)

    def is_wrong_way(self, mpos, mvel, tpos):
        vis_n = (tpos - mpos)
        d = np.linalg.norm(vis_n)
        if d < 300:
            return False
        vis_n /= d
#         mvel1  = mvel / np.linalg.norm(mvel)
#         maxis  = self.missile.x_axis
#         maxis1 = maxis / np.linalg.norm(maxis)
        return mvel @ vis_n < 1

    @staticmethod
    def get_overload(vel0, vel1, tau):
        g = 9.80665
        vel0 = np.array(vel0)
        vel1 = np.array(vel1)
        a = (vel1 - vel0) / tau - np.array([0, -g])
        a_tau = np.dot(a, vel0/np.linalg.norm(vel0)) * vel0/np.linalg.norm(vel0)
        a_n = a - a_tau
        n_y = copysign(np.linalg.norm(a_n)/g, np.cross(vel0, a_n))
        n_x = copysign(np.linalg.norm(a_tau)/g, np.dot(a_tau, vel0))
        return np.array([n_x, n_y])

    @staticmethod
    def _r1(mpos0, tpos0, mpos1, tpos1, r_kill):
        xm0, ym0 = mpos0
        xm1, ym1 = mpos1
        xt0, yt0 = tpos0
        xt1, yt1 = tpos1

        X_1 = (xm1 - xm0) - (xt1 - xt0)
        Y_1 = (ym1 - ym0) - (yt1 - yt0)
        A = X_1**2 + Y_1**2
        B = 2 * X_1 * (xm0 + xt0) + 2 * Y_1 * (ym0 + yt0)
        C = (xm0 + xt0)**2 + (ym0 + yt0)**2

        r0 = C
        r1 = A + B + C

        r_0 = B
        r_1 = 2*A + B
        if r_0 * r_1 >= 0:
            return min(r0, r1) <= r_kill**2
        t_0 = -B / (2*A)
        r_t0 = A * t_0**2 + B * t_0 + C
        return min(r0, r1, r_t0) <= r_kill**2

    def set_state(self, state):
        """
        Метод, задающий новое состояние (state) окружения.      
        arguments: state {np.ndarray} -- numpy-массив, в котором хранится вся необходимая информация для задания нового состояния
        returns: observation в новом состоянии
        """
        self.missile.set_state(state[:self._miss_state_len])
        self.target.set_state(state[self._miss_state_len:self._miss_state_len+self._trg_state_len])
        self.prev_observation[:] = state[self._miss_state_len+self._trg_state_len:]

    def get_state(self):
        mis_state = self.missile.get_state()
        trg_state = self.target.get_state()
        return np.concatenate([mis_state, trg_state, self.prev_observation])
    
    def get_aero_constants(self):
        """
        Метод, вычисляющий аэродинамические коэффициенты ракеты в текущем состоянии state
        returns: {dict}
        """
        mis_state = self.missile.get_state()
        return self.missile.get_aero_constants(mis_state)
    
    def get_etta(self, miss=None, target=None):
        """
        Метод, вычисляющий угол между осью ракеты и линией визирования
        returns: etta = -180..+180 {градусы}
        """
        miss = self.missile if miss is None else miss
        target = self.target if target is None else target
        Q = miss.Q
        vis = target.pos - miss.pos
        vis = vis / np.linalg.norm(vis)
        Q_vis = np.arctan2(vis[1], vis[0])
        angle = np.degrees(Q_vis - Q) % 360
        angle = (angle + 360) % 360
        if angle > 180:
            angle -= 360
        return angle

    def get_current_observation_raw(self):
        """
        Метод возвращает numpy-массив с наблюдаемыми ракетой данными в текущем состоянии окружения
        0 - t    - время, с 0..150
        1 - etta - угол, между осью ракеты и линией визирования, градусы, -180..+180 
        2 - v    - скорость ракеты, м/с, 0..2200
        3 - Q    - угол тангажа, градусы, -180..+180
        4 - alpha
        """        
        t = self.missile.t
        etta = self.get_etta()
        v = self.missile.v
        Q = np.degrees(self.missile.Q)
        alpha = self.missile.alpha
        return np.array([t, etta, v, Q, alpha])

    def get_current_observation(self):
        h = self.observation_current_raw_space_high
        l = self.observation_current_raw_space_low
        return (self.get_current_observation_raw() - l) / (h - l)

    @property
    def observation_current_raw_space_high(self):
        return np.array([self.t_max, 180.0, 2300, 180, self.missile.alpha_max])

    @property
    def observation_current_raw_space_low(self):
        return np.array([0, -180.0, 0, -180, -self.missile.alpha_max])