import numpy as np
from missile import Missile2D
from target import Target2D
from math import *
from IPython.display import clear_output


class MissileGym(object):
    scenario_names = {}  # TODO: сделать сэт сценариев

    def __init__(self, **kwargs):
        self.missile = kwargs['missile']
        self.target = kwargs['target']
        self.tau = kwargs['tau']
        self.dt = kwargs['dt']
        self.t_max = kwargs['t_max']
        self._miss_state_len = self.missile.get_state().shape[0]
        self._trg_state_len = self.target.get_state().shape[0]

    @classmethod
    def make(cls, missile_opts, scenario_name):
        if scenario_name not in cls.scenario_names:
            raise AttributeError(
                f'Error! Unknown scenario: "{scenario_name}" \n Available scenarios: {cls.scenario_names}')
        elif scenario_name == 'standart':
            target = Target2D.get_target()
            missile = Missile2D.get_missile(missile_opts)
        elif scenario_name == 'sc_simple_1':
            pass
        # TODO: дописать все сценарии
        missile_parameters = Missile2D.get_parameters_of_missile_to_meeting_target(target.pos, target.vel, missile.vel_max/2)
        missile.set_init_parameteres(parameters=missile_parameters)
        return cls(missile=missile, target=target)

    @classmethod
    def make_simple_scenario(cls, missile_opts, target_pos, target_vel, tau=0.1, t_max=100, n=10):
        """
        Классовый метод создания простого сценария движения цели, в котором происходит инициилизация
        объектов Missile и Target, начальных параметров наведения ракеты на цель.
        Arguments: missile_opts {dict} -- словарь с опциями ракеты
                   target_pos {tuple/list/np.ndarray} -- положение цели
                   target_vel {tuple/np.ndarray} -- скорость цели
        Returns: {cls}
        """
        dt = tau / n
        target = Target2D.get_simple_target(np.array(target_pos), np.array(target_vel))
        mis_vel_abs = missile_opts.get('vel_max', None)
        mis_vel_abs = None if mis_vel_abs is None else mis_vel_abs / 2
        mis_pos = missile_opts['init_conditions'].get('pos_0', None)
        missile = Missile2D.get_missile(missile_opts)
        mis_params = missile.get_parameters_of_missile_to_meeting_target(target.pos, target.vel, mis_vel_abs, mis_pos)
        missile.set_init_parameters(parameters=mis_params)
        return cls(missile=missile, target=target, t_max=t_max, tau=tau, dt=dt)

    @staticmethod
    def launch(gym, aero=False, record=True, desc=True):
        """

        """
        done = False
        gym.reset()
        state = gym.get_state()
        history = [(state, done, {}, {})]
        alphas_targeting = [0]

        while not done:
            done, info = gym.step_with_guidance()
            state = gym.get_state()
            aero_result = gym.get_aero_constants() if aero else {}
            if record:
                history.append((state, done, info, aero_result))
                alphas_targeting.append(gym.missile.alpha_targeting if abs(
                    gym.missile.alpha_targeting) < gym.missile.alpha_max else copysign(gym.missile.alpha_max,
                                                                                       gym.missile.alpha_targeting))
        if desc:
            information = info['done_reason']
            print(f'info = {information}')

        if record:
            mis_vs, mis_xs, mis_ys, mis_vels, alphas, Qs = [], [], [], [], [], []
            trg_xs, trg_ys, trg_vs, trg_vels = [], [], [], []
            infos, aeros, ts = [], [], []

            for state, done, info, aero_res in history:
                gym.set_state(state)
                mis_summary = gym.missile.get_summary()
                trg_summary = gym.target.get_summary()
                mis_xs.append(mis_summary['x'])
                mis_ys.append(mis_summary['y'])
                trg_xs.append(trg_summary['x'])
                trg_ys.append(trg_summary['y'])
                ts.append(mis_summary['t'])
                mis_vs.append(mis_summary['v'])
                trg_vs.append(trg_summary['v'])
                alphas.append(mis_summary['alpha'])
                Qs.append(mis_summary['Q'])
                trg_vels.append(gym.target.vel)
                mis_vels.append(gym.missile.vel)
                aeros.append(aero_res)
                infos.append(info)

            trg_nys = [MissileGym.get_overload(v0, v1, t1 - t0)[1] for v0, v1, t1, t0 in
                       zip(trg_vels, trg_vels[1:], ts[1:], ts)]
            trg_nys += [trg_nys[-1]]
            mis_nys = [MissileGym.get_overload(v0, v1, t1 - t0)[1] for v0, v1, t1, t0 in
                       zip(mis_vels, mis_vels[1:], ts[1:], ts)]
            mis_nys += [trg_nys[-1]]

            trg_nxs = [MissileGym.get_overload(v0, v1, t1 - t0)[0] for v0, v1, t1, t0 in
                       zip(trg_vels, trg_vels[1:], ts[1:], ts)]
            trg_nxs += [trg_nxs[-1]]
            mis_nxs = [MissileGym.get_overload(v0, v1, t1 - t0)[0] for v0, v1, t1, t0 in
                       zip(mis_vels, mis_vels[1:], ts[1:], ts)]
            mis_nxs += [trg_nxs[-1]]

            dict_res = {
                't': ts,
                'missile': {'v': mis_vs, 'x': mis_xs, 'y': mis_ys, 'Q': Qs, 'alpha': alphas, 'nx': mis_nxs,
                            'ny': mis_nys, 'vel': mis_vels},
                'target': {'v': trg_vs, 'x': trg_xs, 'y': trg_ys, 'ny': trg_nys, 'nx': trg_nxs, 'vel': trg_vels},
                'alpha_targeting': alphas_targeting,
                'aero': aeros,
                'info': infos
            }
        if record:
            return dict_res
        else:
            return state, done, info, aero_res

    def reset(self):
        self.missile.reset()
        self.target.reset()

    def step(self, action):
        """
        Основной метод. Сделать шаг по времени. Изменить внутреннее состояние и вернуть необходимые данные
        argument: action {float} -- управляющее действие на данном шаге
        """
        mis_pos_0, trg_pos_0 = self.missile.pos, self.target.pos
        self.missile.step(action, self.tau)
        self.target.step(self.tau)
        mis_pos_1, trg_pos_1 = self.missile.pos, self.target.pos
        mis_vel_1, trg_vel_1 = self.missile.vel, self.target.vel
        done, info = self.get_info_about_step(mis_pos_0, trg_pos_0, mis_pos_1, trg_pos_1, mis_vel_1)
        return done, info

    def step_with_guidance(self):
        """
        Метод, моделирующий шаг step по времени tau в зависимости от метода наведения.
        Пропорциональное сближение на активном участке и на пассивном, когда скорость ракеты больше скорости цели в 2 раза
        Метод чистой погони на пассивном участке, когда скорость ракеты менее чем в 2 раза превышает скорость цели, либо меньше её
        returns: {(np.array, bool, dict)} -- состояние окружения, флаг окончания моделирования, информация (причина, время, расстояние,...)
        """
        if self.missile.P_itr(self.missile.t) > 0:
            action_guidance = self.missile.get_action_proportional_guidance(self.target)
        #             action_guidance = self.missile.get_action_alignment_guidance(self.target, tau=self.tau)
        elif (self.target.v / self.missile.v) <= 0.5:
            action_guidance = self.missile.get_action_proportional_guidance(self.target)
        else:
            action_guidance = self.missile.get_action_chaise_guidance(self.target)
        return self.step(action_guidance)

    def get_info_about_step(self, mis_pos0, trg_pos0, mis_pos1, trg_pos1, mis_vel1):
        """
        Метод, проверяющий условия остановки шага по времени метода step
        arguments: mpos0, tpos0 -- положение ракеты и цели на текущем шаге по времени tau
                   mpos1, tpos1 -- положение ракеты и цели на следующем шаге по времени tau
                   mvel1, tvel1 -- скорость ракеты и цели на следующем шаге по времени tau
        returns: {(bool, dict)} -- флаг окончания моделирования, информация (причина, время, расстояние,...)
        """
        info = {}
        if mis_pos1[1] < 0:
            info['done_reason'] = 'missile fell'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mis_pos1 - trg_pos1)
            return True, info
        if self.is_hit(mis_pos0, trg_pos0, mis_pos1, trg_pos1):
            info['done_reason'] = 'target destroyed'
            info['t'] = self.missile.t
            return True, info
        if self.is_wrong_way(mis_pos1, mis_vel1, trg_pos1) and self.missile.P_itr(self.missile.t) == 0:
            info['done_reason'] = 'wrong way'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mis_pos1 - trg_pos1)
            return True, info
        if self.missile.t > self.t_max:
            info['done_reason'] = 'a long time to fly'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mis_pos1 - trg_pos1)
            return True, info
        if self.missile.t > 20 and self.missile.v < 340:
            info['done_reason'] = 'velocity is small'
            info['t'] = self.missile.t
            info['distance_to_target'] = np.linalg.norm(mis_pos1 - trg_pos1)
            return True, info
        return False, {'done_reason': 'unknown'}

    def is_hit(self, mis_pos0, trg_pos0, mis_pos1, trg_pos1):
        r0 = np.linalg.norm(mis_pos0 - trg_pos0)
        r1 = np.linalg.norm(mis_pos1 - trg_pos1)
        if min(r1, r0) <= self.missile.r_kill:
            return True
        return False

    def is_wrong_way(self, mis_pos, mis_vel, trg_pos):
        vis_n = (trg_pos - mis_pos)
        d = np.linalg.norm(vis_n)
        if d > 500:
            return False
        vis_n /= d
        # mis_vel1 = mis_vel / np.linalg.norm(mis_vel)
        mis_axis = self.missile.x_axis
        mis_axis1 = mis_axis / np.linalg.norm(mis_axis)
        return mis_axis1 @ vis_n < 0.5

    @staticmethod
    def get_overload(vel0, vel1, tau):
        g = 9.80665
        vel0 = np.array(vel0)
        vel1 = np.array(vel1)
        a = (vel1 - vel0) / tau - np.array([0, -g])
        a_tau = np.dot(a, vel0 / np.linalg.norm(vel0)) * vel0 / np.linalg.norm(vel0)
        a_n = a - a_tau
        n_y = copysign(np.linalg.norm(a_n) / g, np.cross(vel0, a_n))
        n_x = copysign(np.linalg.norm(a_tau) / g, np.dot(a_tau, vel0))
        return np.array([n_x, n_y])

    def set_state(self, state):
        """
        Метод, задающий новое состояние (state) окружения.      
        arguments: state {np.ndarray} -- numpy-массив, в котором хранится вся необходимая информация для задания нового состояния
        returns: observation в новом состоянии
        """
        self.missile.set_state(state[:self._miss_state_len])
        self.target.set_state(state[self._miss_state_len:self._miss_state_len + self._trg_state_len])

    def get_state(self):
        mis_state = self.missile.get_state()
        trg_state = self.target.get_state()
        return np.concatenate([mis_state, trg_state])

    def get_aero_constants(self):
        """
        Метод, вычисляющий аэродинамические коэффициенты и характеристики ракеты в текущем состоянии state
        returns: {dict}
        """
        mis_state = self.missile.get_state()
        # TODO: AERO models
        pass

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