import numpy as np
from interpolation import InterpVec
from scenarios import scenarios

class Target2D(object):
       
    @classmethod
    def get_simple_target(cls, pos, vel):
        velocity_vectors = [[0, np.array(vel)]]
        vel_interp = InterpVec(velocity_vectors)
        parameters = np.array([pos[0], pos[1], 0])
        target = cls(vel_interp=vel_interp,
                     state_init=parameters)
        target.set_init_cond(parameters=parameters)
        return target

    @classmethod
    def get_target(cls, scenario_name='SUCCESS', scenario_i=0):
        velocity_vectors = scenarios[scenario_name][scenario_i]['trg_vels']
        x, y = scenarios[scenario_name][scenario_i]['trg_pos_0']

        vel_interp = InterpVec(velocity_vectors)
        target = cls(vel_interp=vel_interp)
        target.set_init_cond(parameters_of_target=np.array([x, y, 0]))
        return target

    def __init__(self, **kwargs):
        self.g = kwargs.get('g', 9.80665)
        self.state = None
        self.state_init = kwargs['state_init']
        self.vel_interp = kwargs['vel_interp']

    def set_init_cond(self, parameters=None):
        if parameters is None:
            parameters = self.get_init_parameters()
        self.state = np.array(parameters)
        self.state_init = np.array(parameters)

    def get_init_parameters(self):
        return self.state_init

    def reset(self):
        self.set_state(self.state_init)

    def set_state(self, state):
        self.state = np.array(state)

    def get_state(self):
        return self.state
    
    def get_state_init(self):
        return self.state_init

    def step(self, tau=0.1, dtn=0.01):
        x, y, t = self.state
        t_end = t + tau
        flag = True
        while flag:
            if t_end - t > dtn:
                dt = dtn
            else:
                dt = t_end - t
                flag = False
            t += dt
            vx, vy = self.vel_interp(t)
            x += vx * dt 
            y += vy * dt
        self.set_state([x, y, t])

    @property
    def pos(self):
        return self.state[:2]
    
    @property
    def vel(self):
        return self.vel_interp(self.t)

    @property
    def t(self):
        return self.state[-1]
    
    @property
    def Q(self):
        vx, vy = self.vel_interp(self.t)
        return np.sqrt(vx ** 2 + vy ** 2)

    @property
    def v(self):
        vx, vy = self.vel_interp(self.t)
        return np.sqrt(vx ** 2 + vy ** 2)

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    def get_summary(self):
        return { 
            't': self.t,
            'v': self.v,
            'x': self.x,
            'y': self.y,
            'Q': np.degrees(self.Q)
        }