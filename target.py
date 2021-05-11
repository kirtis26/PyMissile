import numpy as np
from math import *
from interpolation import InterpVec

class Target(object):
       
    @classmethod
    def get_simple_target(cls, pos, vel):
        velocity_vectors = [[0, np.array(vel)]]
        vel_interp = InterpVec(velocity_vectors)
        target = cls(vel_interp=vel_interp)
        parameters_of_target = np.array([pos[0], pos[1], 0])
        target.set_init_cond(parameters_of_target=parameters_of_target)
        return target

    def __init__(self, *args, **kwargs):
        self.g   = kwargs.get('g', 9.80665)
        self.dt  = kwargs.get('dt', 0.001)
        self.vel_interp = kwargs['vel_interp']

    def set_init_cond(self, parameters_of_target=None):
        if parameters_of_target is None:
            parameters_of_target = self.get_standart_parameters_of_target()
        self.state = np.array(parameters_of_target)
        self.state_0 = np.array(parameters_of_target)

    def reset(self):
        self.set_state(self.state_0)

    def set_state(self, state):
        self.state = np.array(state)

    def get_state(self):
        return self.state
    
    def get_state_0(self):
        return self.state_0

    def step(self, tau):
        x, y, t = self.state
        t_end = t + tau
        flag = True
        while flag:
            if t_end - t > self.dt:
                dt = self.dt 
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