'''
 Double integrator: push a box to a goal position (s = 10 m).
 Demonstrates how a Bounded() barrier constrains the applied force.
 Runs the problem twice (without / with a force constraint) and saves the
 resulting displacement & force plots.
'''
import numpy as np
import sympy as sp

from ilqr import iLQR
from ilqr.utils import GetSyms, Bounded
from ilqr.containers import Cost, Dynamics

#state and action dimensions
n_x = 2
n_u = 1

#params
m  = 2.0   #mass
dt = 0.1   #discrete time step

#numerical dynamics: s'' = u / m
def f(x, u):
    vel = x[1]
    acc = u[0]/m
    return np.array([vel, acc])

dynamics = Dynamics.Continuous(f, dt)

#symbolic variables for the cost
x, u = GetSyms(n_x, n_u)
s, v = x

#goal: s = 10, v = 0
x_goal = np.array([10, 0])
Q  = np.diag([1, 0.1])
R  = np.diag([0.1])
QT = np.diag([10, 10])

#initial state and control guess
x0 = np.array([0, 0])
N  = 50
us_init = np.random.randn(N, n_u)*0.01


def solve(constrained):
    if constrained:
        #constrain applied force to [-2, 2] N
        cons = Bounded(u, high=[2], low=[-2])
        cost = Cost.QR(Q, R, QT, x_goal, cons)
    else:
        cost = Cost.QR(Q, R, QT, x_goal)
    controller = iLQR(dynamics, cost)
    xs, us, _ = controller.fit(x0, us_init)
    return xs, us


import matplotlib.pyplot as plt

def plot(xs, us, title, outfile):
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    t = np.arange(len(xs)) * dt
    ax[0].plot(t, xs[:, 0], 'b-')
    ax[0].axhline(10, color='k', ls='--', lw=0.8)
    ax[0].set_ylabel('Displacement (m)')
    ax[0].set_title(title)
    ax[0].grid(True)

    ax[1].plot(t[:-1], us[:, 0], 'r-')
    ax[1].axhline(2, color='k', ls='--', lw=0.8)
    ax[1].axhline(-2, color='k', ls='--', lw=0.8)
    ax[1].set_ylabel('Force (N)')
    ax[1].set_xlabel('Time (s)')
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    print('saved', outfile)
    return fig


xs_u, us_u = solve(constrained=False)
xs_c, us_c = solve(constrained=True)

plot(xs_u, us_u, 'Without force constraint', 'imgs/withoutconstraint.png')
plot(xs_c, us_c, 'With force constraint (|u| <= 2N)', 'imgs/withconstraint.png')

plt.show()
