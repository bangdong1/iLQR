'''
 Lane joining / merging
 Ego starts in the lower lane (y = 0) and wants to merge into the main lane
 (y = 1.5). Another vehicle cruises in the main lane just ahead. Ego must
 adjust its speed and steering to slot in BEHIND the other vehicle, then
 settle into the main lane.
 Adjust cost and initial state to get desired behaviors.
'''

import sympy as sp
import numpy as np
from ilqr import *

def vehicle_kinematics(state, action):
    px, py, heading, vel, steer = state
    accel, steer_vel = action

    state_dot = sp.Matrix([
                    vel*sp.cos(heading),
                    vel*sp.sin(heading),
                    vel*sp.tan(steer),
                    accel,
                    steer_vel])

    return state_dot


#state and action dimensions
n_x = 10
n_u = 2

#get symbolic variables
state, action = GetSyms(n_x, n_u)

#Construct dynamics
state_dot = sp.Matrix([0.0]*n_x)
# ego vehicle kinematics
state_dot[:5, :] = vehicle_kinematics(state[:5], action)
# other vehicle kinematics (constant velocity and steering)
state_dot[5:, :] = vehicle_kinematics(state[5:], [0, 0])
#construct
dynamics = Dynamics.SymContinuous(state_dot, state, action)


#Construct cost: slow down, then merge behind the other vehicle
px1, py1, heading1, vel1, steer1 = state[:5]
px2, py2, heading2, vel2, steer2 = state[5:]
#cost for target lane (ego wants to join the other vehicle's lane y = 1.5)
L = 0.5*(py1 - 1.5)**2
#cost on velocity: ego target speed 1.3 (slower than other's 2 -> drops back)
L += (vel1 - 1.3)**2
#want to stay roughly one car-length BEHIND the other vehicle (px2 - px1 ~ 4)
L += 0.2*((px2 - px1) - 4)**2
#penality on actions
L += 0.1*action[1]**2 + 0.1*action[0]**2

#collision avoidance (do not cross ellipse around the other vehicle)
L += SoftConstrain([((px1 - px2)/5.0)**2 + ((py1 - py2)/2.0)**2 - 1])
#constrain steering angle and y-position
L += Bounded([py1, steer1], high = [2.5, 0.523], low = [-2.5, -0.523])
#construct
cost = Cost.Symbolic(L, 0, state, action)

#initialise the controller
controller = iLQR(dynamics, cost)
#prediction Horizon
N = 200
#initial state
# ego   : lower lane y=0,   x=0, heading 0, speed 2   (almost side-by-side)
# other : upper lane y=1.5, x=0, heading 0, speed 2   (cruising forward)
# -> ego slows down, drops back, then merges in behind the other vehicle
x0 = np.array([0, 0,   0, 2, 0,
               0, 1.5, 0, 2, 0])
#initil guess
us_init = np.random.randn(N, n_u)*0.0001
#get optimal states and actions
xs, us, cost_trace = controller.fit(x0, us_init, 100)

#visualize the merging scenario
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.transforms import Affine2D

def visualize(xs):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_xlim(0, 30)
    ax.set_ylim(-3.1, 3.1)
    ax.set_aspect('equal')
    ax.axis("off")

    for boundary_y in [-3, 3]:
        ax.plot([0, 30], [boundary_y, boundary_y], 'k-', linewidth=1.0)

    for lane_y in [0, 1.5]:
        ax.plot([0, 30], [lane_y, lane_y], 'k--', linewidth=1.0)

    ego_length = 2.0
    ego_width = 1.0
    other_length = 2.0
    other_width = 1.0

    ego_rect = patches.Rectangle((0, 0), ego_length, ego_width, fc='r', ec='r', alpha=0.5)
    other_rect = patches.Rectangle((0, 0), other_length, other_width, fc='g', ec='g', alpha=0.5)
    ax.add_patch(ego_rect)
    ax.add_patch(other_rect)

    ego_trajectory, = ax.plot([], [], 'r-', label='Ego vehicle trajectory')
    other_trajectory, = ax.plot([], [], 'g-', label='Other vehicle trajectory')

    def place(rect, cx, cy, angle_deg, length, width):
        # anchor bottom-left at (-length/2, -width/2), rotate about origin,
        # then translate to the center so the rect stays centered on (cx, cy)
        t = (Affine2D()
             .translate(-length / 2, -width / 2)
             .rotate_deg(angle_deg)
             .translate(cx, cy)) + ax.transData
        rect.set_xy((0, 0))
        rect.set_transform(t)

    def init():
        place(ego_rect, xs[0, 0], xs[0, 1], np.degrees(xs[0, 2]), ego_length, ego_width)
        place(other_rect, xs[0, 5], xs[0, 6], np.degrees(xs[0, 7]), other_length, other_width)
        ego_trajectory.set_data([], [])
        other_trajectory.set_data([], [])
        return ego_rect, other_rect, ego_trajectory, other_trajectory

    def update(frame):
        place(ego_rect, xs[frame, 0], xs[frame, 1], np.degrees(xs[frame, 2]), ego_length, ego_width)
        place(other_rect, xs[frame, 5], xs[frame, 6], np.degrees(xs[frame, 7]), other_length, other_width)

        ego_trajectory.set_data(xs[:frame+1, 0], xs[:frame+1, 1])
        other_trajectory.set_data(xs[:frame+1, 5], xs[:frame+1, 6])
        return ego_rect, other_rect, ego_trajectory, other_trajectory

    ani = FuncAnimation(fig, update, frames=range(len(xs)), init_func=init, blit=True, interval=50)

    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Lane Joining / Merging Visualization with Trajectories')
    plt.legend()
    plt.grid(True)
    plt.show()

visualize(xs)
