"""
Nikhef topical lectures April 2021
Andreas Freise, Bas Swinkels 13.04.2021
"""

import numpy as np
import time
import random
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.transforms as trafo
from matplotlib.patches import Polygon
import warnings


class Drone:
    def __init__(self, test=False, name=None, wind=False, flight_range=[-500, -500, 500, 500]):
        """
        creates drone object that provides shape for plotting
        test = True: switches to a 'perfect' drone, only use for initial testing
        wind = True: adds noisy air movement
        name: Name of the student, this is used to set 'random' drone parameters
        flight_range = [lower_left_y, lower_left_z, upper_right_y, upper_right_z]:
                        defines the allowed movement of the drone in y-z space
        """
        self.test = test
        self.wind = wind
        self.name = name

        self.deltaT = 1/60.0 # size of a time step in seconds

        # init forces and state
        self.L_arm = 20 # cm
        self.M = 0.1
        self.Ixx = 10  
        self.g = 10

        self.F = 0.0
        self.tau = 0.0
        self.F_left = 0.0
        self.F_right = 0.0
        self.V_left_amp = 1.0
        self.V_right_amp = 1.0
        self.V_left_offset = 0.0
        self.V_right_offset = 0.0
        self.V_left_clip_m = -5.0
        self.V_left_clip_p =  5.0
        self.V_right_clip_m = -5.0
        self.V_right_clip_p =  5.0
        self.range = flight_range
        self.perlin = Perlin()

        if self.test:
            warnings.warn("Running drone in test mode. Don't use this for system identification", stacklevel=2)
        else:
            if len(self.name) <1:
                warnings.warn("You need to provide your full name in the call to `drone()' for the project",stacklevel=2)
            else:
                key = name_to_int(self.name)
                # print("{0:b}".format(key))
                # print(limited_hash(key, 0,3))
                # print(limited_hash(key, 4,7))
                # print(limited_hash(key, 8,11))
                # print(limited_hash(key, 12,15))
                # print(bits_to_float(key,0,3,0.05))
                # self.V_left_amp += bits_to_float(key,0,3,0.01)
                # self.V_right_amp += bits_to_float(key,4,7,0.01)
                self.V_left_offset += bits_to_float(key,8,11,0.01)
                self.V_right_offset += bits_to_float(key,12,15,0.01)
                self.V_left_clip_m += bits_to_float(key,16,19,1)
                self.V_left_clip_p += bits_to_float(key,20,23,1)
                self.V_right_clip_m += bits_to_float(key,24,27,1)
                self.V_right_clip_p += bits_to_float(key,28,31,1)

        self.reset()

    def reset(self):
        self.stop = False
        self.t = 0

        self.F_wind_z = 0
        self.F_wind_y = 0
        self.perlin_offset1 = random.random()*1000.0
        self.perlin_offset2 = random.random()*1000.0

        # state variable for odeint, stores y,z,phi for t=0 and t=deltaT 
        self.solve_state = np.zeros(6)

        # current position and velocities
        self.pos = self.solve_state[0:3] 
        self.vel = np.zeros(3)
        # current forces 
        self.forces = np.zeros(4) 

        # variables for targets
        self.targets = None
        self.target_idx = 0
        self.num_targets = 0


    def set_targets(self, _targets):
        """
        Function to set a list of targets for the drone, the list needs to be
        a Numpy array with N rows of 3 values (x,y, radius) and N the number of targets
        """

        self.targets = _targets
        assert(isinstance(self.targets,np.ndarray)), "The targets must be a Numpy array of shape (n,3)"
        assert(self.targets.shape[1]==3), "The targets must be a Numpy array of shape (n,3)"
        self.target_idx = 0
        self.num_targets = self.targets.shape[0]

    def dx(self, t, x, F, tau):
        """
        state update function for use with Scipy's odeint(dx, ..., tfirst=True).
        input:
            t = time step vector
            x = [y z phi y' z' phi'], positions and velocities
            F = upward force
            tau = angular momentum
        returns [y' z' phi' y'' z'' phi'']
        
        The function uses self.F_wind_y/z to add random forces
        TODO: we could add drag here

        see 
        https://www.youtube.com/watch?v=lAVYDUeqdW4&t=187s 
        https://www.youtube.com/watch?v=dWwhLP0Iwvg
        """

        F_over_M = F / self.M

        # rough estimate of angular momentum from wind
        tau_wind = 0.01 * self.F_wind_y * self.F_wind_z

        return [x[3],x[4],x[5],
            -np.sin(x[2]) * F_over_M + self.F_wind_y,  # -drag 
            np.cos(x[2]) * F_over_M - self.g + self.F_wind_z,  # -drag 
            tau / self.Ixx  + tau_wind # - drag 
        ]

    def set_V(self, _V_left, _V_right):
        # clipping voltages
        V_left =  np.clip(_V_left, self.V_left_clip_m, self.V_left_clip_p)
        V_right = np.clip(_V_right, self.V_right_clip_m, self.V_right_clip_p)

        self.F_left = V_left * self.V_left_amp + self.V_left_offset
        self.F_right = V_right * self.V_right_amp + self.V_right_offset

        self.F = self.F_right + self.F_left
        self.tau = (self.F_right - self.F_left) * self.L_arm

    def set_state(self):
        # clipping at the boundary of the flight_range
        back1 = 0.999
        if self.solve_state[0] < self.range[0]:
            self.solve_state[0] = back1 * self.range[0]
            self.solve_state[3] = 0.0
        if self.solve_state[0] > self.range[2]:
            self.solve_state[0] = back1 * self.range[2]
            self.solve_state[3] = 0.0
        if self.solve_state[1] < self.range[1]:
            self.solve_state[1] = back1 * self.range[1]
            self.solve_state[4] = 0.0
        if self.solve_state[1] > self.range[3]:
            self.solve_state[1] = back1 * self.range[3]
            self.solve_state[4] = 0.0

        # keeping phi to be between -pi and +pi
        self.solve_state[2] = (self.solve_state[2] + np.pi) % (2 * np.pi) - np.pi

        self.pos = self.solve_state[0:3]
        self.vel = self.solve_state[3:6]
        self.forces = np.array([self.F,self.tau, self.F_left, self.F_right])

        # check targets
        if self.targets is not None and self.target_idx<self.num_targets:
            r = self.targets[self.target_idx,2]
            if np.abs(self.pos[0]-self.targets[self.target_idx,0])< r and np.abs(self.pos[1]-self.targets[self.target_idx,1])< r:
                self.target_idx += 1

    def update(self):
        self.t += self.deltaT
        if self.wind:
            self.F_wind_z = 0.1 * self.perlin.Sum(self.t+self.perlin_offset1, 0.2, 1, 1.05, 1.3) # TODO check octave 1/2
            self.F_wind_y = 2   * self.perlin.Sum(self.t+self.perlin_offset2, 0.2, 1, 1.05, 1.3)

        self.solve_state = odeint(self.dx, self.solve_state, [0, self.deltaT], tfirst=True, args=(self.F, self.tau), atol=1e-6, rtol=1e-6)[1]
        self.set_state()
        return np.concatenate(([self.t], self.pos, self.vel, self.forces, [self.target_idx]))

    def shape(self):
        """ creates drone image as x-y-polygon, returns Polygon"""
        # right part of drone
        right_part = np.array([
            [3, 3],[3, 1],[19, 1],[19, 3],[10, 2],[10, 5],[19, 4],[21, 4],
            [30, 5],[30, 2],[21, 3],[21, 1],[22, 1],[22, -3],[18, -3],
            [18, -1],[3, -1],[3, -3],])
        # add mirror image
        left_part = np.c_[-right_part[::-1,0], right_part[::-1,1]]
        drone_shape = np.concatenate((left_part, right_part))
        return Polygon(drone_shape, closed=True, zorder=2, facecolor="gray")
    
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class Plotter:
    def __init__(self, _drone):

        # shapes for plotting
        self.drone = _drone
        self.drone1 = self.drone.shape()
        self.drone2 = self.drone.shape()

        self.num_targets = 0

        # init figure and axes
        self.fig = plt.figure(figsize=(4.5,8))
        widths = [1, 2]
        heights = [1, 1]
        self.gs1 = self.fig.add_gridspec(nrows=2, ncols=1,height_ratios=heights)
                          
        # full field drone view                  
        self.ax1 = self.fig.add_subplot(self.gs1[0,0])
        # zoomed-in drone view
        self.ax2 = self.fig.add_subplot(self.gs1[1,0])

        self.text0 = self.ax1.text(20,-420, "") # user string
        self.text1 = self.ax1.text(-420,-420, "") # fps
        self.text2 = self.ax2.text(-40,-45, "") # F, tau
        self.text3 = self.ax2.text(-40,-38, "") # force left right
        self.text4 = self.ax2.text(-40,-31, "") # position 
        self.text5 = self.ax2.text(-40,-24, "") # velocity

        self.ax1.axis([-500, 500, -500, 500])
        self.ax2.axis([-50, 50, -50, 50])
        
        self.ax1.set_aspect('equal')
        self.ax2.set_aspect('equal')

        self.gs1.tight_layout(self.fig)
        self.fig.canvas.draw()

        # store background
        self.ax1_bg = self.fig.canvas.copy_from_bbox(self.ax1.bbox)
        self.ax2_bg = self.fig.canvas.copy_from_bbox(self.ax2.bbox)

        self.reset()

        plt.show(block=False)

    def reset_buffer(self):
        self.bufflen = 500
        self.buffidx = 0
        self.t_b = np.zeros(self.bufflen) 
        self.y_b = np.empty(self.bufflen) * np.nan
        self.z_b = np.empty(self.bufflen) * np.nan
        self.F_b = np.empty(self.bufflen) * np.nan
        self.tau_b = np.empty(self.bufflen) * np.nan

        self.line_z, = self.ax3.plot(self.t_b,self.z_b, 'k', label = "z")
        self.line_y, = self.ax3.plot(self.t_b,self.y_b, 'r', label = "y")
        self.line_F, = self.ax4.plot(self.t_b,self.F_b, 'k', label = "F")
        self.line_tau, = self.ax4.plot(self.t_b,self.tau_b, 'r', label = "tau")

    def draw_traces(self):
        # draw traces
        if (self.t-self.t_b[-1] >= 0.1):
            self.t_b = np.roll(self.t_b,-1)
            self.z_b = np.roll(self.z_b,-1)
            self.y_b = np.roll(self.y_b,-1)
            self.F_b = np.roll(self.F_b,-1)
            self.tau_b = np.roll(self.tau_b,-1)
            self.t_b[-1]=self.t
            self.z_b[-1]=self.z
            self.y_b[-1]=self.y
            self.F_b[-1]=self.F
            self.tau_b[-1]=self.tau
            self.line_z.set_xdata(self.t_b-self.t_b[0])
            self.line_z.set_ydata(self.z_b)
            self.line_y.set_xdata(self.t_b-self.t_b[0])
            self.line_y.set_ydata(self.y_b)
            self.line_F.set_xdata(self.t_b-self.t_b[0])
            self.line_F.set_ydata(self.F_b)
            self.line_tau.set_xdata(self.t_b-self.t_b[0])
            self.line_tau.set_ydata(self.tau_b)
            #self.ax3.set_xlim(np.min(self.t_b), np.max(self.t_b))

    def reset(self):
        self.stop = True
        self.str1 = ""
        self.str2 = ""
        self.str3 = ""
        self.str4 = ""
        self.str5 = ""
        self.t=0
        self.frames = 0
        self.last_frame = time.time()

        [p.remove() for p in reversed(self.ax1.patches)]

        self.target_idx = 0
        if self.drone.targets is not None:
            self.num_targets = self.drone.num_targets
            self.targets = [None] * self.num_targets
            self.target_patches = [None] * self.num_targets
            #self.patch3 = self.ax1.add_patch(self.target)

            for i in range(self.num_targets):
                t = self.drone.targets[i,:]
                self.targets[i] = plt.Circle((t[0],t[1]),2*t[2], lw=1, color='red', alpha=0.5, fill=False)
                self.target_patches[i] = self.ax1.add_patch(self.targets[i])

        self.patch1 = self.ax1.add_patch(self.drone1)
        self.patch2 = self.ax2.add_patch(self.drone2)

    def close(self):
        plt.close(self.fig)

    def update_display(self, user_str=""):
        [y, z, phi] = self.drone.pos
        [dy, dz, dphi] = self.drone.vel
        z = self.drone.solve_state[1]
        dz = self.drone.solve_state[4]
        [F, tau, F_left, F_right] = self.drone.forces
        self.frames += 1

        self.str2 = f"F = {F:4.2f}, tau = {tau:4.2f}"
        self.str3 = f"F_left = {F_left:4.2f}, F_right = {F_right:4.2f}"
        self.str4 = f"y = {y:4.1f}, z = {z:4.1f}. phi =  {np.rad2deg(phi):3.1f}"
        self.str5 = f"y' = {dy:4.1f}, z' = {dz:4.1f}. phi' =  {np.rad2deg(dphi):3.1f}"

        # draw drones:
        # see https://stackoverflow.com/a/4891658
        tr1 = trafo.Affine2D().rotate(phi).translate(y, z)
        tr1 = tr1 + self.ax1.transData

        tr2 = trafo.Affine2D().rotate(phi)
        tr2 = tr2 + self.ax2.transData

        self.fig.canvas.restore_region(self.ax1_bg)
        self.fig.canvas.restore_region(self.ax2_bg)
        #self.fig.canvas.restore_region(self.ax3_bg)
        #self.fig.canvas.restore_region(self.ax4_bg)

        # based on https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
        
        self.fps_printer()

        self.text0.set_text(user_str)
        self.text1.set_text(self.str1)
        self.text2.set_text(self.str2)
        self.text3.set_text(self.str3)
        self.text4.set_text(self.str4)
        self.text5.set_text(self.str5)
        self.drone1.set_transform(tr1)
        self.drone2.set_transform(tr2)

        # TODO check when adding ax1/2.patch is needed
        self.ax1.draw_artist(self.ax1.patch)
        self.ax1.draw_artist(self.text0)
        self.ax1.draw_artist(self.text1)
        self.ax1.draw_artist(self.drone1)
        self.ax2.draw_artist(self.ax2.patch)
        self.ax2.draw_artist(self.text2)
        self.ax2.draw_artist(self.text3)
        self.ax2.draw_artist(self.text4)
        self.ax2.draw_artist(self.text5)
        self.ax2.draw_artist(self.drone2)
        #self.ax3.draw_artist(self.line_z)
        #self.ax3.draw_artist(self.line_y)
        #self.ax4.draw_artist(self.line_F)
        #self.ax4.draw_artist(self.line_tau)

        # check if target has been reached
        if self.drone.target_idx > self.target_idx:
            t = self.drone.targets[self.target_idx,:]
            self.target_patches[self.target_idx].fill = True
            self.target_idx += 1
        # draw targets
        for i in range(self.num_targets):
            self.ax1.draw_artist(self.target_patches[i])

        self.fig.canvas.blit(self.ax1.bbox)
        self.fig.canvas.blit(self.ax2.bbox)
        #self.fig.canvas.blit(self.ax3.bbox)
        #self.fig.canvas.blit(self.ax4.bbox)

        self.fig.canvas.flush_events()
        #self.fig.canvas.draw()
             
    def fps_printer(self):
        # print fps every 10s frame
        if not self.frames%10:
            t = time.time()
            self.str1 = f"fps: {10/(t-self.last_frame+1e-6):3.0f}"
            self.last_frame=t

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
class Perlin(object):
    def __init__(self):

        self.hash = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,  7, 225,
                     140, 36, 103, 30, 69, 142,  8, 99, 37, 240, 21, 10, 23, 190,  6, 148,
                     247, 120, 234, 75,  0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
                     57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
                     74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
                     60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
                     65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
                     200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64,
                     52, 217, 226, 250, 124, 123,  5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
                     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
                     119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172,  9,
                     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
                     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
                     81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
                     184, 84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93,
                     222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
                     151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233,  7, 225,
                     140, 36, 103, 30, 69, 142,  8, 99, 37, 240, 21, 10, 23, 190,  6, 148,
                     247, 120, 234, 75,  0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
                     57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
                     74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
                     60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
                     65, 25, 63, 161,  1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
                     200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186,  3, 64,
                     52, 217, 226, 250, 124, 123,  5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
                     207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
                     119, 248, 152,  2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172,  9,
                     129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
                     218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
                     81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
                     184, 84, 204, 176, 115, 121, 50, 45, 127,  4, 150, 254, 138, 236, 205, 93,
                     222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180]
        self.hashMask = 255
        self.gradients1D = [1.0, -1.0]
        self.gradientsMask1D = 1

    def lerp(self, A, B, factor):
        """interpolates A and B by factor"""
        return (1.0 - factor) * A + factor * B

    def Smooth(self, t):
        # t: float
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def Perlin1D(self, point, frequency):
        # point = float, frequcncy = float
        # int: i0, i1
        # float t0, t1, g0, g1, v0, v1, t
        my_int = np.vectorize(np.int)
        my_float = np.vectorize(np.float)
        point *= frequency
        i0 = my_int(np.floor(point))
        t0 = my_float(point - i0)
        t1 = my_float(t0 - 1.0)
        i0 &= self.hashMask
        i1 = i0 + 1

        g0 = self.gradients1D[self.hash[i0] & self.gradientsMask1D]
        g1 = self.gradients1D[self.hash[i1] & self.gradientsMask1D]

        v0 = g0 * t0
        v1 = g1 * t1

        t = self.Smooth(t0)
        return self.lerp(v0, v1, t) * 2.0

    # float: point, frquency, lacunricy, persistamce; int: octaves
    def Sum(self, point, frequency, octaves, lacunarity, persistence):
        sum = self.Perlin1D(point, frequency)
        amplitude = 1.0
        range = 1.0
        for o in np.arange(octaves):
            frequency *= lacunarity
            amplitude *= persistence
            range += amplitude
            sum += self.Perlin1D(point, frequency) * amplitude
        return sum / range

def hash32(ID):
    """Returns a unique int to int mapping with a pseudorandom
    distribution, see http://burtleburtle.net/bob/hash/integer.html
    """
    warnings.filterwarnings('ignore')
    a = np.uint32(ID)
    a -= (a << np.uint32(6))
    a ^= (a >> np.uint32(17))
    a -= (a << np.uint32(9))
    a ^= (a << np.uint32(4))
    a -= (a << np.uint32(3))
    a ^= (a << np.uint32(10))
    a ^= (a >> np.uint32(15))
    warnings.filterwarnings('default')
    return a

def limited_hash(key, startbit, endbit):
    """Takes an input number (key), returns the number resulting from
    taking the bits from startbit -> endbit, numbered like an array.

    eg.
    limited_hash(93, 2, 5) -> 7
    """
    newkey = key >> startbit
    newkey = newkey & (np.power(2, endbit - startbit + 1) - 1)
    return newkey

def name_to_int(s):
    s2=0
    idx = 0
    for c in s[0:13]:
        if c == ' ':
            s2 +=  5**idx
        else:
            s2 += abs(ord(c)-95)*5**idx
        idx += 1
    while s2>2**32:
        s2 = s2/1.5
    return hash32(int(s2))

def bits_to_float(key, bit1, bit2, factor):
    bits = limited_hash(key, bit1, bit2)
    fl = (bits - 7.5)/7.5 * factor
    return fl

