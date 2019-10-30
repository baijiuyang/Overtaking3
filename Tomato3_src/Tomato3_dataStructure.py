'''data structure'''
import numpy as np
import pandas as pd
import os
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy.signal import butter, lfilter
from scipy.interpolate import interp1d

class Trial:
    def __init__(self, subject_id, trial_id, lpos, fpos, fori, tstamps, v0, leader, leader_onset, leader_model, \
                 d0=2, Hz=90, order = 4, cutoff = 0.6):
        self.subject_id = subject_id
        self.trial_id = trial_id
        self.d0 = d0
        self.v0 = v0
        self.tstamps = tstamps
        self.lpos = lpos # unfilered time series of leader position, 2-d np array, column0:x column1:y  
        self.fpos = fpos # unfilered time series of follower position, 2-d np array, column0:x column1:y  
        self.fori = fori # unfilered time series of follower orientation , 2-d np array, column0-2:yaw pitch row  
        self.Hz = Hz
        self.theta = np.arctan(9/11); # The smaller angle of the diagonal of the walking space
        self.leader = leader
        self.leader_onset = leader_onset
        self.leader_model = leader_model
        self.order = order
        self.cutoff = cutoff
        if leader != None:
            self.f1 = next((i for i, p in enumerate(self.lpos - self.lpos[0]) if list(p) != [0, 0, 0]), None)
        else:
            self.f1 = 2
        
        
    
    def rotate_data(self, data):
        '''
            rotate the data so that the new y axis points from homepole to target door.
        '''
        # unify two directions of walking
        if self.trial_id%2 == 0:
            data = -data
        # translate origin to homepole position 
        trans_data = data - np.array([-4.5,-5.5,0])
        # rotate data
        R = np.array([[np.cos(self.theta), np.sin(self.theta)],
                      [-np.sin(self.theta), np.cos(self.theta)]])
        xy = np.matmul(trans_data[:,0:2], R)
        return np.stack((xy[:,0], xy[:,1], trans_data[:,2]), axis=1)
   
    def filter_data(self, data, order, cutoff, Hz):
        # interpolate and extrapolate (add pads on two sides to prevent boundary effects)
        pad = 3
        func = interp1d(self.tstamps, data, axis=0, kind='linear', fill_value='extrapolate')
        indices = [i*1.0/Hz for i in list(range(-pad*Hz, len(data) + pad*Hz))]
        data = func(indices)
        # low pass filter on position
        b, a = butter(order, cutoff/(Hz/2.0))
        data = filtfilt(b, a, data, axis=0, padtype=None) # no auto padding
        # remove pads
        data = data[pad*Hz:-pad*Hz]
        return data
    
    def get_positions(self, role, is_rotated=True, is_filtered=True, order=None, cutoff=None):
        if order == None:
            order = self.order
        if cutoff == None:
            cutoff = self.cutoff
        if role == 'l':
            data = self.lpos
            if is_filtered or is_rotated:
                data = self.rotate_data(self.lpos)
            if is_filtered:
                pos0 = np.tile([0, 0, 0], (self.f1 - 1, 1))
                vel = np.tile([0, self.v0/Hz, 0], (len(self.tstamps) - self.f1 + 1, 1))
                pos1 = np.cumsum(vel, axis=0) + data[self.f1]
                data = np.concatenate((pos0, pos1))
        elif role == 'f':
            data = self.fpos
            if is_rotated:
                data = self.rotate_data(data)
            if is_filtered:
                data = self.filter_data(data, order, cutoff, self.Hz)
        return data
    
    def get_velocities(self, role, is_rotated=True, is_filtered=True, order=None, cutoff=None):        
        if role == 'l':
            pos = self.get_positions(role, is_rotated, is_filtered, order, cutoff)
            pos[:self.f1] = pos[self.f1]
        else:
            pos = self.get_positions(role, is_rotated, is_filtered, order, cutoff)
        return np.gradient(pos, axis=0)*self.Hz
    
    def get_speeds(self, role, is_rotated=True, is_filtered=True, order=None, cutoff=None):
        vel = self.get_velocities(role, is_rotated, is_filtered, order, cutoff)
        return np.linalg.norm(vel[:,0:2], axis=1)
    
    def get_accelerations(self, role, is_rotated=True, is_filtered=True, order=None, cutoff=None):
        vel = self.get_velocities(role, is_rotated, is_filtered, order, cutoff)
        return np.gradient(vel, axis=0)*self.Hz

    def plot_positions(self, accelerations=False, is_rotated=True, is_filtered=True, order=None, \
                       cutoff=None, links=False):
        # get data
        fpos = self.get_positions('f', is_rotated, is_filtered, order, cutoff)
        fspd = self.get_speeds('f', is_rotated, is_filtered, order, cutoff)
        facc = self.get_accelerations('f', is_rotated, is_filtered, order, cutoff)
        lpos = self.get_positions('l', is_rotated, is_filtered, order, cutoff)
        lspd = self.get_speeds('l', is_rotated, is_filtered, order, cutoff)
        f1 = self.f1
        f2 = len(self.tstamps)
        
        # build figure
        fig = plt.figure(figsize=(5,6))
        if is_rotated:
            ax = plt.axes(xlim=(-3, 3), ylim=(-1, 15))
        else:
            ax = plt.axes(xlim=(-4.5, 4.5), ylim=(-5.5, 5.5))
            
        # set the aspect ratio equal to that of the actual value
        ax.set_aspect('auto')
        cmap = cm.get_cmap('plasma')
#         cmap = cm.get_cmap('rainbow')
    
        # plot leader and follower pos  
        ax.scatter(lpos[self.f1:,0], lpos[self.f1:,1], c = cmap((lspd[self.f1:]-0.8)/0.8), marker=',', s=[0.5]*len(lpos[self.f1:]))
        ax.scatter(fpos[:,0], fpos[:,1], c = cmap((fspd-0.8)/0.8), marker=',',s=[0.5]*len(fpos))
        
        # plot acceleration vectors as arrows
        if accelerations and is_filtered:                 
            for i in range(0, f2, 9):
                plt.arrow(fpos[i,0], fpos[i,1], facc[i,0], facc[i,1], head_width=0.03, length_includes_head=True, color='k')
        
        # plot links between follower position and leader position
        if links:
            for i in range(f1, f2, int(self.Hz/2)):
                x1, y1 = fpos[i,0], fpos[i,1]
                x2, y2 = lpos[i,0], lpos[i,1]
                plt.plot([x1,x2], [y1,y2], '--', lw=1, c='0.5')
        
        # add labels and color bar
        norm = mpl.colors.Normalize(vmin=0.8, vmax=1.6)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
        cb.set_label('m/s')
        plt.xlabel('position x')
        plt.ylabel('position y')
        filt = ', filtered data' if is_filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)
        plt.show()
    
    def plot_speeds(self, distance=True, is_rotated=True, is_filtered=True, order=None, cutoff=None):
        # get data
        fspd = self.get_speeds('f', is_rotated, is_filtered, order, cutoff)
        lspd = self.get_speeds('l', is_rotated, is_filtered, order, cutoff)
        lpos = self.get_positions('l', is_rotated, is_filtered, order, cutoff)
        fpos = self.get_positions('f', is_rotated, is_filtered, order, cutoff)
        t = self.tstamps
        
        # build figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 12), ylim=(-0.5, 2))
        
        # plot distance
        if distance:
            for i in range(self.f1+1, len(lpos)):
                x1, x2, y1, y2 = t[i], t[i], lspd[i], lspd[i] + (lpos[i,1] - fpos[i,1]) / 10
                line3 = ax.plot([x1, x2], [y1, y2], c='0.8')
                
        # plot leader and follower spd
        line1 = ax.plot(t[self.f1+1:], lspd[self.f1+1:])
        line2 = ax.plot(t, fspd)
        plt.xlabel('time')
        plt.ylabel('speed (m/s)')        
        plt.legend((line1[0], line2[0], line3[0]), (str(self.leader), 'subject', 'distance/10'))
        filt = ', filtered data' if is_filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)        
        plt.show()
    
    def play_trial(self, velocities = True, interval=11, save=False, is_rotated=True, is_filtered=True, order=None, cutoff=None):
        lpos = self.get_positions('l', is_rotated, is_filtered, order, cutoff)
        lpos[:self.f1] = [99,99,0] # make leader out of the ploting range before its onset        
        fpos = self.get_positions('f', is_rotated, is_filtered, order, cutoff)
        lspd = self.get_speeds('l', is_rotated, is_filtered, order, cutoff)
        fspd = self.get_speeds('f', is_rotated, is_filtered, order, cutoff)
        pos_x = np.stack((lpos[:,0], fpos[:,0]), axis=1)
        pos_y = np.stack((lpos[:,1], fpos[:,1]), axis=1)
        fvel = self.get_velocities('f', is_rotated, is_filtered, order, cutoff)
        
        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(4,7))
        if is_rotated:
            ax = plt.axes(xlim=(-3.5, 3.5), ylim=(-1, 15))
        else:
            ax = plt.axes(xlim=(-4.5, 4.5), ylim=(-5.5, 5.5))
        plt.xlabel('position x')
        plt.ylabel('position y')
        # set the aspect ratio equal to that of the actual value
        ax.set_aspect('equal')
        filt = ', filtered data' if is_filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)       
        # initialize animation data
        leader, = ax.plot(lpos[0,0], lpos[0,1], 'ro', ms=10)
        follower, = ax.plot(fpos[0,0], fpos[0,1], 'bo', ms=10)
        clr = 'k' if velocities else 'w'
        sign = '+' if fspd[0] >= lspd[0] else '-' 
        s = str(round(fspd[0],2)) + '(' + sign + str(round(fspd[0]-lspd[0],2)) + ')m/s'
        spd = ax.text(fpos[0,0] + 0.5, fpos[0,1] - 0.5, s)
        # slow animation function redraw everything at each frame. Good for saving video
        def animate_slow(i):
            # ms is the short for markersize
            # figure labels and size
            ax.clear()
            if is_rotated:
                ax.set_xlim(-3.5, 3.5)
                ax.set_ylim(-1, 15)
            else:
                ax.set_xlim(-4.5, 4.5)
                ax.set_ylim(-5.5, 5.5)
            ax.set_xlabel('position x')
            ax.set_ylabel('position y')
            
            # set the aspect ratio equal to that of the actual value
            ax.set_aspect('equal')
            # set title
            filt = ', filtered data' if is_filtered else ', raw data'
            ax.set_title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)       
            
            # update data
            leader, = ax.plot(lpos[i,0], lpos[i,1], 'ro', ms=10)
            follower, = ax.plot(fpos[i,0], fpos[i,1], 'bo', ms=10)
            sign = '+' if fspd[i] >= lspd[i] else '-' 
            s = str(round(fspd[i],2)) + '(' + sign + str(round(fspd[i]-lspd[i],2)) + ')m/s'
            spd = ax.text(fpos[i,0] - 1, fpos[i,1] - 0.7, s)
            arr = ax.arrow(fpos[i,0], fpos[i,1], fvel[i,0], fvel[i,1], head_width=0.1, length_includes_head=True, color=clr)
            return leader, follower, arr

        # slow animation function redraw everything at each frame. Good for saving video
        def animate_fast(i):
            # ms is the short for markersize
            leader.set_data(lpos[i,0], lpos[i,1])
            follower.set_data(fpos[i,0], fpos[i,1])
            sign = '+' if fspd[i] >= lspd[i] else '-' 
            s = str(round(fspd[i],2)) + '(' + sign + str(round(fspd[i]-lspd[i],2)) + ')m/s'
            spd.set_text(s)
            spd.set_position((fpos[i,0] - 1, fpos[i,1] - 0.7))            
            arr = ax.arrow(fpos[i,0], fpos[i,1], fvel[i,0], fvel[i,1], head_width=0.1, length_includes_head=True, color=clr)
            return leader, follower, spd, arr
        # call the animator.  blit=True means only re-draw the parts that have changed.
        animate = animate_slow if save else animate_fast
        anim = animation.FuncAnimation(fig, animate, frames=len(pos_x), interval=interval, blit=True)

        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        if save:
            filename = 'Subj' + str(self.subject_id) + 'Trial' + str(self.trial_id) + '.mp4'
            anim.save(filename, fps=None)
        return anim
        # plt.show()
    
# create subject class
class Subject:
    def __init__(self, id, gender=None, IPD=None, leader=None, trials=None, freewalk=None):
        self.id = id
        self.gender = gender
        self.IPD = IPD
        self.leader = leader
        self.trials = trials if trials is not None else {}
        self.freewalk = freewalk if freewalk is not None else {}
        
# create Experiment class
class Experiment:
    def __init__(self, n=None, subjects=None):
        self.n = n
        self.subjects = subjects if subjects is not None else {}
    
    def plot_positions(self):
        pass
