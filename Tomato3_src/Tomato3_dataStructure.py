'''data structure'''
import numpy as np
import matplotlib as mpl
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.signal import butter, filtfilt
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
        self.tstamps_smooth = np.linspace(0, self.tstamps[-1], num=len(self.tstamps))
        self.theta = np.arctan(9/11); # The smaller angle of the diagonal of the walking space
        self.leader = leader
        self.leader_onset = leader_onset
        self.leader_model = leader_model
        self.order = order
        self.cutoff = cutoff
        # find f1, the index when the leader appears
        if leader != None:
            # find the index of the first non zero value
            self.f1 = (self.lpos - self.lpos[0] != [0,0,0]).argmax()//3
        else:
            self.f1 = 1
        
        
    
    def rotate_data(self, data):
        '''
            Rotate the data so that the new y axis points from homepole 
            to target door.
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
   
    def filter_data(self, data, order, cutoff):
        '''
            Filter the data using butterwirth low pass digital foward
            and backward filter.
        '''
        # interpolate and extrapolate (add pads on two sides to prevent boundary effects)
        pad = 3
        func = interp1d(self.tstamps, data, axis=0, kind='linear', fill_value='extrapolate')
        indices = [i*1.0/self.Hz for i in list(range(-pad*self.Hz, len(data) + pad*self.Hz))]
        data = func(indices)
        # low pass filter on position
        b, a = butter(order, cutoff/(self.Hz/2.0))
        data = filtfilt(b, a, data, axis=0, padtype=None) # no auto padding
        # remove pads
        data = data[pad*self.Hz:-pad*self.Hz]
        return data
    
    def get_time(self, filtered):
        return self.tstamps_smooth if filtered else self.tstamps
    
    def get_positions(self, role, **kwargs):
        # load kwargs
        order = self.order if 'order' not in kwargs else kwargs['order']
        cutoff = self.cutoff if 'cutoff' not in kwargs else kwargs['cutoff']
        rotated = True if 'rotated' not in kwargs else kwargs['rotated']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        if role == 'l':
            data = self.lpos
            if filtered or rotated:
                data = self.rotate_data(self.lpos)
            if filtered:
                pos0 = np.tile([0, 0, 0], (self.f1, 1))
                vel = np.tile([0, self.v0/self.Hz, 0], (len(self.tstamps) - self.f1, 1))
                pos1 = np.cumsum(vel, axis=0) + data[self.f1]
                data = np.concatenate((pos0, pos1))
        elif role == 'f':
            data = self.fpos
            if rotated:
                data = self.rotate_data(data)
            if filtered:
                data = self.filter_data(data, order, cutoff)
        return data
    
    def get_velocities(self, role, **kwargs):
        pos = self.get_positions(role, **kwargs)
        if role == 'l':
            pos[:self.f1] = pos[self.f1]
        return np.gradient(pos, axis=0)*self.Hz
    
    def get_speeds(self, role, **kwargs):
        vel = self.get_velocities(role, **kwargs)
        return np.linalg.norm(vel[:,0:2], axis=1)
    
    def get_accelerations(self, role, **kwargs):
        vel = self.get_velocities(role, **kwargs)
        return np.gradient(vel, axis=0)*self.Hz

    def plot_trajectory(self, frames=None, accelerations=False, links=False, **kwargs):
        '''
            Show the trajectories of follower and leader using scatter plot.
            args:
                frames (array of int): List of indices to be plotted.
                accelerations (boolean): Whether draw acceleration vectors.
                links (boolean): Whether draw links between the positions of 
                       follower and leader at the same moment for a sense 
                       of concurrency.
        ''' 
        # load kwargs
        rotated = True if 'rotated' not in kwargs else kwargs['rotated']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        # get data
        fpos = self.get_positions('f', **kwargs)
        fspd = self.get_speeds('f', **kwargs)
        facc = self.get_accelerations('f', **kwargs)
        lpos = self.get_positions('l', **kwargs)
        lspd = self.get_speeds('l', **kwargs)
        f1 = self.f1
        f2 = len(self.tstamps)
        if not frames: frames = list(range(f2))
        
        # build figure
        fig = plt.figure(figsize=(5,6))
        if rotated:
            ax = plt.axes(xlim=(-3, 3), ylim=(-1, 15))
        else:
            ax = plt.axes(xlim=(-4.5, 4.5), ylim=(-5.5, 5.5))
        plt.xlabel('position x')
        plt.ylabel('position y')
        filt = ', filtered data' if filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)
        
        # set the aspect ratio equal to that of the actual value
        ax.set_aspect('auto')
        cmap = cm.get_cmap('plasma')
#         cmap = cm.get_cmap('rainbow')
        # add labels and color bar
        norm = mpl.colors.Normalize(vmin=0.8, vmax=1.6)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
        cb.set_label('m/s')
    
        # plot leader and follower pos  
        ax.scatter(lpos[self.f1:,0], lpos[self.f1:,1], c=cmap((lspd[self.f1:] - 0.8) / 0.8), \
                    marker=',', s=[0.5]*len(lpos[self.f1:]))
        ax.scatter(fpos[frames,0], fpos[frames,1], c=cmap((fspd[frames] - 0.8) / 0.8), \
                    marker=',', s=[0.5] * len(frames))
        
        # plot acceleration vectors as arrows
        if accelerations and filtered:                 
            for i in range(frames[0], frames[-1], 9):
                plt.arrow(fpos[i,0], fpos[i,1], facc[i,0], facc[i,1], head_width=0.03, length_includes_head=True, color='k')
        
        # plot links between follower position and leader position
        if links:
            for i in range(frames[0], frames[-1], int(self.Hz/2)):
                if i >= f1:
                    x1, y1 = fpos[i,0], fpos[i,1]
                    x2, y2 = lpos[i,0], lpos[i,1]
                    plt.plot([x1,x2], [y1,y2], '--', lw=1, c='0.5')
        plt.tight_layout()
        plt.show()
    
    def plot_positions(self, component='x', frames=None, **kwargs):
        '''
            Plot positions of follower and leader by time.
            args:
                component (str): 'x' lateral position, 'y' forward position,
                                default is 'x'.
                frames (array of int): List of indices to be plotted.
        '''
        # load kwargs
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        # get data
        if component == 'x':
            fpos = self.get_positions('f')[:, 0]
            lpos = self.get_positions('l')[:, 0]
            yrange = (-2, 2)
        elif component == 'y':
            fpos = self.get_positions('f')[:, 1]
            lpos = self.get_positions('l')[:, 1]
            yrange = (-1, 15)
        t = self.get_time(filtered)
        if not frames: frames = list(range(len(t)))
        
        # build figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 12), ylim=yrange)
        plt.xlabel('time')
        plt.ylabel(component + ' position (m)')
        filt = ', filtered data' if filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)
        
        # plot data
        lines, labels = [], []
        if component == 'y':
            # plot leader pos
            line1 = ax.plot(t[self.f1 + 1:], lpos[self.f1 + 1:])
            lines.append(line1[0])
            labels.append(str(self.leader))
        # plot follower pos
        line2 = ax.plot(t[frames], fpos[frames])
        lines.append(line2[0])
        labels.append('follower')
        
        # add legend
        ax.legend(lines, labels)        
        plt.tight_layout()
        plt.show()
    
    def plot_speeds(self, component='', frames=None, distance=True, **kwargs):
        '''
            Plot speeds of follower and leader by time
            args:
                component (str): 'x' lateral speed, 'y' forward speed,
                                default is total speed.
                frames (array of int): List of indices to be plotted.
                distance (boolean): Whether draw distance indicator
                          (distance/10) on top of leader speed.
        '''
        # load kwargs
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        # get data
        fspd = self.get_speeds('f', **kwargs)
        yrange = (-0.5, 2)
        if component == 'x':
            fspd = self.get_velocities('f', **kwargs)[:, 0]
            yrange = (-1, 1)
        elif component == 'y':
            fspd = self.get_velocities('f', **kwargs)[:, 1]
            yrange = (-0.5, 2)
        lspd = self.get_speeds('l', **kwargs)
        lpos = self.get_positions('l', **kwargs)
        fpos = self.get_positions('f', **kwargs)
        t = self.get_time(filtered)
        if not frames: frames = list(range(len(t)))
     
        # build figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 12), ylim=yrange)
        plt.xlabel('time')
        plt.ylabel(component + ' speed (m/s)')
        filt = ', filtered data' if filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)
        
        # plot data
        lines, labels = [], []
        if component != 'x':
            # plot distance
            if distance:
                for i in range(self.f1 + 1, len(lpos)):
                    x1, x2, y1, y2 = t[i], t[i], lspd[i], lspd[i] + (lpos[i,1] - fpos[i,1]) / 10
                    line3 = ax.plot([x1, x2], [y1, y2], c='0.8')
                lines.append(line3[0])
                labels.append('distance/10')
            # plot leader spd
            line1 = ax.plot(t[self.f1 + 1:], lspd[self.f1 + 1:])
            lines.append(line1[0])
            labels.append(str(self.leader))
        # plot follower spd
        line2 = ax.plot(t[frames], fspd[frames])
        lines.append(line2[0])
        labels.append('follower')
        
        # add legend
        ax.legend(lines, labels)        
        plt.tight_layout()
        plt.show()
    
    def plot_accelerations(self, component='', frames=None, distance=True, **kwargs):
        '''
            Plot the acceleration of the follower of follower and leader
            by time.
            args:
                component (str): 'x' lateral acceleration, 'y' forward acceleration,
                                default is total acceleration.
                frames (array of int): List of indices to be plotted.
                accelerations (boolean): Whether draw acceleration vectors.
                links (boolean): Whether draw links between the positions of 
                       follower and leader at the same moment for a sense 
                       of concurrency.
        '''
        
        # load kwargs
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        # get data 
        facc = np.linalg.norm(self.get_accelerations('f')[:, 0:2], axis=1)
        yrange = (-0.5, 2)
        if component == 'x':
            facc = self.get_accelerations('f')[:, 0]
            yrange = (-1, 1)
        elif component == 'y':
            facc = self.get_accelerations('f')[:, 1]
            yrange = (-0.5, 2)
        t = self.get_time(filtered)
        if not frames: frames = list(range(len(t)))
        
        # build figure
        fig = plt.figure()
        ax = plt.axes(xlim=(0, 12), ylim=yrange)
        plt.xlabel('time')
        plt.ylabel(component + ' acceleration (m^2/s)')
        filt = ', filtered data' if filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)
        
        # plot accelerations
        ax.plot(t[frames], facc[frames])
        plt.tight_layout()
        plt.show()
        
    def play_trial(self, frames=None, velocities = True, interval=11, save=False, **kwargs):
        '''
            Animate the trial. Red dot represents the leader, blue dot
            represent the follower.
            args:
                frames (array of int): List of indices to be plotted.
                velocities (boolean): Whether draw velocity vectors.
                interval (int): Delay between frames in milliseconds.
                save (boolean): Whether save animation as a video clip.
        '''
        
        # load kwargs
        rotated = True if 'rotated' not in kwargs else kwargs['rotated']
        filtered = True if 'filtered' not in kwargs else kwargs['filtered']
        
        # get data
        lpos = self.get_positions('l', **kwargs)
        lpos[:self.f1] = [99,99,0] # make leader out of the ploting range before its onset        
        fpos = self.get_positions('f', **kwargs)
        lspd = self.get_speeds('l', **kwargs)
        fspd = self.get_speeds('f', **kwargs)
        pos_x = np.stack((lpos[:,0], fpos[:,0]), axis=1)
        pos_y = np.stack((lpos[:,1], fpos[:,1]), axis=1)
        fvel = self.get_velocities('f', **kwargs)
        t = self.get_time(filtered)
        
        # set up the figure, the axis, and the plot element we want to animate
        fig = plt.figure(figsize=(4,7))
        if rotated:
            ax = plt.axes(xlim=(-3.5, 3.5), ylim=(-1, 15))
        else:
            ax = plt.axes(xlim=(-4.5, 4.5), ylim=(-5.5, 5.5))
        plt.xlabel('position x')
        plt.ylabel('position y')
        # set the aspect ratio equal to that of the actual value
        ax.set_aspect('equal')
        filt = ', filtered data' if filtered else ', raw data'
        plt.title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)       
        # initialize animation data
        leader, = ax.plot(lpos[0,0], lpos[0,1], 'ro', ms=10)
        follower, = ax.plot(fpos[0,0], fpos[0,1], 'bo', ms=10)
        clr = 'k' if velocities else 'w'
        sign = '+' if fspd[0] >= lspd[0] else '-' 
        s = str(round(fspd[0],2)) + '(' + sign + str(round(fspd[0]-lspd[0],2)) + ')m/s'
        spd = ax.text(fpos[0,0] + 0.5, fpos[0,1] - 0.5, s)
        time = ax.text(-2.5, -0.5, str(round(t[0], 2)))
        # 
        def animate_slow(i):
            '''
                slow animation function redraw everything at each frame. 
                Good for saving video but too slow to watch in real time.
            '''
            # ms is the short for markersize
            # figure labels and size
            ax.clear()
            if rotated:
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
            filt = ', filtered data' if filtered else ', raw data'
            ax.set_title('subject ' + str(self.subject_id) + ' trial ' + str(self.trial_id) + '\n v0 = ' + str(self.v0) + filt)       
            
            # update data
            leader, = ax.plot(lpos[i,0], lpos[i,1], 'ro', ms=10)
            follower, = ax.plot(fpos[i,0], fpos[i,1], 'bo', ms=10)
            sign = '+' if fspd[i] >= lspd[i] else '-' 
            s = str(round(fspd[i],2)) + '(' + sign + str(round(fspd[i]-lspd[i],2)) + ')m/s'
            time.set_text(str(round(t[i], 2)))
            spd = ax.text(fpos[i,0] - 1, fpos[i,1] - 0.7, s)
            arr = ax.arrow(fpos[i,0], fpos[i,1], fvel[i,0], fvel[i,1], head_width=0.1, length_includes_head=True, color=clr)
            return leader, follower, spd, arr, time

        def animate_fast(i):
            '''
                Fast animation function update without clear. Good for
                watching in real time, but will leave trace if saved.
            '''
            # ms is the short for markersize
            leader.set_data(lpos[i,0], lpos[i,1])
            follower.set_data(fpos[i,0], fpos[i,1])
            sign = '+' if fspd[i] >= lspd[i] else '-' 
            s = str(round(fspd[i],2)) + '(' + sign + str(round(fspd[i]-lspd[i],2)) + ')m/s'
            spd.set_text(s)
            spd.set_position((fpos[i,0] - 1, fpos[i,1] - 0.7))
            time.set_text(str(round(t[i], 2)))
            arr = ax.arrow(fpos[i,0], fpos[i,1], fvel[i,0], fvel[i,1], head_width=0.1, length_includes_head=True, color=clr)
            return leader, follower, spd, arr, time
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

