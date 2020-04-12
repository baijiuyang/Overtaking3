import numpy as np
import helper

def angle_overtake(trial, threshold=55):
    '''
    return a boolean value indicating whether the follower overtake
    the leader by the criterion of whether the angle between y axis 
    and the line connecting leader and follower is larger than a threshold
    
    Args:
        trial: An instance of the Trial class
        threshold: Angle in degree
    '''
    threshold = np.cos(threshold * np.pi / 180)
    lpos = trial.get_positions('l')
    fpos = trial.get_positions('f')
    vec1 = lpos[-1, 0:2] - fpos[-1, 0:2]
    vec0 = [0, 1]
    cos = np.dot(vec0, vec1)/np.linalg.norm(vec1)
    if cos < threshold:
        return True
    else:
        return False
    
def lateral_overtake(trial, threshold=0.3, window=1):
    '''
    return a boolean value indicating whether the follower overtake
    the leader by the criterion of (1) the maximum lateral deviation 
    of follower is larger than a threshold, (2) the lateral deviation
    when leader appears is smaller than 0.2 meter, and (3) the max_average
    of fspd_y over 1 second window is larger than leader speed
    
    Args:
        trial: An instance of the Trial class
        threshold: Distance in meter
    '''
    
    fpos_x = trial.get_positions('f')[:, 0]
    fpos_x_max = max(abs(fpos_x[trial.f1:]))
    fspd_y = trial.get_velocities('f')[:, 1]
    fspd_y_max = helper.max_average(fspd_y, window)

    if valid_trial(trial) and fspd_y_max > trial.v0 and fpos_x_max > threshold:
        return True
    else:
        return False
        
def overtake_rates(subject, threshold=0.2):
    '''
    return a dictionary. The keys are v0, the values are overtake rates.
    Overtake is labeled by the lateral criterion.
    
    Args:
        subject: An instance of the Subject class
    '''
    count = {0.8:[0, 0], 0.9:[0, 0], 1.0:[0, 0], 
             1.1:[0, 0], 1.2:[0, 0], 1.3:[0, 0]}
    for i, t in subject.trials.items():
        # only count among valid trials
        if valid_trial(t):
            count[t.v0][0] += 1.0
        if lateral_overtake(t, threshold):
            count[t.v0][1] += 1.0
    for key in count.keys():
        count[key] = count[key][1] / count[key][0]
    return count

def max_freewalk_spd(subject):
    '''
    return the maximum speed in all freewalk trials.
    
    Args:
        subject: An instance of the Subject class
    '''
    max_fspd = 0.0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        _max = max(fspd)
        if max_fspd < _max:
            max_fspd = _max
    return max_fspd
   
def average_freewalk_spd(subject, window):
    '''
    Args:
        subject: An instance of the Subject class
        window (float): window of averaging in seconds
    Return:
        The averaged maximum averaged speed over <window> seconds window.
    '''
    sum_fspd = 0.0
    num = 0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        sum_fspd += helper.max_average(fspd, window * t.Hz)
        num += 1
    return sum_fspd / num
    
def overtake_onset(trial, tolerance=0.02):
    '''
    return the index when participants initiate overtaking, accoding to
    the following criterion: find an interval from leader appear to the
    peak of lateral speed before reaching the peak lateral deviation,
    then find the max among (1) time of leader appear, (2) time of speed 
    intersect with zero, and (3) time of speed interset with average speed
    
    Args:
        trial: An instance of the Trial class.
    Return:
        An int as the index of the onset of overtaking
    '''
    l = trial.f1
    fvel_x = trial.get_velocities('f')[:, 0]
    fpos_x = trial.get_positions('f')[:, 0]
    averge_x = helper.running_average(fvel_x)
    
    pos_peak = min(np.argmax(abs(fpos_x)), trial.length - trial.Hz)
    
    if pos_peak == 0: 
        pos_peak = 1
    ipeak = np.argmax(abs(fvel_x[:pos_peak]))
    z = helper.find_intersections(fvel_x[:ipeak], np.zeros(ipeak), tolerance)
    if len(z) == 0:
        z = 0
    else:
        z = z[-1]
    a = helper.find_intersections(fvel_x[:ipeak], averge_x[:ipeak], tolerance)
    if len(a) == 0:
        a = 0
    else:
        a = a[-1]
    return int(max(l, a, z))

def average_onset_delays(subject):
    '''
    Compute the average time between leader appears and onset of
    overtaking in each conditions for a subject. 
    
    Args:
        subject: An instance of the Subject class
    Return:
        An python dictionary. Keys are leader speeds, values are 
        the average onset delays in this condition.
    '''
    delays = {0.8:[0, 0], 0.9:[0, 0], 1.0:[0, 0], 
             1.1:[0, 0], 1.2:[0, 0], 1.3:[0, 0]}
    for i, t in subject.trials.items():
        if valid_trial(t) and lateral_overtake(t):
            delay = overtake_onset(t) - t.f1
            delays[t.v0][0] += 1.0
            delays[t.v0][1] += delay
    for key in delays.keys():
        if delays[key][0] == 0:
            delays[key] = delays[round(key - 0.1, 1)]
        else:
            delays[key] = int(delays[key][1] / delays[key][0])
    return delays
    
def average_onset_spds(subject):
    '''
    return an array that contains the average speed of a subject 
    when leader appears in each condition (from v0=0.8 to v0=1.3) 
    in experimental trials.
    
    Args:
        subject: An instance of the Subject class            
    '''
    fspds = [0.0] * 6
    for i, t in subject.trials.items():
        fspds[int(t.v0 * 10 - 8)] += t.get_speeds('f')[t.f1] / 10   
    return fspds
    
def expansion_at(trial, relative, frames=None, w=1.8):
    '''
    return the (relative) rate of expansion at a given frame in a trial.
    
    Args:
        trial: An instance of the Trial class.
        relative (boolean): Whether compute relative rate of expansion.
        frame (int): The frame of the needed expansion.
        w (float): The size of the leader.
    '''  
    if not frames: frames = list(range(trial.length))
    lpos = trial.get_positions('l')[frames, 0:2]
    fpos = trial.get_positions('f')[frames, 0:2]
    lvel = trial.get_velocities('l')[frames, 0:2]
    fvel = trial.get_velocities('f')[frames, 0:2]
    return helper.expansions(lpos, fpos, lvel, fvel, relative, w)
    
def average_expansions(subject, relative, w=1.8):
    '''
    return the average (relative) rate of expansion when leader appears
    in each condition.
    
    Args:
        subject: An instance of the Subject class  
        relative (boolean): Whether compute relative rate of expansion.
        w (float): The of the leader.
    '''
    es = [0.0] * 6
    for i, t in subject.trials.items():
        es[int(t.v0 * 10 - 8)] += expansion_at(t, relative, w=1.8) / 10
    return es

def time_to_pass(trial):
    '''
    Calculate the time-to-pass between follower and leader
    at each moment after leader appears.
    
    Args:
        trial: An instance of the Trial class. 
    Return:
        An array of time-to-pass.
    '''
    fspd_y = trial.get_velocities('f')[:, 1]
    fpos_y = trial.get_positions('f')[:, 1]
    lpos_y = trial.get_positions('l')[:, 1]
    return helper.time_to_contact(lpos_y, fpos_y, trial.v0, fspd_y)
    
def valid_trial(trial, threshold=0.2):
    '''
    Decide whether a trial is valid based on the criterion of
    follower's lateral deviation when leader appears. 
    
    Args:
        trial: An instance of the Trial class
        threshold: Distance in meter
    Return:
        A boolean representing whether the trial is valid
    '''
    return abs(trial.get_positions('f')[trial.f1, 0]) < threshold









    