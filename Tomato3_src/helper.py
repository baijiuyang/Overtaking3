import numpy as np

def angle_overtake(trial, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the angle between y axis 
        and the line connecting leader and follower is larger than a threshold
        args:
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

def lateral_overtake(trial, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the maximum lateral deviation 
        of follower is larger than a threshold
        args:
            trial: An instance of the Trial class
            threshold: Distance in meter
    '''
    fspd = trial.get_speeds('f')
    fpos = trial.get_positions('f')
    _fspd = fspd[np.argmax(abs(fpos[:, 0]))]
    if max(abs(fpos[:, 0])) > threshold and _fspd > trial.v0:
        return True
    else:
        return False
        
def overtake_rates(subject, threshold):
    '''
        return a dictionary. The keys are v0, the values are overtake rates.
        Overtake is labeled by the lateral criterion.
        args:
            subject: An instance of the Subject class
    '''
    count = {0.8:0, 0.9:0, 1.0:0, 1.1:0, 1.2:0, 1.3:0}
    for i, t in subject.trials.items():
        fpos = t.get_positions('f')
        fspd = t.get_speeds('f')
        if lateral_overtake(t, threshold):
            count[t.v0] += 0.1
    return count

def max_freewalk_spd(subject):
    '''
        return the maximum speed in all freewalk trials.
        @args:
            subject: An instance of the Subject class
    '''
    max_fspd = 0.0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        _max = max(fspd)
        if max_fspd < _max:
            max_fspd = _max
    return max_fspd

def max_average(data, window):
    '''
        @args:
            data (1d array): An array to be averaged. 
            window (int): window of averaging in seconds
        @return:
            The maximum averaged value over <window> size window.
    '''
    if window > len(data): 
        raise Exception('Window length is bigger than array')
    window = int(window)
    means = []
    for i in range(len(data) - window + 1):
        means.append(sum(data[i:i + window]) * 1.0 / window)
    return max(means)
   
def average_freewalk_spd(subject, window):
    '''
        @args:
            subject: An instance of the Subject class
            window (float): window of averaging in seconds
        @return:
            The averaged maximum averaged speed over <window> seconds window.
    '''
    sum_fspd = 0.0
    num = 0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        sum_fspd += max_average(fspd, window * t.Hz)
        num += 1
    return sum_fspd / num

def find_intersections(x1, x2, tolerance):
    '''
        Find the intersections of two curve. Returns the index of those
        intersections.
        args:
            x1, x2 (numpy array): Two curves.
    '''
    mask = abs(x1 - x2) < tolerance
    return np.argwhere(np.diff(mask) != 0).reshape(-1)
    
def overtake_onset(trial, tolerance=0.01):
    '''
        return the index when participants initiate overtaking, accoding to
        certain criterion.
        args:
            trial: An instance of the Trial class.
    '''
    l = trial.f1
    fvel_x = trial.get_velocities('f')[:, 0]
    fpos_x = trial.get_positions('f')[:, 0]
    averge_x = running_average(trial, 'vel')[:, 0]
    pos_peak = np.argmax(abs(fpos_x))
    ipeak = np.argmax(abs(fvel_x[:pos_peak]))
    z = find_intersections(fvel_x[:ipeak], np.zeros(ipeak), tolerance)
    if len(z) == 0:
        z = 0
    else:
        z = z[-1]
    a = find_intersections(fvel_x[:ipeak], averge_x[:ipeak], tolerance)
    if len(a) == 0:
        a = 0
    else:
        a = a[-1]
    return int(max(l, a, z))
    
def average_onset_spds(subject):
    '''
        return an array that contains the average speed of a subject 
        when leader appears in each condition (from v0=0.8 to v0=1.3) 
        in experimental trials.
        args:
            subject: An instance of the Subject class            
    '''
    fspds = [0.0] * 6
    for i, t in subject.trials.items():
        fspds[int(t.v0 * 10 - 8)] += t.get_speeds('f')[t.f1] / 10   
    return fspds
    
def expansion(lpos, fpos, lvel, fvel, relative, w=1.8):
    '''
        return the rate of (relative) optical expansion or contraction
        of the leader in the eye of the follower
        args:
            lpos (np array of float): Time series of leader position.
            fpos (np array of float): Time series of follower position.
            lvel (np array of [float, float]): Time series of leader velocity.
            fvel (np array of [float, float]): Time series of follower velocity.
            relative (boolean): Whether compute relative rate of expansion.
            w (float): The of the leader.
    '''
    dspd = np.linalg.norm(fvel - lvel)
    dist = np.linalg.norm(lpos - fpos)
    e = w * dspd / (dist ** 2 + w ** 2 / 4)
    if relative:
        e /= 2 * np.arctan(w / (2 * dist))
    return e
    
def expansion_at(trial, relative, frame=None, w=1.8):
    '''
        return the (relative) rate of expansion at a give frame in a trial.
        args:
            trial: An instance of the Trial class.
            relative (boolean): Whether compute relative rate of expansion.
            frame (int): The frame of the needed expansion.
            w (float): The of the leader.
    '''  
    if not frame: frame = trial.f1
    lpos = trial.get_positions('l')[frame, 0:2]
    fpos = trial.get_positions('f')[frame, 0:2]
    lvel = trial.get_velocities('l')[frame, 0:2]
    fvel = trial.get_velocities('f')[frame, 0:2]
    return expansion(lpos, fpos, lvel, fvel, relative, w)
    
def average_expansions(subject, relative, w=1.8):
    '''
        return the average (relative) rate of expansion when leader appears
        in each condition.
        args:
            subject: An instance of the Subject class  
            relative (boolean): Whether compute relative rate of expansion.
            w (float): The of the leader.
    '''
    es = [0.0] * 6
    for i, t in subject.trials.items():
        es[int(t.v0 * 10 - 8)] += expansion_at(t, relative, w=1.8) / 10
    return es

def running_average(trial, var):
    '''
        return the running average of the variable var of the follower
        args:
            trial: An instance of the Trial class. 
            var (str): 'pos' for position, 'vel', for velocity,
                        'acc' for acceleration. All for followers.
    '''
    if var == 'pos':
        data = trial.get_positions('f')
    elif var == 'vel':
        data = trial.get_velocities('f')
    elif var == 'acc':
        data = trial.get_accelerations('f')
    ns = np.expand_dims(np.linspace(1, len(trial.tstamps), len(trial.tstamps)), axis=1)
    return np.cumsum(data, axis=0) / ns













    