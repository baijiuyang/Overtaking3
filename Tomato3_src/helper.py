import numpy as np

def angle_overtake(lpos, fpos, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the angle between y axis 
        and the line connecting leader and follower is larger than a threshold
        args:
            lpos: time series of leader position in meter
            fpos: time series of follower position in meter
            threshold: angle in degree
    '''
    threshold = np.cos(threshold * np.pi / 180)
    vec1 = lpos[-1, 0:2] - fpos[-1, 0:2]
    vec0 = [0, 1]
    cos = np.dot(vec0, vec1)/np.linalg.norm(vec1)
    if cos < threshold:
        return True
    else:
        return False

def lateral_overtake(fpos, fspd, v0, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the maximum lateral deviation 
        of follower is larger than a threshold
        args:
            fpos: time series of follower position in meter
            fspd: time series of follower speed in meter/second
            v0: speed of the leader in meter/second
            threshold: distance in meter
    '''
    _fspd = fspd[np.argmax(abs(fpos[:, 0]))]
    if max(abs(fpos[:, 0])) > threshold and _fspd > v0:
        return True
    else:
        return False
        
def overtake_rate(subject, threshold):
    '''
        return a dictionary. The keys are v0, the values are overtake rates.
        Overtake is labeled by the lateral criterion.
        args:
            subject: an instance of the Subject class
    '''
    count = {0.8:0, 0.9:0, 1.0:0, 1.1:0, 1.2:0, 1.3:0}
    for i, t in subject.trials.items():
        fpos = t.get_positions('f')
        fspd = t.get_speeds('f')
        if lateral_overtake(fpos, fspd, t.v0, threshold):
            count[t.v0] += 0.1
    return count

def max_freewalk_spd(subject):
    '''
        return the maximum speed in all freewalk trials.
        args:
            subject: an instance of the Subject class
    '''
    max_fspd = 0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        _max = max(fspd)
        if max_fspd < _max:
            max_fspd = _max
    return max_fspd
    
def average_freewalk_spd(subject, window):
    '''
        return the average speed during the last <window> seconds of all 
        freewalk trials.
        args:
            subject: an instance of the Subject class
            window: window of averaging in seconds
    '''
    sum_fspd = 0
    num = 0
    for i, t in subject.freewalk.items():
        fspd = t.get_speeds('f')
        sum_fspd += sum(fspd[-window * t.Hz:])
        num += window * t.Hz
    return sum_fspd / num
    
    
    
    