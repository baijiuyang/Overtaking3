import numpy as np

def angle_overtake(lpos, fpos, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the angle between y axis 
        and the line connecting leader and follower is larger than a threshold
        args:
            lpos: time series of leader position
            fpos: time series of follower position
            threshold: angle in degree
    '''
    threshold = np.cos(threshold * np.pi / 180)
    vec1 = lpos[-1, 0:2] - fpos[-1, 0:2]
    vec0 = [0, 1]
    cos = np.dot(vec0, vec1)/np.linalg.norm(vec1)
    if cos <= threshold:
        return True
    else:
        return False

def lateral_overtake(fpos, threshold):
    '''
        return a boolean value indicating whether the follower overtake
        the leader by the criterion of whether the maximum lateral deviation 
        of follower is larger than a threshold
        args:
            fpos: time series of follower position
            threshold: distance in meter
    '''
    if max(abs(fpos[:, 0])) > threshold:
        return True
    else:
        return False