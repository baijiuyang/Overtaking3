import numpy as np

def max_average(data, window):
    '''
    args:
        data (1d array): An array to be averaged. 
        window (int): Number of frames over thich the average
        is computed.
    return:
        The maximum averaged value over <window> size window.
    '''
    if window > len(data): 
        raise Exception('Window length is bigger than array')
    window = int(window)
    max_mean = 0
    for i in range(len(data) - window + 1):
        mean = sum(data[i:i + window]) * 1.0 / window
        if mean > max_mean:
            max_mean = mean
    return max_mean
   
def find_intersections(x1, x2, tolerance):
    '''
    Find the indices on which the difference between two variables are
    smaller than a tolerance.
    
    Args:
        x1, x2 (numpy array): Two curves.
    Return:
        Indices (int) of intersections.
        
    '''
    mask = abs(x1 - x2) < tolerance
    return np.argwhere(np.diff(mask) != 0).reshape(-1)
    
def expansions(lpos, fpos, lvel, fvel, relative, w):
    '''
    return the rate of (relative) optical expansion or contraction
    of the leader in the eye of the follower
    
    Args:
        lpos, fpos (2-d np array of float): Leader and follower positions
        in meter.
        lvel, fvel (2-d np array of float): Leader and follower velocities
        in meter / second.
        relative (boolean): Whether compute relative rate of expansion.
        w (float): The size of the leader in meters. Height or width.
    Return:
        e (1-d np array of float): An array of optical expansion.
    '''
    axis_val = 1 if len(lpos.shape) == 2 else None
    dspd = np.linalg.norm(fvel - lvel, axis=axis_val)
    dist = np.linalg.norm(lpos - fpos, axis=axis_val)
    e = w * dspd / (dist ** 2 + w ** 2 / 4)
    if relative:
        e /= 2 * np.arctan(w / (2 * dist))
    return e

def running_average(data):
    '''
    Args:
        data (2-d np array of float): With size (steps, number of variables).
    Return:
        Array(s) of float representing the running average of all variables
        in data.        
    '''
    if len(data.shape) == 2:
        ns = np.expand_dims(np.arange(1, len(data) + 1), axis=1)
    elif len(data.shape) == 1:
        ns = np.arange(1, len(data) + 1)
    else:
        raise Exception('Data dimension larger than 2')
    return np.cumsum(data, axis=0) / ns

def time_to_contact(lpos, fpos, lspd, fspd):
    '''
    Calculate the time to contact in 1-d case, assuming constant speed.
    
    Args:
        lpos, fpos (1-d np array of float): Leader and follower positions
        in meter.
        lspd, fspd (1-d np array of float): Leader and follower speeds
        in meter / second.
    Return:
        1-d np array of float, representing array of float Time to contact
        at each moment.
    '''
    return (lpos - fpos) / (fspd - lspd)
    










    