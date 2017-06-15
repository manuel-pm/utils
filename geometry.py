import numpy as np
try:
    import scipy as sp
    scipy_available = True
except ImportError:
    sp = None
    scipy_available = False

from utils.bcolors import print_error


def cross_product_matrix(v):
    return np.array([[0., -v[2], v[1]],
                     [v[2], 0., -v[0]],
                     [-v[1], v[0], 0.]])


def tensor_product(u, v):
    assert len(u) == len(v)
    d = len(u)
    return np.dot(u.reshape(d, 1), v.reshape(1, d))


def rotation1(axis, angle, degrees=False):
    """ Rotation matrix to rotate angle rad around axis

    Parameters
    ----------
    axis : np.ndarray
        Unit vector defining rotation direction
    angle : np.ndarray
        Angle of rotation around axis

    See Also
    --------
    rotation

    """
    if degrees:
        angle = angle*np.pi/180.
    u = axis/np.linalg.norm(axis)
    c, s = np.cos(angle), np.sin(angle)
    return (c*np.eye(3) +
            s*cross_product_matrix(u) +
            (1 - c)*tensor_product(u, u))

def rotation2(u, v):
    """ Rotation matrix to bring direction u to direction v

    Parameters
    ----------
    u : np.ndarray
        Unit vector defining source direction
    v : np.ndarray
        Unit vector defining destination direction

    """
    cp = np.cross(u, v)
    s = np.linalg.norm(cp)
    c = np.dot(u, v)
    cpm = cross_product_matrix(cp)
    R = np.eye(3) + cpm + np.dot(cpm, cpm)*(1 - c)/(s*s)
    # np.dot(R, u) = v
    return R

def rotation(axis, angle, degrees=False):
    if not scipy_available:
        print_error("Scipy not available. Use rotation1 instead")
        raise NotImplementedError
    if degrees:
        angle = angle*np.pi/180.
    u = axis/np.linalg.norm(axis)
    return sp.linalg.expm(np.cross(np.eye(3), u*angle))

def rotation_x(angle, degrees=False):
    if degrees:
        angle = angle*np.pi/180.
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.],
                     [0.,  c, -s],
                     [0.,  s,  c]])

def rotation_z(angle, degrees=False):
    if degrees:
        angle = angle*np.pi/180.
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[ c, -s, 0.],
                     [ s,  c, 0.],
                     [0., 0., 1.]])

def rotation_euler(alpha, beta, gamma, degrees=False):
    if degrees:
        alpha = alpha*np.pi/180.
        beta = beta*np.pi/180.
        gamma = gamma*np.pi/180.
    return np.dot(np.dot(rotation_z(gamma), rotation_x(beta)), rotation_z(alpha))

