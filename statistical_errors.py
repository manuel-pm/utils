from __future__ import print_function

import numpy as np
try:
    from scipy.special import erfinv
    has_scipy = True
except:
    has_scipy = False


# Root mean square error
def rmse(Y_obs, Y_pred):
    n_data = Y_obs.shape[0]
    E = Y_obs - Y_pred
    return np.sqrt(np.dot(E.T, E) / n_data)


# Standarized root mean square error
def srmse(Y_obs, Y_pred):
    return rmse(Y_obs, Y_pred)/np.std(Y_obs)
    # n_data = Y_obs.shape[0]
    # sE = (Y_obs - Y_pred) / np.std(Y_obs)
    # return np.sqrt(np.dot(sE.T, sE) / n_data)


# Mean absolute error
def mae(Y_obs, Y_pred):
    n_data = Y_obs.shape[0]
    E = Y_obs - Y_pred
    return np.sum(np.abs(E)) / n_data


# Standarized mean absolute error
def smae(Y_obs, Y_pred):
    n_data = Y_obs.shape[0]
    sE = (Y_obs - Y_pred) / np.std(Y_obs)
    return np.sum(np.abs(sE)) / n_data


# Mean signed error
def mse(Y_obs, Y_pred):
    n_data = Y_obs.shape[0]
    E = Y_obs - Y_pred
    return np.sum(E) / n_data


# Standarized mean signed error
def smse(Y_obs, Y_pred):
    n_data = Y_obs.shape[0]
    sE = (Y_obs - Y_pred) / np.std(Y_obs)
    return np.sum(sE) / n_data


# Mean standarized log loss for gaussian predictive distribution
# A.K.A. Negative log predictive density (for gaussian predictive distribution)
def msll_g(Y_obs, Y_pred, sigma_pred):
    n_data = Y_obs.shape[0]
    E = Y_obs - Y_pred
    nll = 0.5 * np.log(2 * np.pi * sigma_pred**2) + 0.5 * (E / sigma_pred)**2
    return np.sum(nll) / n_data


# Mean standarized log loss
# A.K.A. Negative log predictive density
def msll(Y_obs, pred_dist):
    n_data = Y_obs.shape[0]
    nll = -np.log(pred_dist(Y_obs))
    return np.sum(nll) / n_data


def ltqnorm(p):
    """
    Modified from the author's original perl code (original comments follow below)
    by dfield@yahoo-inc.com.  May 3, 2004.

    Lower tail quantile for standard normal distribution function.

    This function returns an approximation of the inverse cumulative
    standard normal distribution function.  I.e., given P, it returns
    an approximation to the X satisfying P = Pr{Z <= X} where Z is a
    random variable from the standard normal distribution.

    The algorithm uses a minimax approximation by rational functions
    and the result has a relative error whose absolute value is less
    than 1.15e-9.

    Author:      Peter J. Acklam
    Time-stamp:  2000-07-19 18:26:14
    E-mail:      pjacklam@online.no
    WWW URL:     http://home.online.no/~pjacklam
    """

    if p <= 0 or p >= 1:
        # The original perl code exits here, we'll throw an exception instead
        raise ValueError( "Argument to ltqnorm %f must be in open interval (0,1)" % p )

    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if p < plow:
       q  = math.sqrt(-2*math.log(p))
       return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for upper region:
    if phigh < p:
       q  = math.sqrt(-2*math.log(1-p))
       return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# Percentage of observations within 100*CI% of predictions for gaussian pred dis
def within_ci_g(Y_obs, Y_pred, sigma_pred, ci=0.95):
    n_data = Y_obs.shape[0]
    if has_scipy:
        ci = np.sqrt(2) * erfinv(ci) * sigma_pred
    else:
        ci = ltqnorm(0.5 * (1. + ci)) * sigma_pred

    within = np.logical_and(Y_obs > Y_pred - ci,
                            Y_obs < Y_pred + ci)
    return np.count_nonzero(within) / n_data

# Percentage of observations within 100*CI% of predictions
# FIXME. Just uses mean and std. dev. Not properly calculated
def within_ci(Y_obs, pred_dist, ci=0.95):
    y = np.repeat(np.linspace(-100., 100., 10000), (Y_obs.shape[0], 1))
    Y_pred = np.sum(y*pred_dist(y), axis=0) * (y[1] - y[0])
    sigma
    n_data = Y_obs.shape[0]
    if has_scipy:
        ci = np.sqrt(2) * erfinv(ci) * sigma_pred
    else:
        ci = ltqnorm(0.5 * (1. + ci)) * sigma_pred

    within = np.logical_and(Y_obs > Y_pred - ci,
                            Y_obs < Y_pred + ci)
    return np.count_nonzero(within) / n_data

