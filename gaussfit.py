from __future__ import print_function

class GaussFit(object):
    """Defines a Gaussian fit to a series of points.
    
    Notes
    -----
    This class fits the log of the points to a parabola, so
    points are assumed to be from a positive definite region
    of the function we want to approximate. If the function
    we want to approximate is a pdf from which samples are
    available, consider using
    utils.VariationalMixtureOfGaussians.
    
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.dim = X.shape[1]
        self.fit(X, Y)

    def fit(self, X, Y):
        from BayesianLinearRegression.myBayesianLinearRegression import BLR as fitter
        #from BayesianLinearRegression.rvm import RelevanceVectorMachine as fitter
        from BayesianLinearRegression.myExpansionBasis import Basis

        order = [3] * self.dim
        if self.dim == 1:
            order = 3

        self._quad_idx = range(3)  # [0, 1, 2, 3, 4, 6]
        for i in range(1, 3):
            self._quad_idx += [3 * i + j for j in range(3 - i)]
        print(self._quad_idx)
        self.idx_term = ['', 'x', 'x^2', 'y', 'xy', 'y^2']

        quad_basis = Basis('monomial', order=order, dim=self.dim)
        design_matrix = quad_basis.eval(X[0]).T.copy()
        for i in range(len(X) - 1):
            phi = quad_basis.eval(X[i + 1]).T
            design_matrix = np.vstack((design_matrix, phi))

        design_matrix = design_matrix#[:, self._quad_idx]
        blr = fitter()
        print(X, Y)
        blr.regression(Y, design_matrix=design_matrix)
        print(blr.m)
        mean_fit_str = ' '.join(['{:+.3f}{}'.format(v, t) for v, t in zip(blr.m.ravel(), self.idx_term)])
        print('mean: {}'.format(mean_fit_str))
        covariance = np.zeros((self.dim, self.dim))
        for i in range(1, self.dim + 1):
            covariance[i-1, i-1] = -2. * blr.m.ravel()[3**i - 3**(i-1)]
        print(covariance)

if __name__ == "__main__":

    import numpy as np
    import scipy.stats
    np.random.seed(0)
    nsamples = 200
    f = scipy.stats.multivariate_normal.pdf
    X = np.random.normal(size=(nsamples, 3))
    Y = np.log(scipy.stats.multivariate_normal.pdf(X, mean=np.zeros(X.shape[1]), cov=np.eye(X.shape[1])))
    # print(X, Y)
    gfit = GaussFit(X, Y[:, np.newaxis])

