from __future__ import print_function

import numpy as np
import scipy.special as sp
import scipy.stats as st

from utils.bcolors import print_error, print_success, print_warning


class KMeans:
    def __init__(self, X, number_components, dissimilarity='euclidean_sq'):
        self.K = number_components
        self.D = len(X[0, :])
        self.ndata = len(X[:, 0])
        self.x = X
        data_mean = np.mean(X, axis=0)
        data_var = (data_mean/10)**2
        np.random.seed(0)
        noise = np.random.multivariate_normal(np.zeros(self.D),
                                              np.diag(data_var),
                                              size=self.K)
        self.m = np.tile(data_mean.reshape((1, self.D)), (self.K, 1))
        self.m = self.m + noise
        self.r = np.zeros((self.ndata, self.K))
        if dissimilarity in ['euclidean', 'L2']:
            self.dissimilarity = self.euclidean
        elif dissimilarity == 'euclidean_sq':
            self.dissimilarity = self.euclidean_sq
        else:
            self.dissimilarity = dissimilarity

    @staticmethod
    def euclidean(x1, x2):
        d = np.sqrt(np.einsum('...i, ...i', x1-x2, x1-x2))
        return d

    @staticmethod
    def euclidean_sq(x1, x2):
        d = np.einsum('...i, ...i', x1-x2, x1-x2)
        return d

    def distortion(self):
        tmp = self.x[:, np.newaxis, :] - self.m[np.newaxis, :, :]
        j = np.sum(self.r*np.einsum('ijk, ijk -> ij', tmp, tmp))
        # j = 0
        # for n in range(self.ndata):
        #     for k in range(self.K):
        #         d = np.dot(self.x[n]-self.m[k], self.x[n]-self.m[k])
        #         j += self.r[n, k]*d
        return j

    def compute(self, max_iter=100):
        for i in range(max_iter):
            m_pre = self.m.copy()
            self.r = np.zeros((self.ndata, self.K))
            for n in range(self.ndata):
                d = self.dissimilarity(self.x[n], self.m)
                kmin = np.argmin(d)
                self.r[n, kmin] = 1.

            self.m = np.einsum('nk, nd', self.r, self.x).T
            self.m = np.divide(self.m,
                               self.r.sum(axis=0).reshape((self.K, 1)))
            m_pre -= self.m
            delta = np.divide(np.linalg.norm(m_pre, axis=1),
                              np.linalg.norm(self.m, axis=1))
            if (delta < 1e-4).all():
                break
        if i == max_iter-1:
            print_warning("maximum iterations exceeded. You can increase them "
                          "with the parameter max_iters.")
        else:
            print_success("K-means converged in {} iterations.".format(i))
        print("K-means result:")
        print(self.m.T)


class VariationalMixtureOfGaussians:
    def __init__(self, X, number_components, r=None, alpha0=None,
                 beta0=None, nu0=None, m0=None):
        self.K = number_components
        self.D = len(X[0, :])
        self.ndata = len(X[:, 0])
        self.x = X

        if alpha0 is not None:
            self.alpha0 = alpha0
        else:
            self.alpha0 = 1.e-4
        if beta0 is not None:
            self.beta0 = beta0
        else:
            self.beta0 = 1.e-4
        if nu0 is not None:
            self.nu0 = nu0
        else:
            self.nu0 = self.D
        if m0 is not None:
            self.m0 = m0
        else:
            self.m0 = np.zeros(self.D)
        self.W0i = self.alpha0*np.eye(self.D)

        if r is None:
            self.r = np.random.uniform(size=(self.ndata, self.K))
            norm = self.r.sum(axis=1)
            self.r = np.dot(np.linalg.inv(np.diag(norm)), self.r)
        elif r == 'K-means':
            self.kmm = KMeans(X, self.K)
            self.kmm.compute()
            self.r = self.kmm.r
        else:
            self.r = r
            norm = self.r.sum(axis=1)
            self.r = np.divide(self.r, norm.reshape(self.ndata, 1))

        self.filled = False

        print(self.K, self.D, self.ndata)

    def __call__(self, X):
        if not self.filled:
            print_error("call compute method before evaluating")
            return None
        return self.eval(X)

    def reset(self, X=None, number_components=None, r=None, alpha0=None,
              beta0=None, nu0=None, m0=None):
        if X is not None:
            self.D = len(X[0, :])
            self.ndata = len(X[:, 0])
            self.x = X
        if number_components is not None:
            self.K = number_components
        if r is not None:
            self.r = r
            norm = self.r.sum(axis=1)
            self.r = np.divide(self.r, norm.reshape(self.ndata, 1))
        if alpha0 is not None:
            self.alpha0 = alpha0
        if beta0 is not None:
            self.beta0 = beta0
        if nu0 is not None:
            self.nu0 = nu0
        if m0 is not None:
            self.m0 = m0

    def prune_arrays(self, prune):
        self.K = prune.sum()
        self.r = self.r[:, prune]
        norm = self.r.sum(axis=1)
        self.r = np.divide(self.r, norm.reshape(self.ndata, 1))
        self.N = self.N[prune]
        self.alpha = self.alpha[prune]
        self.beta = self.beta[prune]
        self.nu = self.nu[prune]
        self.x_mean = self.x_mean[prune, :]
        self.m = self.m[prune, :]
        self.S = self.S[prune, :, :]
        self.Wi = self.Wi[prune, :, :]
        self.ln_Ltilde = self.ln_Ltilde[prune]
        self.ln_pitilde = self.ln_pitilde[prune]

    def update_distribution_parameters(self):
        self.N = self.r.sum(axis=0)
        self.alpha = self.alpha0 + self.N
        self.beta = self.beta0 + self.N
        self.nu = self.nu0 + self.N
        self.x_mean = np.einsum('nk, nd', self.r, self.x).T
        self.x_mean = np.divide(self.x_mean,
                                (self.N+1e-5).reshape((self.K, 1)))
        self.m = np.divide(self.beta0*self.m0 +
                           np.multiply(self.N.reshape(self.K, 1),
                                       self.x_mean),
                           self.beta.reshape(self.K, 1))
        tmp = (self.x.reshape(self.ndata, 1, self.D) -
               self.x_mean.reshape(1, self.K, self.D))
        self.S = np.einsum('n..., n...i, n...j', self.r, tmp, tmp)
        self.S = np.divide(self.S, (self.N+1e-5).reshape((self.K, 1, 1)))
        self.Wi = (np.tile(self.W0i, (self.K, 1, 1)) +
                   np.multiply(self.N.reshape(self.K, 1, 1), self.S) +
                   np.multiply(np.divide(self.beta0*self.N,
                                         self.beta0 +
                                         self.N).reshape(self.K, 1, 1),
                               np.einsum('...i, ...j',
                                         self.x_mean - self.m0,
                                         self.x_mean - self.m0)))

    def update_rnk(self):
        W = np.linalg.inv(self.Wi)
        log_det_W = np.log(np.linalg.det(W))
        L = sp.psi((self.nu.reshape(self.K, 1) + 1 -
                    np.arange(self.D).reshape(1, self.D))/2.).sum(axis=1)
        self.ln_Ltilde = L + self.D*np.log(2) + log_det_W
        alpha_hat = self.alpha.sum()
        self.ln_pitilde = sp.psi(self.alpha) - sp.psi(alpha_hat)
        arg1 = self.ln_pitilde + self.ln_Ltilde/2
        dx = (self.x.reshape(self.ndata, 1, self.D) -
              self.m.reshape(1, self.K, self.D))
        E = np.einsum('...i, ...ij, ...j', dx, W, dx)
        E = np.multiply(self.nu.reshape(1, self.K), E)
        E = E + (self.D/self.beta).reshape(1, self.K)
        E = arg1.reshape(1, self.K) - E/2.
        self.r = np.exp(E)
        norm = np.nan_to_num(self.r).sum(axis=1)
        self.r = np.divide(self.r, norm.reshape(self.ndata, 1))

    def compute(self, max_iter=1000, prune_level=None):
        self.update_distribution_parameters()
        for i in range(max_iter):
            m_pre = self.m.copy()
            self.update_rnk()
            self.update_distribution_parameters()
            m_pre -= self.m
            delta = (np.linalg.norm(m_pre, axis=1) /
                     np.linalg.norm(self.m, axis=1))
            if prune_level is not None:
                prune = self.N/self.N.sum() > prune_level
                if not prune.all():
                    self.prune_arrays(prune)
            if (delta < 1e-6).all():
                break
        if i == max_iter-1:
            print_warning("maximum iterations exceeded. You can increase them "
                          "with the parameter max_iters or just run compute() "
                          "again")
        else:
            print_success("Converged in {} iterations.".format(i))
        print("Means:")
        print(self.m.T)
        print("Covariances:")
        for k in range(self.K):
            print(self.Wi[k]/self.nu[k])

        print("Mixing coefficients:")
        alpha_hat = self.alpha.sum()
        print(self.alpha/alpha_hat)
        self.filled = True

    def variational_lower_bound(self):
        L = np.zeros(7)
        alpha_hat = self.alpha.sum()
        ln_C_alpha = (sp.gammaln(alpha_hat) -
                      sp.gammaln(self.alpha).sum())
        ln_C_alpha0 = (sp.gammaln(self.K*self.alpha0) -
                       self.K*sp.gammaln(self.alpha0))
        W = np.linalg.inv(self.Wi)
        L[0] = (np.multiply(self.N, self.ln_Ltilde - self.D/self.beta -
                self.nu*np.einsum('...ij, ...ji', self.S, W) -
                np.multiply(self.nu, np.einsum('...i, ...ij, ...j',
                                               self.x_mean-self.m,
                                               W,
                                               self.x_mean-self.m)) -
                self.D*np.log(2*np.pi)).sum()/2)
        L[1] = np.multiply(self.r, np.tile(self.ln_pitilde,
                                           (self.ndata, 1))).sum()
        L[2] = ln_C_alpha0 + (self.alpha0 - 1)*self.ln_pitilde.sum()
        L31 = (self.D*np.log(self.beta0/2/np.pi) + self.ln_Ltilde -
               self.D*self.beta0/self.beta -
               self.beta0*self.nu*np.einsum('...i, ...ij, ...j',
                                            self.m-self.m0,
                                            W,
                                            self.m-self.m0))
        ln_B0 = (-self.nu0/2.*np.log(1./np.linalg.det(self.W0i)) -
                 (self.nu0*self.D/2.*np.log(2.) +
                 self.D*(self.D-1)/4.*np.log(np.pi) +
                 sp.gammaln((self.nu0 + 1 -
                            np.arange(self.D))/2.).sum()))
        L[3] = (L31.sum()/2. + self.K*ln_B0 +
                ((self.nu0-self.D-1)*self.ln_Ltilde -
                self.nu*np.einsum('ij, kji', self.W0i, W)).sum()/2.)
        L[4] = -np.multiply(self.r+1.e-10, np.log(self.r+1.e-10)).sum()
        L[5] = -(np.multiply(self.ln_pitilde, self.alpha-1).sum() +
                 ln_C_alpha)
        ln_B = (-self.nu/2.*np.log(np.linalg.det(W)) -
                (self.nu*self.D/2.*np.log(2.) +
                self.D*(self.D-1)/4.*np.log(np.pi) +
                sp.gammaln((self.nu.reshape(self.K, 1) + 1 -
                           np.arange(self.D).reshape(1,
                                                     self.D))/2.).sum(axis=1)))
        H = -ln_B - (self.nu-self.D-1)/2.*self.ln_Ltilde + self.nu*self.D/2.
        L[6] = -(self.ln_Ltilde/2 + self.D/.2*np.log(self.beta/2./np.pi) -
                 self.D/2. - H).sum()

        return L.sum()

    def eval(self, X):
        Y = np.zeros_like(X)
        alpha_hat = self.alpha.sum()
        for k in range(self.K):
            Y += (self.alpha[k]/alpha_hat *
                  st.norm.pdf(X, self.m[k],
                              np.sqrt(self.Wi[k][0, 0]/self.nu[k])))
        return Y

    def plot(self):
        if self.D == 1:
            self.plot_1d()
        elif self.D == 2:
            self.plot_2d()
        else:
            print_error("plotting implemented for 1D and 2D data only.")

    def plot_1d(self):
        import matplotlib.pyplot as plt
        h = plt.hist(self.x[:, 0], bins=50, normed=True,
                     zorder=1, color='0.75', histtype='stepfilled')
        Xe = np.linspace(np.min(self.x[:, 0]), np.max(self.x[:, 0]), 10000)
        Ymix = np.zeros_like(Xe)
        alpha_hat = self.alpha.sum()
        most_relevant = np.argsort(-self.N)[:3]

        for k in range(self.K):
            Yvgm = (self.alpha[k]/alpha_hat *
                    st.norm.pdf(Xe, self.m[k],
                                np.sqrt(self.Wi[k][0, 0]/self.nu[k])))
            c = u'black'
            if k in most_relevant:
                if k == most_relevant[0]:
                    c = (1, 0, 0)
                elif k == most_relevant[1]:
                    c = (0, 1, 0)
                elif k == most_relevant[2]:
                    c = (0, 0, 1)
            plt.plot(Xe, Yvgm, linewidth=2, c=c)
            Ymix = Ymix + Yvgm

        pdfmax = np.max(Ymix)
        plt.plot(Xe, Ymix, linewidth=2, c=u'black')

        if hasattr(self, 'kmm'):
            for k in range(self.kmm.K):
                plt.axvline(self.kmm.m[k], c='black', linewidth=2)

        color = []
        for n in range(self.ndata):
            if len(most_relevant) == 3:
                color.append((self.r[n, most_relevant[0]],
                              self.r[n, most_relevant[1]],
                              self.r[n, most_relevant[2]]))
            elif len(most_relevant) == 1:
                color.append((self.r[n, most_relevant[0]], 0.0, 0.0))
            elif len(most_relevant) == 2:
                color.append((self.r[n, most_relevant[0]],
                              self.r[n, most_relevant[1]], 0.0))
        plt.scatter(X[:, 0], np.zeros_like(X[:, 0]),
                    marker='+', c=color, s=50,
                    zorder=5)
        plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
        plt.ylim((0, pdfmax*1.1))
        plt.show(block=True)

    def plot_2d(self):
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.subplot(111)
        from matplotlib.patches import Ellipse
        most_relevant = np.argsort(-self.N)[:3]

        for k in range(self.K):
            cov = self.Wi[k]/self.nu[k]
            mean = self.m[k]
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)
            ell = Ellipse(xy=(mean[0], mean[1]),
                          width=lambda_[0]*2, height=lambda_[1]*2,
                          angle=np.rad2deg(np.arccos(v[0, 0])))
            c = u'black'
            if k in most_relevant:
                if k == most_relevant[0]:
                    c = (1, 0, 0)
                elif k == most_relevant[1]:
                    c = (0, 1, 0)
                elif k == most_relevant[2]:
                    c = (0, 0, 1)
            ell.set_facecolor(c)
            ell.set_edgecolor(c)

            ell.set_linewidth(4)
            ell.set_alpha(0.4)
            ax.add_artist(ell)

        color = []
        for n in range(self.ndata):
            if len(most_relevant) == 3:
                color.append((self.r[n, most_relevant[0]],
                              self.r[n, most_relevant[1]],
                              self.r[n, most_relevant[2]]))
            elif len(most_relevant) == 1:
                color.append((self.r[n, most_relevant[0]], 0.0, 0.0))
            elif len(most_relevant) == 2:
                color.append((self.r[n, most_relevant[0]],
                              self.r[n, most_relevant[1]], 0.0))

        plt.scatter(X[:, 0], X[:, 1], c=color, edgecolors=color)
        if hasattr(self, 'kmm'):
            plt.scatter(self.kmm.m[:, 0], self.kmm.m[:, 1],
                        c=u'red', s=40, marker='x')
        plt.scatter(self.m[:, 0], self.m[:, 1], c=u'red', s=40, marker='^')
        plt.show(block=True)

"""
Example of use:
"""
if __name__ == "__main__":

    nsamples = 100
    X = np.zeros((nsamples, 1))
    prob = [0.2, 0.8]
    prob = np.array(prob)/np.array(prob).sum()
    mean = [0.2, 1.0]
    std = [0.1, 0.2]
    model = np.random.choice(len(prob), nsamples, p=prob)
    for i in range(nsamples):
        X[i] = np.random.normal(mean[model[i]], std[model[i]])

    # X = np.loadtxt(open('../faithful.dat', 'rb'), skiprows=1)

    vlb = []
    K = 5

    vgm = VariationalMixtureOfGaussians(X, K, r='K-means')
    vgm.compute(prune_level=0.05, max_iter=2000)
    vgm.plot_1d()
    vlb.append([vgm.variational_lower_bound(), np.log(np.math.factorial(K))])

    vlb = np.array(vlb)
    np.set_printoptions(precision=3)
    print("VLB:\n {}".format(vlb))
