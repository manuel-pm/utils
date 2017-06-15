from __future__ import print_function

import numpy as np

from utils.bcolors import print_error

class FiniteDifference:
    def __init__(self, grid):
        self.grid = grid
        self.h = grid[1] - grid[0]
        self.h2 = self.h*self.h
        self.fill_coeffs_1st()
        self.fill_coeffs_2nd()

    def derivative(self, order, point, function, acc_order=1, mode='c'):
        if order == 1:
            return self.first_derivative(point, function, acc_order, mode)
        elif order == 2:
            return self.second_derivative(point, function, acc_order, mode)
        else:
            print_error("order {} not implemented".format(order))
            raise NotImplementedError

    def first_derivative(self, point, function, order=1, mode='c'):
        if mode == 'c':
            f = function[point - order: point + order + 1]
            num = np.dot(f, self.c_coeffs_1st[order - 1])
        elif mode == 'f':
            f = function[point: point + order + 1]
            num = np.dot(f, self.f_coeffs_1st[order - 1])
        else:
            print_error("mode \'{}\' not implemented for first "
                        "derivative".format(mode))
            raise NotImplementedError
        return num / self.h

    def second_derivative(self, point, function, order=1, mode='c'):
        if mode == 'c':
            lp = function.shape[0] - 1
            if lp - point < order:
                if lp - point == 0:
                    return self.second_derivative(point, function, order, 'b')
                else:
                    return self.second_derivative(point, function,
                                                  lp - point, mode)
            if point < order:
                if point == 0:
                    return self.second_derivative(point, function, order, 'f')
                else:
                    return self.second_derivative(point, function,
                                                  point, mode)
            f = function[point - order: point + order + 1]
            num = np.dot(f, self.c_coeffs_2nd[order - 1])
        elif mode == 'f':
            f = function[point: point + order + 2]
            num = np.dot(f, self.f_coeffs_2nd[order - 1])
        elif mode == 'b':
            f = function[point - order - 2: point][::-1]
            num = np.dot(f, self.f_coeffs_2nd[order - 1])
        else:
            print_error("mode \'" + mode + "\' not yet implemented\nUse modes "
                                           "\'c\', \'f\' or \'b\'")
            raise NotImplementedError
        return num / self.h2

    def second_derivative_operator(self, size=None, start=None, end=None,
                                   order=1, mode='c'):
        if start is None:
            assert size is not None
            C = np.zeros((size, size))
            C[0, :order + 2] = self.f_coeffs_2nd[order - 1]
            for i in range(1, order):
                C[i, :2*i + 1] = self.c_coeffs_2nd[i - 1]
            for i in range(order, size - order):
                C[i, i - order:i + order + 1] = self.c_coeffs_2nd[order - 1]
            for i in range(size - order, size - 1):
                C[i, 2*i - size + 1:] = self.c_coeffs_2nd[size - 2 - i]
            C[-1, size - order - 2:] = self.f_coeffs_2nd[order - 1][::-1]
        elif start is not None:
            assert end is not None
            size = end - start
            tsize = self.grid.shape[0]
            C = np.zeros((size, size + 2*order))
            if start == 0:
                C[0, :order + 2] = self.f_coeffs_2nd[order - 1]
            for i in range(start, order):
                C[i - start, :2*i + 1] = self.c_coeffs_2nd[i - 1]
            for i in range(np.max((start, order)),
                            np.min((tsize - order, end))):
                C[i - start,
                  i - order:i + order + 1] = self.c_coeffs_2nd[order - 1]
            for i in range(tsize - order, np.min((tsize - 1, end))):
                C[i - start, 2*i - size + 1:] = self.c_coeffs_2nd[size - 2 - i]
            if end == tsize:
                C[-1, size - order - 2:] = self.f_coeffs_2nd[order - 1][::-1]
        return C / self.h2

    def fill_coeffs_1st(self):
        self.c_coeffs_1st = []
        self.c_coeffs_1st.append(np.array([-1./2, 0, 1./2]))
        self.c_coeffs_1st.append(np.array([1./12, -2./3, 0., 2./3, -1./12]))
        self.c_coeffs_1st.append(np.array([-1./60, 3./20, -3./4,
                                           0.,
                                           3./4, -3./20, 1./60]))
        self.c_coeffs_1st.append(np.array([1./280, -4./105, 1./5, -4./5,
                                           0.,
                                           4./5, -1./5, 4./105, -1./280]))
        self.f_coeffs_1st = []
        self.f_coeffs_1st.append(np.array([-1., 1.]))
        self.f_coeffs_1st.append(np.array([-3./2, 2., -1./2]))
        self.f_coeffs_1st.append(np.array([-11./6, 3., -3./2, 1./3]))
        self.f_coeffs_1st.append(np.array([-25./12, 4., -3., 4./3, -1./4]))

    def fill_coeffs_2nd(self):
        self.c_coeffs_2nd = []
        self.c_coeffs_2nd.append(np.array([1, -2, 1]))
        self.c_coeffs_2nd.append(np.array([-1./12, 4./3, -5./2, 4./3, -1./12]))
        self.c_coeffs_2nd.append(np.array([1./90, -3./20, 3./2,
                                           -49./18,
                                           3./2, -3./20, 1./90]))
        self.c_coeffs_2nd.append(np.array([-1./560, 8./315, -1./5, 8./5,
                                           -205./72,
                                           8./5, -1./5, 8./315, -1./560]))
        self.f_coeffs_2nd = []
        self.f_coeffs_2nd.append(np.array([1., -2., 1.]))
        self.f_coeffs_2nd.append(np.array([2., -5., 4., -1.]))
        self.f_coeffs_2nd.append(np.array([35./12, -26./3, 19./2,
                                           -14./3, 11./12]))
        self.f_coeffs_2nd.append(np.array([15./4, -77./6, 107./6,
                                           -13., 61./12, -5./6]))

    def fill_coeffs_3rd(self):
        self.c_coeffs_3rd = []
        self.c_coeffs_3rd.append(np.array([-1./2, 1, 0, -1, 1./2]))
        self.c_coeffs_3rd.append(np.array([1./8, -1, 13./8, 0, -13./8, 1, -1./8]))
        self.c_coeffs_3rd.append(np.array([-7./240, 3./10, -169./120, 61./30, 0,
                                           -61./30, 169./120, -3./10, 7./240]))

