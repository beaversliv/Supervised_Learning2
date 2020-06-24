#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel as sk_polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel


def gaussian_kernel_2(sigma):
    def k(x, y):
        return rbf_kernel(x, y, gamma=sigma)
    return k

def polynomial_kernel(d):
    def kernel(X, Y):
        return sk_polynomial_kernel(X, Y, degree=d, gamma=1.0, coef0=0.0)

    return kernel
