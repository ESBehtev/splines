# -*- coding: utf-8 -*-
"""P-сплайн.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1lyjb0Hrpy1m-_glLpfo45jr6EvAWLwJN
"""

import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt

def p_spline_basis(x, knots, degree):
    """Construct B-spline basis functions for given x, knots, and degree."""
    n_bases = len(knots) - degree - 1
    basis = np.zeros((len(x), n_bases))
    for i in range(n_bases):
        coeffs = np.zeros(n_bases)
        coeffs[i] = 1
        spline = BSpline(knots, coeffs, degree)
        basis[:, i] = spline(x)
    return basis

def difference_matrix(n_bases, order):
    """Construct difference matrix of given order for n_bases."""
    D = np.eye(n_bases)
    for _ in range(order):
        D = np.diff(D, n=1, axis=0)
    return D

def fit_p_spline(x, y, n_knots, degree, lam, diff_order):
    """Fit a P-spline to data (x, y)."""
    # Create knots
    x_min, x_max = np.min(x), np.max(x)
    knots = np.linspace(x_min, x_max, n_knots - degree + 2)
    # Pad knots for spline degree
    knots = np.concatenate((
        np.repeat(x_min, degree),
        knots,
        np.repeat(x_max, degree)
    ))
    # Build B-spline basis
    B = p_spline_basis(x, knots, degree)
    n_bases = B.shape[1]
    # Construct penalty matrix
    D = difference_matrix(n_bases, diff_order)
    P = D.T @ D
    # Solve penalized least squares
    A = B.T @ B + lam * P
    b = B.T @ y
    coeffs = np.linalg.solve(A, b)
    return coeffs, B

# Example usage
if __name__ == '__main__':
    # Generate sample data
    x = np.linspace(0, 10, 100)
    np.random.seed(0)
    y_true = np.sin(x)
    y = y_true + np.random.normal(0, 0.2, size=len(x))

    # P-spline parameters
    n_knots = 20       # Number of knots
    degree = 3         # Degree of the spline (cubic)
    lam = 1e-2         # Smoothing parameter
    diff_order = 2     # Order of differences in penalty (second-order differences)

    # Fit P-spline
    coeffs, B = fit_p_spline(x, y, n_knots, degree, lam, diff_order)
    y_fit = B @ coeffs

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o', label='Noisy data')
    plt.plot(x, y_true, label='True function')
    plt.plot(x, y_fit, label='P-spline fit')
    plt.legend()
    plt.title('P-spline fit to data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()