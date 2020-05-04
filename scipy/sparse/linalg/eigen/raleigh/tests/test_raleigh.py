# Copyright 2019 United Kingdom Research and Innovation 
# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

""" raleigh interfaces tests.

Requires:

- raleigh package: pip/pip3 install --user raleigh
- MKL 10.3 or later: pip/pip3 install --user mkl
"""

import numpy
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import scipy.linalg as sla
import scipy.sparse as scs

try:
    from scipy.sparse.linalg import partial_hevp
except ImportError:
    import sys
    print('This module requires raleigh package, please install by')
    if sys.version_info[0] == 3:
        print('pip3 install --user raleigh')
    else:
        print('pip install --user raleigh')
    exit()


''' Test utilities
'''

def _lap1d(n, a):
    h = a/(n + 1)
    d = numpy.ones((n,))/(h*h)
    return scs.spdiags([-d, 2*d, -d], [-1, 0, 1], n, n, format='csr')


def _lap2d(nx, ny, ax, ay):
    L = _lap1d(nx, ax)
    Ly = _lap1d(ny, ay)
    return scs.csr_matrix(scs.kron(scs.eye(ny), L) + scs.kron(Ly, scs.eye(nx)))


def _lap3d(nx, ny, nz, ax, ay, az):
    L = _lap2d(nx, ny, ax, ay)
    Lz = _lap1d(nz, az)
    return scs.csr_matrix(scs.kron(scs.eye(nz), L) + scs.kron(Lz, scs.eye(nx*ny)))


''' Tests
'''


def test_laplace_d():
    matrix = _lap3d(10, 10, 10, 1.0, 1.0, 1.0)
    sigma = 0
    nev = 10
    tol = 0
    expected = numpy.array([29.40810116, 58.02204583, 58.02204583, 58.02204583,
    86.6359905, 86.6359905, 86.6359905, 103.12910316, 103.12910316, 103.12910316])
    vals, vecs, status = partial_hevp(matrix, sigma=sigma, which=nev, tol=tol, verb=-1)
    msg = 'partial_hevp execution status: %d' % status
    assert_equal(status, 0, err_msg=msg)
    print('converged eigenvalues are:')
    print(vals)
    assert_allclose(vals[:nev], expected)


def test_laplace_z():
    matrix = _lap3d(10, 10, 10, 1.0, 1.0, 1.0).astype(numpy.complex128)
    sigma = 0
    nev = 10
    tol = 0
    expected = numpy.array([29.40810116, 58.02204583, 58.02204583, 58.02204583,
    86.6359905, 86.6359905, 86.6359905, 103.12910316, 103.12910316, 103.12910316])
    vals, vecs, status = partial_hevp(matrix, sigma=sigma, which=nev, tol=tol, verb=-1)
    msg = 'partial_hevp execution status: %d' % status
    assert_equal(status, 0, err_msg=msg)
    print('converged eigenvalues are:')
    print(vals)
    assert_allclose(vals[:nev], expected)


def test_laplace_s():
    matrix = _lap3d(10, 10, 10, 1.0, 1.0, 1.0).astype(numpy.float32)
    sigma = numpy.float32(0)
    nev = 10
    tol = 0
    expected = numpy.array([29.4080254, 58.02197692, 58.02198319, 58.022002,
    86.63592455, 86.63593154, 86.63593853, 103.12904046, 103.12904046, 103.12904046])
    vals, vecs, status = partial_hevp(matrix, sigma=sigma, which=nev, tol=tol, verb=-1)
    msg = 'partial_hevp execution status: %d' % status
    assert_equal(status, 0, err_msg=msg)
    print('converged eigenvalues are:')
    print(vals)
    assert_allclose(vals[:nev], expected, rtol=1e-5)


def test_laplace_c():
    matrix = _lap3d(10, 10, 10, 1.0, 1.0, 1.0).astype(numpy.complex64)
    sigma = numpy.float32(0)
    nev = 10
    tol = 0
    expected = numpy.array([29.40810116, 58.02204583, 58.02204583, 58.02204583,
    86.6359905, 86.6359905, 86.6359905, 103.12910316, 103.12910316, 103.12910316])
    vals, vecs, status = partial_hevp(matrix, sigma=sigma, which=nev, tol=tol, verb=-1)
    msg = 'partial_hevp execution status: %d' % status
    assert_equal(status, 0, err_msg=msg)
    print('converged eigenvalues are:')
    print(vals)
    assert_allclose(vals[:nev], expected, rtol=1e-5)


def test_laplace_low_acc():
    matrix = _lap3d(10, 10, 10, 1.0, 1.0, 1.0).astype(numpy.float32)
    sigma = numpy.float32(0)
    nev = 10
    tol = 1e-3
    expected = numpy.array([29.4080254, 58.02197692, 58.02198319, 58.022002,
    86.63592455, 86.63593154, 86.63593853, 103.12904046, 103.12904046, 103.12904046])
    vals, vecs, status = partial_hevp(matrix, sigma=sigma, which=nev, tol=tol, verb=-1)
    msg = 'partial_hevp execution status: %d' % status
    assert_equal(status, 0, err_msg=msg)
    print('converged eigenvalues are:')
    print(vals)
    assert_allclose(vals[:nev], expected, rtol=1e-5)


if __name__ == "__main__":
    test_laplace_d()
    test_laplace_z()
    test_laplace_s()
    test_laplace_c()
    test_laplace_low_acc()
