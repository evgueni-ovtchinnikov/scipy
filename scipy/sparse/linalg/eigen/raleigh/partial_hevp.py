# Author: Evgueni Ovtchinnikov (evgueni.ovtchinnikov@stfc.ac.uk)

"""Partial eigenvalue problem solver for a sparse symmetric/Hermitian matrix.

--------------------------------------------------------------------------------
Requires MKL 10.3 or later and raleigh (see installation instructions below).
--------------------------------------------------------------------------------
"""

import numpy
import time

try:
    from raleigh.interfaces.partial_hevp import partial_hevp as phevp
except ImportError:
    import sys
    print('This module requires mkl and raleigh packages, please install by')
    if sys.version_info[0] == 3:
        print('pip3 install --user mkl')
        print('pip3 install --user raleigh')
    else:
        print('pip install --user mkl')
        print('pip install --user raleigh')
    raise ImportError()


def partial_hevp(A, B=None, T=None, buckling=False, sigma=0, which=6, tol=1e-4,\
                 verb=0):
    '''Computes several eigenpairs of sparse real symmetric/Hermitian eigenvalue
    problems using either shift-invert or preconditioning technique.

    Parameters
    ----------
    A : scipy's sparse matrix
        The (stiffness) matrix of the problem. Must be positive definite if
        buckling is True.
    B : scipy's sparse matrix
        In buckling case (buckling is True), stress stiffness matrix, otherwise
        mass matrix, which must be positive definite. If None, eigenvalues and
        eigenvectors of A are computed, otherwise those of the generalized
        problem A x = lambda B x, or, if buckling is True, A x = -alpha B x.
    T : a Python object
        If T is not None, then A must be positive definite.
        Preconditioner (roughly, approximate inverse of A). Must have method
        apply(x, y) that, for a given equally shaped 2D ndarrays x and y with
        the second dimension equal to the problem size, applies preconditioning
        to rows of x and places the results into respective rows of y.
        The method apply(x, y) must act as a self-adjoint positive definite 
        linear operator, i.e. for any x, the matrix numpy.dot(x, y), where y is
        computed by apply(x, y), must be real symmetric/Hermitian and positive
        definite.
    buckling : Boolean
        Flag for buckling mode. Ignored if T is not None.
    sigma : float
        Shift inside the spectrum for the sake of faster convergence. Must be
        negative if buckling is True. Ignored if T is not None.
    which : an integer or tuple of two integers
        Specifies which eigenvalues are wanted.
        Integer: if T is not None or buckling is True, then it is
        the number of wanted smallest eigenvalues, otherwise the number of
        wanted eigenvalues nearest to sigma.
        Tuple (k, l): k nearest eigenvalues left from sigma and l nearest
        eigenvalues right from sigma are wanted.
    tol : float
        Eigenvector error tolerance.
    verb : integer
        Verbosity level.
        < 0 : nothing printed
          0 : error and warning messages printed
          1 : + number of iteration and converged eigenvalues printed
          2 : + current eigenvalue iterates, residuals and error estimates
              printed

    Returns
    -------
    lmd : one-dimensional numpy array
        Eigenvalues in ascending order.
    x : two-dimensional numpy array
        The matrix of corresponding eigenvectors as columns.
    status : int
        Execution status
        0 : success
        1 : maximal number of iterations exceeded
        2 : no search directions left (bad problem data or preconditioner)
        3 : some of the requested left eigenvalues may not exist
        4 : some of the requested right eigenvalues may not exist
       <0 : fatal error, error message printed if verb is non-negative
    '''

    return phevp(A, B=B, T=T, buckling=buckling, \
        sigma=sigma, which=which, tol=tol, verb=verb)