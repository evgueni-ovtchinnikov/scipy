"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg.eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""
from .arpack import *
from .lobpcg import *
try:
    from .raleigh import *
except:
    import sys
    print('This module requires raleigh package, please install by')
    if sys.version_info[0] == 3:
        print('pip3 install --user raleigh')
    else:
        print('pip install --user raleigh')

__all__ = [s for s in dir() if not s.startswith('_')]

from scipy._lib._testutils import PytestTester
test = PytestTester(__name__)
del PytestTester
