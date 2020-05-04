try:
    from .partial_hevp import *
except:
    import sys
    print('This module requires raleigh package, please install by')
    if sys.version_info[0] == 3:
        print('pip3 install --user raleigh')
    else:
        print('pip install --user raleigh')