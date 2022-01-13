"""Init file for Pytorch-specific Gin-Config package. Adapted from Gin-config"""

try:
    from gin import *
    from . import external_configurables
except ImportError:
    print('Please install Gin-config first via \'pip install gin-config\'')
    raise
