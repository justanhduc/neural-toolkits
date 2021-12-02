"""Init file for Pytorch-specific Gin-Config package. Adapted from Gin-config"""

try:
    from gin import *
    from neural_toolkits.gin_nnt import external_configurables
except ImportError:
    print('Please install Gin-config first via \'pip install gin-config\'')
    raise
