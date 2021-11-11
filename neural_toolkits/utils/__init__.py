import logging

log_formatter = logging.Formatter('%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

from .tensor_utils import *
from .layer_utils import *
from .numpy_utils import *
from .activation_utils import *
from .cv_utils import *
from .misc_utils import *
from .data_utils import *
from .model_utils import *
from .trainer_utils import *

import torch

cuda_available = torch.cuda.is_available()

del torch
del logging
