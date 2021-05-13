import sys
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader

IP = "127.0.0.1"
PORT_SERVER = 6000
CLIENT_ID = sys.argv[1]
PORT_CLIENT = int(sys.argv[2])
OPT_FLAG = int(sys.argv[3])

class Client():
    pass
