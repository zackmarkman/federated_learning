import sys
import socket

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
from torch.utils.data import DataLoader

PORT_HOST = sys.argv[1]
SUB_CLIENTS = int(sys.argv[2])
IP = "127.0.0.1"

class Server():
    pass