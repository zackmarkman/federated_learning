import sys
import socket
import pickle
import time
import threading

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


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class Server():
    def __init__(self, *args):
        self.args = args
        self.clients = set() # List of clients
        self.num_iters = 100
        self.loss = []
        self.accuracy = []
        self.global_train_size = 0
        self.client_IDs = {}
        self.num_connected = 0

    def run(self):
        # Create a new socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # created an object
            print("Server start")
            print("Server host names: ", IP, "Port: ", PORT_HOST)
            s.bind((IP, PORT_HOST))
            s.listen(5)

            # Initiate model
            global_model = MCLR()
            t_start = time.time()

            while time.time() - t_start < 30:
                # Establish connection from client
                client, addr = s.accept()
                self.clients.add(client)
                mess_recv = client.recv(65536) # clientID and train size
                clientID, client_train_size = pickle.loads(mess_recv)
                self.client_IDs[client] = clientID
                self.global_train_size += client_train_size
                self.num_connected = len(self.clients)

            for i in range(self.num_iters):
                print("Global iteration {}:".format(i))
                print("Total number of clients:",self.num_connected)

                global_model_file = open('global_model', 'w+b')
                glob_send = pickle.dump(global_model)


                for client in self.clients:
                    s.send()





