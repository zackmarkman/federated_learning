import copy
import json
import os
import pickle
import random
import sys
import socket
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.client_models = {}
        self.num_connected = 0

    def run(self):
        # Create a new socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # created a socket object
            print("Server start")
            print("Server host names: ", IP, "Port: ", PORT_HOST)
            s.bind((IP, PORT_HOST))
            s.listen(5)

            # Initiate model
            global_model = MCLR()
            t_start = time.time()

            # initialise server and clients
            while time.time() - t_start < 30:
                # Establish connection from client
                client, addr = s.accept()
                self.clients.add(client)
                mess_recv = client.recv(65536) # clientID and train size # do we need to know size of model beforehand?
                clientID, client_train_size = pickle.loads(mess_recv) # receive via byte stream as only file id and train size
                self.client_IDs[client] = clientID
                self.global_train_size += client_train_size
                self.num_connected = len(self.clients)

            # run server
            for i in range(self.num_iters):
                print("Global iteration {}:".format(i))
                print("Total number of clients:", self.num_connected)
                glob_send = pickle.dumps(global_model) # byte stream

                for client in self.clients:
                    s.send(glob_send) # send pickled model to all connected clients

                i = len(self.clients)
                while i > 0:
                    clientID = self.client_IDs[client]
                    print("Getting local model from client {}".format(clientID))
                    mess_recv = client.recv(65536) # client model # do we need to know size of model beforehand?
                    client_model = pickle.loads(mess_recv)
                    self.client_models[client] = copy.deepcopy(client_model) # overwrite existing client model
                    i -= 1

                print("Aggregating new global model")
                global_model = aggregate_parameters(global_model, self.global_train_size, self.client_models)
                print("Broadcasting new global model")

                # need additional thread to listen for new clients here
                # add them to client_models and broadcast new global model in the next iteration

    def aggregate_parameters(global_model, global_train_size, client_models):
        # clear gobal model
        for parameter in global_model.parameters():
            parameter.data = torch.zeros_like(parameter.data)

        # aggregate models
        if SUB_CLIENTS == 0:
            for client_model in client_models:
                for global_parameter, client_parameter in zip(global_model.parameters(), client_model.parameters()):
                    global_parameter.data = global_parameter.data + client_parameter.data.clone() * client_model.train_samples / global_train_size
        elif SUB_CLIENTS == 1:
            # randomly select two clients to sample
            x = random.sample([range(0, len(self.clients))], 2)
            sample_clients = [self.client_models[x[0]], self.client_models[x[1]]]
            for client_model in sample_clients:
                for global_parameter, client_parameter in zip(global_model.parameters(), client_model.parameters()):
                    global_parameter.data = global_parameter.data + client_parameter.data.clone() * client_model.train_samples / global_train_size

        return global_model
