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


PORT_HOST = int(sys.argv[1])
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
        self.loss = [] # Avg loss for each iteration for plotting
        self.accuracy = [] # Avg accuracy for each iteration for plotting
        self.global_train_size = 0 # Initial total train size
        self.client_IDs = {} # Client socket object : client ID integer
        self.client_models = {} # Client socket object : client local model
        self.client_train_sizes = {} # Client socket object : client train size
        self.evaluation_log = open("evaluation_log.txt", "w+")
        self.host_socket = None

    def run(self):
        # Create a new socket
        self.host_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Server start")
        print("Server host names: ", IP, "Port: ", PORT_HOST)
        self.host_socket.bind((IP, PORT_HOST))
        self.host_socket.listen(5)
        self.host_socket.settimeout(30) # timeout

        # Initiate model
        global_model = MCLR()
        t_start = time.time()

        # initialise server and clients
        client, addr = self.host_socket.accept()
        client.settimeout(10)
        mess_recv = client.recv(65536)
        clientID, client_train_size = pickle.loads(mess_recv)
        self.clients.add(client)
        self.client_IDs[client] = clientID
        self.client_train_sizes[client] = client_train_size
        self.global_train_size += client_train_size
        while time.time() - t_start < 30:
            try:
                # Establish connection from client
                client, addr = self.host_socket.accept()
                client.settimeout(10)
                mess_recv = client.recv(65536)
                clientID, client_train_size = pickle.loads(mess_recv)
                self.clients.add(client)
                self.client_IDs[client] = clientID
                self.client_train_sizes[client] = client_train_size
                self.global_train_size += client_train_size
            except: # error raised when anything times out
                break

        background_connector = ClientConnector(self)
        background_connector.start()

        # run server
        for i in range(1, self.num_iters+1):
            client_lock.acquire()
            print("Global iteration {}:".format(i))
            print("Total number of clients:", len(self.clients))
            glob_send = pickle.dumps((global_model, i)) # byte stream
            disconnections = set()

            for client in self.clients:
                try:
                    client.send(glob_send) # send pickled model to all connected clients
                except:
                    disconnections.add(client)
            self.remove_clients(disconnections)

            client_losses = []
            client_acc = []
            for client in self.clients: # Need timeout to test disconnection
                try:
                    mess_recv = client.recv(65536) # client model
                    clientID = self.client_IDs[client]
                    print("Getting local model from client {}".format(clientID))
                    client_model,client_train_loss,client_test_acc = pickle.loads(mess_recv)
                    self.client_models[client] = copy.deepcopy(client_model) # overwrite existing client model
                    client_losses.append(client_train_loss)
                    client_acc.append(client_test_acc)
                except:
                    disconnections.add(client)
            self.remove_clients(disconnections)

            print("Aggregating new global model\n")
            global_model = self.aggregate_parameters(global_model, self.global_train_size,
                                                     self.client_models, self.client_train_sizes)
            avg_loss, avg_acc = self.evaluate(client_losses,client_acc)

            self.evaluation_log.write("Communication round {}\n".format(i))
            self.evaluation_log.write("Average training loss: {}\n".format(avg_loss))
            self.evaluation_log.write("Average testing accuracy: {}%\n".format(avg_acc*100))
            self.evaluation_log.flush()

            print("Broadcasting new global model")

            client_lock.release()
        # Generate plot here

        self.host_socket.close()
        self.evaluation_log.close()


    def aggregate_parameters(self, global_model, global_train_size, client_models, client_sizes):
        # clear gobal model
        for parameter in global_model.parameters():
            parameter.data = torch.zeros_like(parameter.data)

        # aggregate models
        if SUB_CLIENTS == 0:
            sample_clients = client_models
        else:
            # randomly select two clients to sample
            sample_clients = random.sample(client_models.keys(), 2)

        for client_socket in sample_clients:
            for global_parameter, client_parameter in zip(global_model.parameters(), client_models[client_socket].parameters()):
                global_parameter.data = global_parameter.data + client_parameter.data.clone() * client_sizes[client_socket] / global_train_size

        return global_model


    def evaluate(self, client_losses, client_accuracies):
        total_loss = 0
        total_accurancy = 0
        for i in range(len(client_losses)):
            total_loss += client_losses[i]
            total_accurancy += client_accuracies[i]
        return total_loss/len(client_losses), total_accurancy/len(client_accuracies)


    def remove_clients(self, disconnections):
        for client in disconnections:
            self.clients.discard(client)
            del self.client_IDs[client]
            self.global_train_size -= self.client_train_sizes[client]
            del self.client_train_sizes[client]
            if client in self.client_models:
                del self.client_models[client]


class ClientConnector(threading.Thread):
    def __init__(self, server):
        threading.Thread.__init__(self)
        self.server = server
    def run(self):
        while True:
            client, addr = server.host_socket.accept()
            client_lock.acquire()
            client.settimeout(10)
            mess_recv = client.recv(65536)
            clientID, client_train_size = pickle.loads(mess_recv)
            server.clients.add(client)
            server.client_IDs[client] = clientID
            server.client_train_sizes[client] = client_train_size
            server.global_train_size += client_train_size
            client_lock.release()

    def stop(self):
        self._stop.set()

client_lock = threading.Lock()
server = Server()
server.run()
