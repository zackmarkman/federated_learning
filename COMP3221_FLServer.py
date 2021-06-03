import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import sys
import socket
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


PORT_HOST = int(sys.argv[1])
SUB_CLIENTS = int(sys.argv[2])
IP = "127.0.0.1"

# Tunable parameters
GLOBAL_ROUNDS = 100
SUBCLIENTS_NUMBER = 5


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
        self.clients = [] # List of clients
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

        # Initiate model
        global_model = MCLR()

        # Wait for client, then initialise data from handshake
        client, addr = self.host_socket.accept()
        mess_recv = client.recv(65536)
        clientID, client_train_size = pickle.loads(mess_recv)
        self.clients.append(client)
        self.client_IDs[client] = clientID
        self.client_train_sizes[client] = client_train_size
        self.global_train_size += client_train_size

        # Wait for clients in the background
        background_connector = ClientConnector(self)
        background_connector.start()

        # Wait for 30 seconds before running
        time.sleep(30)

        # run server
        for i in range(1, GLOBAL_ROUNDS+1):
            client_lock.acquire()
            print("Global iteration {}:".format(i))
            print("Total number of clients:", len(self.clients))
            glob_send = pickle.dumps((global_model, i)) # byte stream

            disconnections = set()
            # Send global model to all clients
            for client in self.clients:
                try:
                    client.send(glob_send) # send pickled model to all connected clients
                except:
                    disconnections.add(client)
            self.remove_clients(disconnections)

            # If all clients disconnect
            if len(self.clients) == 0:
                client_lock.release()
                timestamp = time.time()
                print("All clients disconnected, waiting for new clients...")
                while time.time() - timestamp < 30:
                    if len(self.clients) > 0:
                        break
                else:
                    print("No clients connected, exiting...")
                    break
                continue

            # Receive model and stats from clients
            client_losses = []
            client_acc = []
            disconnections = set()
            for client in self.clients:
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

            # If all clients disconnect
            if len(self.clients) == 0:
                client_lock.release()
                timestamp = time.time()
                while time.time() - timestamp < 30:
                    if len(self.clients) > 0:
                        break
                else:
                    print("Exiting...")
                    break
                continue

            # Aggregate and evaluate client models
            print("Aggregating new global model")
            global_model = self.aggregate_parameters(global_model, self.global_train_size,
                                                     self.client_models, self.client_train_sizes)
            avg_loss, avg_acc = self.evaluate(client_losses,client_acc)
            self.loss.append(avg_loss)
            self.accuracy.append(avg_acc)

            self.evaluation_log.write("Communication round {}\n".format(i))
            self.evaluation_log.write("Average training loss: {}\n".format(avg_loss))
            self.evaluation_log.write("Average testing accuracy: {}%\n".format(avg_acc*100))
            self.evaluation_log.flush()

            print("Broadcasting new global model\n")

            client_lock.release()
            time.sleep(0.1)

        self.host_socket.close()
        self.evaluation_log.close()

        lossfile = np.array(self.loss)
        accfile = np.array(self.accuracy)

        # [loss/acc]_subclients_batch
        np.savetxt("loss_5_GD.csv", lossfile)
        np.savetxt("accuracy_5_GD.csv", accfile)

        # Generate plot
        plt.figure(figsize=(8, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss, label="FedAvg", linewidth=1)
        plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
        plt.ylabel('Training Loss')
        plt.xlabel('Global rounds')

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy, label="FedAvg", linewidth=1)
        plt.legend(loc='lower right', prop={'size': 12}, ncol=2)
        plt.yticks(np.arange(0.1, 1.01, 0.1))
        plt.ylabel('Testing Acc')
        plt.xlabel('Global rounds')
        plt.show()


    def aggregate_parameters(self, global_model, global_train_size, client_models, client_sizes):
        # clear gobal model
        for parameter in global_model.parameters():
            parameter.data = torch.zeros_like(parameter.data)

        # aggregate models
        if SUB_CLIENTS == 0:
            sample_clients = client_models
        else:
            # randomly select sub clients to sample
            sample_clients = random.sample(client_models.keys(), min(len(client_models), SUBCLIENTS_NUMBER))

        for client_socket in sample_clients:
            for global_parameter, client_parameter in zip(global_model.parameters(), client_models[client_socket].parameters()):
                global_parameter.data += client_parameter.data.clone() * client_sizes[client_socket] / global_train_size

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
            self.clients.remove(client)
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
            try:
                client, addr = server.host_socket.accept()
                client_lock.acquire()
                mess_recv = client.recv(65536)
                clientID, client_train_size = pickle.loads(mess_recv)
                server.clients.append(client)
                server.client_IDs[client] = clientID
                server.client_train_sizes[client] = client_train_size
                server.global_train_size += client_train_size
                client_lock.release()
            except:
                break


client_lock = threading.Lock()
server = Server()
server.run()
