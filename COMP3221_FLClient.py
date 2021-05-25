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

CLIENT_ID = sys.argv[1][-1]
PORT_CLIENT = int(sys.argv[2])
OPT_FLAG = int(sys.argv[3])
IP = "127.0.0.1"
PORT_SERVER = 6000

# Tuneable parameters
learning_rate = 0.01
batch_size = 20


def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0']['y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        output = F.log_softmax(x, dim=1)
        return output


class Client():
    def __init__(self, client_id, learning_rate, batch_size):
        # load data
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(
            client_id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.trainloader = DataLoader(self.train_data, batch_size)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.loss = nn.NLLLoss()
        self.id = client_id

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        LOSS = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            #print(str(self.id) + ", Accuracy of client ", self.id, " is: ", test_acc)
        return test_acc

    def run(self):
        # connect to server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((IP, PORT_SERVER))
            message = (CLIENT_ID, self.train_samples)
            mess = pickle.dumps(message)
            s.sendall(mess)
            while (True):
                print("I am client", self.id)
                print("Receiving new global model")
                data_recv = s.recv(65536)
                global_model = pickle.loads(data_recv)
                self.model = copy.deepcopy(global_model)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
                train_loss = self.train(2)
                print("Training loss: {:.2f}".format(train_loss.item()))
                test_accuracy = self.test()
                print("Testing accuracy: {}%".format(int(test_accuracy*100)))
                print("Local training...")
                model_send = pickle.dumps(self.model)
                #s.sendall(model_send)
                print("Sending new local model")


client = Client(CLIENT_ID,learning_rate,batch_size)
client.run()