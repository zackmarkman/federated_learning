import json
import os
import pickle
import sys
import socket
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


CLIENT_ID = sys.argv[1][-1]
PORT_CLIENT = int(sys.argv[2])
OPT_FLAG = int(sys.argv[3])
IP = "127.0.0.1"
PORT_SERVER = 6000

# Tunable parameters
LEARNING_RATE = 0.1
BATCH_SIZE = 20
EPOCHS = 1


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
    def __init__(self):
        # load data
        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(CLIENT_ID)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        if OPT_FLAG == 0:
            self.trainloader = DataLoader(self.train_data, self.train_samples)
            print("Batch size:",self.train_samples,'\n')
        else:
            self.trainloader = DataLoader(self.train_data, BATCH_SIZE)
            print("Batch size:", BATCH_SIZE,'\n')
        self.evalloader = DataLoader(self.train_data, self.train_samples)
        self.testloader = DataLoader(self.test_data, self.test_samples)
        self.loss = nn.NLLLoss()
        self.model = MCLR()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = LEARNING_RATE)
        self.log_file = open("client{}_log.txt".format(CLIENT_ID),'w+')

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for image, label in self.trainloader:
                self.optimizer.zero_grad()
                output = self.model(image)
                loss = self.loss(output, label)
                loss.backward()
                self.optimizer.step()

    def eval_train_loss(self):
        self.model.eval()
        train_loss = 0
        for image, label in self.evalloader:
            output = self.model(image)
            train_loss += self.loss(output, label)
        return train_loss

    def test(self):
        self.model.eval()
        test_acc = 0
        for image, label in self.testloader:
            output = self.model(image)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == label) * 1. / label.shape[0]).item()
        return test_acc


    def run(self):
        # connect to server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((IP, PORT_SERVER))
            message = (CLIENT_ID, self.train_samples)
            mess = pickle.dumps(message)
            s.sendall(mess)
            comm_round = 1
            while (comm_round < 101):
                print("I am client", CLIENT_ID)
                data_recv = s.recv(65536)
                print("Receiving new global model")
                global_model, comm_round = pickle.loads(data_recv)
                # Set parameters to global model
                self.set_parameters(global_model)

                # Evaluate training loss of global model
                train_loss = self.eval_train_loss()
                print("Training loss: {:.2f}".format(train_loss.item()))

                # Evaluate test accuracy of global model
                test_accuracy = self.test()
                print("Testing accuracy: {}%".format(int(test_accuracy * 100)))

                # Train model on local data
                print("Local training...")
                self.train(EPOCHS)

                # Send local model, train loss and test accuracy to server
                print("Sending new local model\n")
                model_send = pickle.dumps((self.model,train_loss.item(),test_accuracy))
                s.sendall(model_send)

                # Write results to log file
                self.log_file.write("Communication round {}\n".format(comm_round))
                self.log_file.write("Training loss: {}\n".format(train_loss.item()))
                self.log_file.write("Testing accuracy: {}%\n".format(test_accuracy * 100))
                self.log_file.flush()
                comm_round += 1

            self.log_file.close()


client = Client()

client.run()
