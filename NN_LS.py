1#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F  
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
import pandas as pd
import time
        
class Net(torch.nn.Module):  
    def __init__(self, net_sizes):
        super(Net, self).__init__()     
        self.layers = nn.ModuleList()
        for i in range(len(net_sizes) - 2):
            self.layers.append(nn.Linear(net_sizes[i], net_sizes[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(net_sizes[len(net_sizes) - 2], net_sizes[len(net_sizes) - 1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# loss function
def loss_func(y_pre, y_true):
    y_error = y_true-y_pre
    y_error = y_error.float()
    y_true = y_true.float()
    relative_loss = torch.norm(y_error, p=2, dim=1)/torch.norm(y_true, p=2, dim=1)
    return relative_loss.sum()

torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(1111)

X = np.load("Dataset_Input_mat.npy")  #(10000, 2)
Y = np.load("Dataset_Output_mat.npy")  #(10000, 20)
POD_basis = np.load("POD_basis.npy")  #(5731, 20)
truth_sol = np.load('truth_sol.npz')
Gen_RB = np.load("Gen_RB.npz") 
truth_sol_mat = truth_sol['truth_sol_mat']  #(5731, 2000)
rb_mat = Gen_RB['rb_mat'] #(5731, 20)

n_input = X.shape[1]
n_output = Y.shape[1]

n_train = 6000
n_valid = 2000
n_test = 2000
truth_sol_mat = truth_sol_mat[:, 0:n_test]
device = torch.device("cuda:0")

X = torch.from_numpy(X)
Y = torch.from_numpy(Y)
POD_basis = torch.from_numpy(POD_basis)
POD_basis = POD_basis.to(device)

X_train = X[:n_train]
Y_train = Y[:n_train]
X_valid = X[n_train:n_train + n_valid]
Y_valid = Y[n_train:n_train + n_valid]
X_test = X[n_train + n_valid:n_train + n_valid + n_test]
Y_test = Y[n_train + n_valid:n_train + n_valid + n_test]

n_depth = 2
n_neurons = 500
n_epochs = 2000
lr = 0.0001
batch_size = 100
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size = batch_size, shuffle=False)
net_sizes = [n_input] + [n_neurons]*n_depth + [n_output]   
net = Net(net_sizes)
net = net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)

valid_loss_max = float('inf')
train_loss_hist = []
valid_loss_hist = []
test_loss_hist = []
train_start_time = time.time()
for epoch in range(n_epochs):
    start_time = time.time()
    net.train()
    train_loss = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        prediction = net.forward(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss
    train_loss /= n_train

    net.eval()
    valid_loss = 0
    test_loss = 0
    with torch.no_grad():
        x = X_valid
        y = Y_valid
        x = x.to(device)
        y = y.to(device)
        prediction = net(x)
        loss = loss_func(prediction, y)
        valid_loss += loss
        valid_loss /= n_valid

        x = X_test
        y = Y_test
        x = x.to(device)
        y = y.to(device)
        prediction = net(x)
        loss = loss_func(prediction, y)
        test_loss += loss
        test_loss /= n_test

    end_time = time.time()
    epoch_time = end_time - start_time

    train_loss_hist.append(train_loss.item())
    valid_loss_hist.append(valid_loss.item())
    test_loss_hist.append(test_loss.item())
    if valid_loss < valid_loss_max:
        torch.save(net.state_dict() , 'best_net.pth')
        valid_loss_max = valid_loss
        print(f'Find better net in epoch: {epoch + 1} | train loss: {train_loss.item():.8f} | valid loss: {valid_loss.item():.8f} | test loss: {test_loss.item():.8f} | time: {epoch_time:.2f}s')
    else:
        print(f'==================>epoch: {epoch + 1} | train loss: {train_loss.item():.8f} | valid loss: {valid_loss.item():.8f} | test loss: {test_loss.item():.8f} | time: {epoch_time:.2f}s')

    if epoch < 2000:
        train_end_time = time.time()

train_time = train_end_time - train_start_time

net.load_state_dict(torch.load('best_net.pth'))
net.eval()
with torch.no_grad():
    x = X_test
    y = Y_test
    x = x.to(device)
    y = y.to(device)
    test_start_time = time.time()
    NN_prediction = net(x)
    NN_prediction = NN_prediction.t()
    NN_prediction = torch.matmul(POD_basis, NN_prediction)   #(5731, 2000)

test_end_time = time.time()
NN_test_time = test_end_time - test_start_time
POD_basis = POD_basis.cpu().numpy()
NN_prediction = NN_prediction.cpu().numpy()
NN_relative_err = np.linalg.norm(NN_prediction - truth_sol_mat, axis=0)/np.linalg.norm(truth_sol_mat, axis=0)

print(f'NN train time = {train_time:.8f}s, NN test time = {NN_test_time:.8f}s, NN relative err = {np.mean(NN_relative_err)}')
torch.cuda.empty_cache() 
########################################################

nodes_path = 'PointCloud.mat'
nodes_mat = h5py.File(nodes_path, mode='r')
nodes = nodes_mat["Nodes"]
BoundaryIndex = nodes_mat["BoundaryIndex"][0, :].astype(np.int32) - 1
BoundaryIndex.sort()
InteriorIndex = [i for i in range(nodes.shape[0]) if i not in BoundaryIndex]
InteriorIndex.sort()
N = nodes.shape[0]

n = 50
phi = 'imq'  
eps = 3.0
order = -1

d = np.zeros((N,))
d[InteriorIndex] = -10 * np.sin(8 * nodes[InteriorIndex, 0] * (nodes[InteriorIndex, 1] - 1))
d[BoundaryIndex] = 0

d = torch.from_numpy(d)
d = d.to(device)  #(5731,)
rb_mat = torch.from_numpy(rb_mat)
rb_mat = rb_mat.to(device) #(5731, 20)

LS_start_time = time.time()
LS_relative_err = np.zeros((n_test,))
for i in range(n_test):
    mu_1 = X_test[i, 0]
    mu_2 = X_test[i, 1]
    A_interior = weight_matrix(
        x=nodes[InteriorIndex],
        p=nodes,
        n=n,
        diffs=[[2, 0], [0, 2], [0, 0]],
        coeffs=[-1, -mu_1, -mu_2],
        phi=phi,
        eps=eps,
        order=order)
    A_boundary = weight_matrix(
        x=nodes[BoundaryIndex],
        p=nodes,
        n=1,
        diffs=[0, 0])
    A = expand_rows(A_interior, InteriorIndex, N)
    A += expand_rows(A_boundary, BoundaryIndex, N)
    A = A.toarray()
    A = torch.from_numpy(A)
    A = A.to(device)  #(5731, 5731)
    M = torch.matmul(A, rb_mat)  #(5731, 20)
    Mm = torch.matmul(M.t(), M)  #(20, 20)
    Md = torch.matmul(M.t(), d.unsqueeze(1))  #(20, 5731) (5731, 1)  -->  (20, 1)
    coef = torch.linalg.solve(Mm, Md) #(20, 20)*(20, 1) = (20, 1)  -->  (20, 1)
    LS_sol = torch.matmul(rb_mat, coef) #(5731, 20) (20, 1)  -->  (5731, 1)
    LS_sol = LS_sol.squeeze()
    LS_sol = LS_sol.cpu().numpy()
    LS_relative_err[i] = np.linalg.norm(LS_sol - truth_sol_mat[:,i])/np.linalg.norm(truth_sol_mat[:,i])

LS_end_time = time.time()
LS_time = LS_end_time - LS_start_time

print(f'LS time = {LS_time:.8f}s, LS relative err = {np.mean(LS_relative_err)}')