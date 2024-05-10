import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import h5py
import os
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains

import time


def gram_schmidt(A):
    Q = np.zeros_like(A, dtype=np.float64)
    for i in range(A.shape[1]):
        v = A[:, i]
        for j in range(i):
            q = Q[:, j]
            v = v - np.dot(q, v) * q
        Q[:, i] = v / np.linalg.norm(v)
    return Q


# creating the geometry domain using MATLAB
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

size = 10

mu_1_space = np.linspace(0.1, 4, size)
mu_2_space = np.linspace(0, 2, size)
L = [[None] * size for i in range(size)]
beta = np.zeros((size, size))

for i, mu_1 in enumerate(mu_1_space):
    for j, mu_2 in enumerate(mu_2_space):
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
        L[i][j] = A
        U, S, V = np.linalg.svd(A.toarray())
        beta[i, j] = np.min(S ** 2)

mid = size // 2 - 1
rb_mu = [(mu_1_space[mid], mu_2_space[mid])]
u_soln = spsolve(L[mid][mid], d)
rb_mat = np.empty((N, 0))
rb_mat = np.column_stack((rb_mat, u_soln[:, np.newaxis]))


for _ in range(1, 20):
    est_err_max = 0
    for i, mu_1 in enumerate(mu_1_space):
        for j, mu_2 in enumerate(mu_2_space):
            if (mu_1, mu_2) in rb_mu:
                continue
            A = L[i][j].toarray()
            M = np.dot(A, rb_mat)
            Mm = np.dot(M.T, M)
            Md = np.dot(M.T, d)
            if _ == 1:
                c_coef = Md / Mm[0, 0]
            else:
                c_coef = np.linalg.solve(Mm, Md)
            u_soln = np.dot(rb_mat, c_coef)

            est_err = np.linalg.norm(d - np.dot(A, u_soln)) / np.sqrt(beta[i, j])
            if est_err > est_err_max:
                A_next = L[i][j]
                mu_1_next = mu_1
                mu_2_next = mu_2
                est_err_max = est_err
    u_soln = spsolve(A_next, d)
    rb_mat = np.column_stack((rb_mat, u_soln[:, np.newaxis]))
    rb_mat = gram_schmidt(rb_mat)
    rb_mu.append((mu_1_next, mu_2_next))


np.savez('Gen_RB.npz', rb_mat=rb_mat, diff_mat=L, beta=beta, rb_mu=rb_mu)
