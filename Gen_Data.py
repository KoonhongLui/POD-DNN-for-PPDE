import h5py
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import h5py

from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains

# creating the geometry domain using MATLAB
nodes_path = 'PointCloud.mat'
nodes_mat = h5py.File(nodes_path, mode='r')
nodes = nodes_mat["Nodes"]
BoundaryIndex = nodes_mat["BoundaryIndex"][0, :].astype(np.int32) - 1
BoundaryIndex.sort()
InteriorIndex = [i for i in range(nodes.shape[0]) if i not in BoundaryIndex]
InteriorIndex.sort()
N = nodes.shape[0]

n = 50  # stencil size. Increase this will generally improve accuracy

phi = 'imq'  # radial basis function used to compute the weights. Odd
# order polyharmonic splines (e.g., phs3) have always performed
# well for me and they do not require the user to tune a shape
# parameter. Use higher order polyharmonic splines for higher
# order PDEs.

eps = 3.0

order = -1  # Order of the added polynomials. This should be at least as
# large as the order of the PDE being solved (2 in this case). Larger
# values may improve accuracy

# create "right hand side" vector
d = np.zeros((N,))
d[InteriorIndex] = -10 * np.sin(8 * nodes[InteriorIndex, 0] * (nodes[InteriorIndex, 1] - 1))
d[BoundaryIndex] = 0

Snapshot_mat = np.empty((N, 0))

for mu_1 in np.linspace(0.1, 4, 10):
    for mu_2 in np.linspace(0, 2, 10):
        # create the components for the "left hand side" matrix.
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
        # Expand and add the components together
        A = expand_rows(A_interior, InteriorIndex, N)
        A += expand_rows(A_boundary, BoundaryIndex, N)

        # find the solution at the nodes
        u_soln = spsolve(A, d)
        Snapshot_mat = np.column_stack((Snapshot_mat, u_soln[:, np.newaxis]))

# POD generates the reduced basis

U, S, V = np.linalg.svd(Snapshot_mat)

eps_POD = 1e-6
total_energy = (S**2).sum()
energy = 0
rank = 0

while energy/total_energy < 1 - eps_POD**2 and rank < len(S):
    rank += 1
    energy += S[rank - 1]**2

POD_basis = U[:, 0:rank]

np.save('POD_basis.npy', POD_basis)
np.save('Snapshot_mat.npy', Snapshot_mat)

# Generates data set
Proj_mat = np.transpose(POD_basis)
Input_mat = np.empty((0, 2))
Output_mat = np.empty((0, rank))

for mu_1 in np.linspace(0.1, 4, 100):
    for mu_2 in np.linspace(0, 2, 100):
        input_tup = np.array([mu_1, mu_2])
        # create the components for the "left hand side" matrix.
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
        # Expand and add the components together
        A = expand_rows(A_interior, InteriorIndex, N)
        A += expand_rows(A_boundary, BoundaryIndex, N)

        # find the solution at the nodes
        u_soln = spsolve(A, d)
        proj_u_coef = np.dot(Proj_mat, u_soln)

        Input_mat = np.row_stack((Input_mat, input_tup[np.newaxis, :]))
        Output_mat = np.row_stack((Output_mat, proj_u_coef[np.newaxis, :]))

shuffle_indices = np.random.permutation(Input_mat.shape[0])
Input_mat = Input_mat[shuffle_indices]
Output_mat = Output_mat[shuffle_indices]

np.save('Dataset_Input_mat.npy', Input_mat)
np.save('Dataset_Output_mat.npy', Output_mat)