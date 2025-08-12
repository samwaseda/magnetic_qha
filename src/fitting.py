import numpy as np
from scipy.spatial import cKDTree
from pyiron_base import PythonTemplateJob


def get_neigh_indices(structure, s_count):
    neigh = structure.get_neighbors(num_neighbors=100)
    tree = cKDTree(neigh.vecs[0][neigh.shells[0] <= s_count])
    dist, ind = tree.query(neigh.vecs[neigh.shells <= s_count])
    ind = ind.reshape(len(structure), -1)
    return np.array([nn[ii] for nn, ii in zip(neigh.indices, ind)])


def get_unique_args(structure, displacements, magmoms):
    struct = structure.copy()
    struct.positions += displacements
    struct.set_initial_magnetic_moments(magmoms)
    args = struct.get_symmetry(use_magmoms=True).arg_equivalent_atoms
    return np.unique(args, return_index=True)[1]


class FittingParameters(PythonTemplateJob):  # Create a custom job class
    def __init__(self, project, job_name):
        super().__init__(project, job_name)
        self.input.structure = None
        self.input.shell_mat = None
        self.input.indices = None
        self.input.positions = None
        self.input.magmoms = None
        self.input.max_exp = None
        self.input.rotations = None
        self.input.energy = None
        self.input.forces = None

    def run_static(self):
        X, Y = [], []
        for rr in self.input.rotations:
            x = self.input.positions @ rr
            unique_atoms = get_unique_args(self.input.structure, x, self.input.magmoms)
            J = (
                np.array(
                    [
                        self.input.magmoms.dot(s.dot(self.input.magmoms))
                        for s in self.input.shell_mat
                    ]
                )
                / 2
            )
            dJdm = np.array(
                [s.dot(self.input.magmoms) for s in self.input.shell_mat]
            ).T[unique_atoms]
            H = np.einsum("ij,ikl->kl", x, x[self.input.unique_vectors]) / 2
            dHdx = (
                x[self.input.unique_vectors][unique_atoms][:, None, :, :]
                * np.ones(3)[:, None, None]
            )
            K = (
                np.einsum(
                    "i,ij,ik->k",
                    self.input.magmoms,
                    x,
                    self.input.magmoms[self.input.unique_vectors],
                )
                / 2
            )
            dKdx = (
                np.einsum(
                    "i,j,ik->ijk",
                    self.input.magmoms,
                    np.ones(3),
                    self.input.magmoms[self.input.unique_vectors],
                )[unique_atoms]
                / 2
            )
            dKdm = (
                0.5
                * (
                    np.einsum(
                        "ij,ik->ik", x, self.input.magmoms[self.input.unique_vectors]
                    )
                    + np.einsum(
                        "ikj,ik->ik",
                        x[self.input.unique_vectors],
                        self.input.magmoms[self.input.unique_vectors],
                    )
                )[unique_atoms]
            )
            dLdx = np.einsum(
                "i,j,ik,ikl->ijkl",
                self.input.magmoms,
                np.ones(3),
                self.input.magmoms[self.input.unique_vectors],
                x[self.input.unique_vectors],
            )[unique_atoms]
            L = (
                np.einsum(
                    "i,ij,ik,ikl->kl",
                    self.input.magmoms,
                    x,
                    self.input.magmoms[self.input.unique_vectors],
                    x[self.input.unique_vectors],
                )
                / 2
            )
            dLdm = np.einsum(
                "ij,ik,ikl->ikl",
                x,
                self.input.magmoms[self.input.unique_vectors],
                x[self.input.unique_vectors],
            )[unique_atoms]
            A = [
                np.sum(self.input.magmoms ** (2 * (ii + 1)))
                for ii in range(self.input.max_exp)
            ]
            dAdm = np.array(
                [
                    2 * (ii + 1) * self.input.magmoms ** (2 * ii + 1)
                    for ii in range(self.input.max_exp)
                ]
            ).T
            shape = 3 * len(unique_atoms)
            E_stack = np.array(
                H.flatten().tolist()
                + J.tolist()
                + K.flatten().tolist()
                + L.flatten().tolist()
                + A
            )
            dEdx_stack = np.array(
                [
                    _dHdx.flatten().tolist()
                    + len(J.flatten()) * [0]
                    + _dKdx.flatten().tolist()
                    + _dLdx.flatten().tolist()
                    + self.input.max_exp * [0]
                    for _dHdx, _dKdx, _dLdx in zip(
                        dHdx.reshape(shape, -1),
                        dKdx.reshape(shape, -1),
                        dLdx.reshape(shape, -1),
                    )
                ]
            )
            dEdm_stack = np.array(
                [
                    len(H.flatten()) * [0]
                    + _dJdm.tolist()
                    + _dKdm.flatten().tolist()
                    + _dLdm.flatten().tolist()
                    + _dAdm.tolist()
                    for _dJdm, _dKdm, _dLdm, _dAdm in zip(dJdm, dKdm, dLdm, dAdm)
                ]
            )
            X.append(
                np.array(
                    E_stack.reshape(1, -1).tolist()
                    + dEdx_stack.tolist()
                    + dEdm_stack.tolist()
                )
            )
            Y.append(
                [self.input.energy]
                + (self.input.forces @ rr)[unique_atoms].flatten().tolist()
                + len(unique_atoms) * [0]
            )
        self.output.X = X
        self.output.Y = Y
        self.status.finished = True
        self.to_hdf()
