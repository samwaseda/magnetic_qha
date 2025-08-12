import numpy as np
from scipy.spatial import cKDTree
from pyiron_atomistics.atomistics.structure.factory import StructureFactory


def get_unique_args(structure, displacements, magmoms):
    struct = structure.copy()
    struct.positions += displacements
    struct.set_initial_magnetic_moments(magmoms)
    args = struct.get_symmetry(use_magmoms=True).arg_equivalent_atoms
    return np.unique(args, return_index=True)[1]


def get_structure(cs):
    factory = StructureFactory()
    if cs == "bcc":
        return factory.bulk("Fe", cubic=True).repeat(4)
    else:
        return factory.bulk(
            "Fe", cubic=True, crystalstructure="fcc", a=3.45
        ).repeat(4)


def fit_parameters(
    cs, shell_mat, unique_vectors, positions, magmoms, max_exp, rotations, energy, forces
):
    structure = get_structure(cs)
    X, Y = [], []
    for rr in rotations:
        x = positions @ rr
        unique_atoms = get_unique_args(structure, x, magmoms)
        J = np.array([magmoms.dot(s.dot(magmoms)) for s in shell_mat]) / 2
        dJdm = np.array([s.dot(magmoms) for s in shell_mat]).T[unique_atoms]
        H = np.einsum("ij,ikl->kl", x, x[unique_vectors]) / 2
        dHdx = (
            x[unique_vectors][unique_atoms][:, None, :, :] * np.ones(3)[:, None, None]
        )
        K = np.einsum("i,ij,ik->k", magmoms, x, magmoms[unique_vectors]) / 2
        dKdx = (
            np.einsum(
                "i,j,ik->ijk",
                magmoms,
                np.ones(3),
                magmoms[unique_vectors],
            )[unique_atoms]
            / 2
        )
        dKdm = (
            0.5
            * (
                np.einsum("ij,ik->ik", x, magmoms[unique_vectors])
                + np.einsum(
                    "ikj,ik->ik",
                    x[unique_vectors],
                    magmoms[unique_vectors],
                )
            )[unique_atoms]
        )
        dLdx = np.einsum(
            "i,j,ik,ikl->ijkl",
            magmoms,
            np.ones(3),
            magmoms[unique_vectors],
            x[unique_vectors],
        )[unique_atoms]
        L = (
            np.einsum(
                "i,ij,ik,ikl->kl",
                magmoms,
                x,
                magmoms[unique_vectors],
                x[unique_vectors],
            )
            / 2
        )
        dLdm = np.einsum(
            "ij,ik,ikl->ikl",
            x,
            magmoms[unique_vectors],
            x[unique_vectors],
        )[unique_atoms]
        A = [np.sum(magmoms ** (2 * (ii + 1))) for ii in range(max_exp)]
        dAdm = np.array(
            [2 * (ii + 1) * magmoms ** (2 * ii + 1) for ii in range(max_exp)]
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
                + max_exp * [0]
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
            [energy]
            + (forces @ rr)[unique_atoms].flatten().tolist()
            + len(unique_atoms) * [0]
        )
    return X, Y
