import numpy as np
from collections import defaultdict
from pyiron_atomistics.atomistics.structure.factory import StructureFactory


def get_unique_args(structure, displacements, magmoms):
    struct = structure.copy()
    struct.positions += displacements
    struct.set_initial_magnetic_moments(magmoms)
    args = struct.get_symmetry(use_magmoms=True).arg_equivalent_atoms
    return np.unique(args, return_index=True)[1]


def get_structure(cs, n_repeat=4):
    factory = StructureFactory()
    if cs == "bcc":
        return factory.bulk("Fe", cubic=True, a=2).repeat(n_repeat)
    else:
        return factory.bulk("Fe", cubic=True, crystalstructure="fcc", a=2).repeat(n_repeat)


def get_neigh_indices(structure, s_count):
    neigh = structure.get_neighbors(num_neighbors=100)
    vecs = neigh.vecs[neigh.shells <= s_count]
    vecs, indices = np.unique(
        np.round(vecs).astype(int).reshape(-1, 3), axis=0, return_inverse=True
    )
    return vecs, indices.reshape(len(structure), -1)


def get_shell_matrix(structure, s_count):
    neigh = structure.get_neighbors(num_neighbors=100)
    return neigh.get_shell_matrix()[:s_count]


def vec_to_str(vector):
    return "_".join([str(vec) for vec in vector]).replace("-", "m")


def get_all_keys(vecs, s_counts, max_exp):
    keys = [f"J_{ii}" for ii in range(s_counts)]
    for v in vecs:
        for key_1 in ["x", "y", "z"]:
            for key_2 in ["x", "y", "z"]:
                keys.append(f"H_{vec_to_str(v)}_{key_1}_{key_2}")
    for v in vecs:
        for key in ["x", "y", "z"]:
            keys.append(f"K_{vec_to_str(v)}_{key}")
    for v in vecs:
        for key_1 in ["x", "y", "z"]:
            for key_2 in ["x", "y", "z"]:
                keys.append(f"L_{vec_to_str(v)}_{key_1}_{key_2}")
    for ii in range(max_exp):
        keys.append(f"A_{2 * (ii + 1)}")
    keys.append("y")
    return keys


def fit_parameters(positions, magmoms, max_exp, energy, forces):
    if len(positions) == 128:
        s_count = 5
        cs = "bcc"
    else:
        s_count = 6
        cs = "fcc"
    structure = get_structure(cs)
    shell_mat = get_shell_matrix(structure, s_count)
    unique_vecs, unique_vec_indices = get_neigh_indices(structure, s_count)
    symmetry = structure.get_symmetry()
    rotations = np.unique(symmetry.rotations, axis=0)
    if np.isclose(np.linalg.norm(positions), 0):
        rotations = [np.eye(3)]
    X = []
    tags = []
    for rr in rotations:
        x = positions @ rr
        unique_atoms = get_unique_args(structure, x, magmoms)
        J = np.array([magmoms.dot(s.dot(magmoms)) for s in shell_mat]) / 2
        dJdm = np.array([s.dot(magmoms) for s in shell_mat]).T[unique_atoms]
        H = np.einsum("ij,ikl->kjl", x, x[unique_vec_indices]) / 2
        dHdx = np.einsum(
            "ikm,jl->ijklm", x[unique_vec_indices][unique_atoms], np.eye(3)
        )
        K = np.einsum("i,ij,ik->kj", magmoms, x, magmoms[unique_vec_indices]) / 2
        dKdx = (
            np.einsum(
                "i,jl,ik->ijkl",
                magmoms,
                np.eye(3),
                magmoms[unique_vec_indices],
            )[unique_atoms]
            / 2
        )
        dKdm = (
            0.5
            * (
                np.einsum("ij,ik->ikj", x, magmoms[unique_vec_indices])
                + np.einsum(
                    "ikj,ik->ikj",
                    x[unique_vec_indices],
                    magmoms[unique_vec_indices],
                )
            )[unique_atoms]
        )
        L = (
            np.einsum(
                "i,ij,ik,ikl->kjl",
                magmoms,
                x,
                magmoms[unique_vec_indices],
                x[unique_vec_indices],
            )
            / 2
        )
        dLdx = np.einsum(
            "i,jm,ik,ikl->ijkml",
            magmoms,
            np.eye(3),
            magmoms[unique_vec_indices],
            x[unique_vec_indices],
        )[unique_atoms]
        dLdm = np.einsum(
            "ij,ik,ikl->ikjl",
            x,
            magmoms[unique_vec_indices],
            x[unique_vec_indices],
        )[unique_atoms]
        A = [np.sum(magmoms ** (2 * (ii + 1))) for ii in range(max_exp)]
        dAdm = np.array(
            [2 * (ii + 1) * magmoms ** (2 * ii + 1) for ii in range(max_exp)]
        ).T
        shape = 3 * len(unique_atoms)
        E_stack = np.array(
            J.tolist()
            + H.flatten().tolist()
            + K.flatten().tolist()
            + L.flatten().tolist()
            + A
        )
        dEdx_stack = np.array(
            [
                len(J.flatten()) * [0]
                + _dHdx.flatten().tolist()
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
                _dJdm.tolist()
                + len(H.flatten()) * [0]
                + _dKdm.flatten().tolist()
                + _dLdm.flatten().tolist()
                + _dAdm.tolist()
                for _dJdm, _dKdm, _dLdm, _dAdm in zip(dJdm, dKdm, dLdm, dAdm)
            ]
        )
        X.extend(
            np.append(
                E_stack.reshape(1, -1).tolist()
                + dEdx_stack.tolist()
                + dEdm_stack.tolist(),
                np.reshape(
                    [energy]
                    + (forces @ rr)[unique_atoms].flatten().tolist()
                    + len(unique_atoms) * [0],
                    (-1, 1),
                ),
                axis=-1,
            )
        )
        tags.extend(["E"] + 3 * len(unique_atoms) * ["f"] + len(unique_atoms) * ["nu"])
    indices = np.unique(np.round(X, decimals=8), axis=0, return_index=True)[1]
    return (
        np.asarray(X)[indices],
        get_all_keys(unique_vecs, s_count, max_exp),
        np.asarray(tags)[indices].tolist(),
    )


def combine_fits(project, job_list):
    data_dict = {"bcc": defaultdict(list), "fcc": defaultdict(list)}
    for job in project.iter_jobs(job="fit_parameters*", convert_to_object=False):
        volume = round(job.content["user/volume"], ndigits=8)
        cs = job.content["input"]["data"]["cs"]
        X, Y = job.content["storage/output/result"]
        Y = [yyy for yy in Y for yyy in yy]
        X = [xxx for xx in X for xxx in xx]
        data = np.append(np.reshape(Y, (-1, 1)), X, axis=-1)
        for xx in data:
            if (
                len(data_dict[cs][volume]) > 0
                and np.isclose(
                    np.abs(
                        np.einsum("x,nx->n", xx, data_dict[cs][volume])
                        / np.linalg.norm(xx)
                        / np.linalg.norm(data_dict[cs][volume], axis=-1)
                    ),
                    1.0,
                ).any()
            ):
                continue
            data_dict[cs][volume].append(xx)
    return data_dict
