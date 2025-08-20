import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from collections import defaultdict
from tqdm.auto import tqdm
from magnetic_qha.src.collect_data import get_job_table
from magnetic_qha.src.fitting import get_structure


def regularize_structure(structure):
    struct = structure.copy()
    struct.positions[0] *= 0
    cell = np.eye(3) * (struct.analyse.get_layers().max(axis=0) + 1)
    struct.set_cell(cell, scale_atoms=True)
    struct.positions = np.round(struct.positions).astype(int)
    return struct


def get_heisenberg_data(project, max_exp=5):
    data_dict = {}
    df = get_job_table(project)
    for cs, s_count in zip(["bcc", "fcc"], [5, 6]):
        m_lst = []
        A_lst = []
        E_lst = []
        v_lst = []
        n_lst = []
        for job_name in tqdm(df.job):
            if not job_name.startswith(cs):
                continue
            job = project.inspect(job_name)
            job.structure = job["input/structure"].to_object()
            if not job.structure.analyse.pyscal_voronoi_volume().ptp() < 1.0e-4:
                continue
            s_lst = job.structure.get_neighbors(num_neighbors=100).get_shell_matrix()[:s_count]
            try:
                m = job["output/generic/dft"]["atom_spins"][-1]
            except KeyError:
                m = np.zeros(len(job.structure))
            try:
                nu = job.content["output/generic/dft"]["magnetic_forces"][-1]
            except (KeyError, IndexError):
                nu = np.zeros(len(m))
            E_lst.extend([job["output/generic/energy_pot"][-1]] + nu.tolist())
            m_lst.extend(np.transpose([[-0.5 * m.dot(ss.dot(m))] + (-ss.dot(m)).tolist() for ss in s_lst[:s_count]]))
            A_lst.extend(np.transpose([[np.sum(m**(2 * (ii + 1)))] + (2 * (ii + 1) * m**(2 * ii + 1)).tolist() for ii in range(max_exp)]))
            v_lst.extend((len(m) + 1) * [job.structure.get_volume(per_atom=True)])
            n_lst.extend([len(job.structure)] + len(m) * [0])
        data_dict[cs] = pd.DataFrame(m_lst, columns=[f"J_{ii}" for ii in range(s_count)])
        for ii, AA in enumerate(np.transpose(A_lst)):
            data_dict[cs][f"A_{ii}"] = AA
        data_dict[cs]["n"] = n_lst
        data_dict[cs]["v"] = v_lst
        data_dict[cs]["E"] = E_lst
    return data_dict


def get_heisenberg_forces(project):
    result_df = {}
    for cs in ["bcc", "fcc"]:
        for job in project.iter_jobs(job=f"{cs}*", convert_to_object=False):
            if job.status not in ["finished", "not_converged"]:
                continue
            struct = job.content["input/structure"].to_object()
            if not np.isclose(struct.positions[0, 0], 0):
                continue
            structure = regularize_structure(struct)
            unique_ids = np.unique(structure.get_symmetry(use_magmoms=True).arg_equivalent_atoms)
            data_dict["volume"].extend(len(unique_ids) * [struct.get_volume(per_atom=True)])
            m = job.content["output/generic/dft/atom_spins"][-1]
            neigh = structure.get_neighbors(num_neighbors=58)
            for vv in neigh.vecs[0].astype(int):
                for ii, jj in zip(*np.where(np.all(neigh.vecs[unique_ids] == vv, axis=-1))):
                    data_dict[str(vv)].append(m[neigh.indices[ii][jj]] * m[ii])
            for ii, xyz in enumerate(["x", "y", "z"]):
                for unique_id in unique_ids:
                    data_dict[f"f_{xyz}"].append(job.content["output/generic/forces"][-1][unique_id][ii])
        result_df = {cs: pd.DataFrame(data_dict)}
    return result_df 

 
def get_heisenberg_displacement(project):

    result_df = {}
    for cs in ["bcc", "fcc"]:
        structure = get_structure(cs=cs, n_repeat=4)
        if cs == "bcc":
            ref_vecs = np.round(structure.get_neighbors(num_neighbors=58).vecs[0]).astype(int)
        else:
            ref_vecs = np.round(structure.get_neighbors(num_neighbors=86).vecs[0]).astype(int)

        tree = cKDTree(ref_vecs)
        # keys = ["_".join(str(key).replace("-", "m") for key in vec) for vec in ref_vecs]
        keys = [str(vec) for vec in ref_vecs]
        structure.positions[0, 0] += 0.01

        symmetry = structure.get_symmetry()
        rotations = symmetry.rotations[np.sort(np.unique(symmetry.rotations, return_index=True, axis=0)[1])]

        output_data = defaultdict(list)
        for job in project.iter_jobs(job=f"{cs}*", convert_to_object=False):
            if job.status == "aborted":
                continue
            structure = job.content["input/structure"].to_object()
            if np.isclose(structure.positions[0], [0, 0, 0]).all():
                continue
            structure.positions[0] = np.zeros(3)
            v = structure.get_volume(per_atom=True)
            if cs == "fcc":
                neigh = structure.get_neighbors(num_neighbors=86)
                a_0 = (4 * v)**(1 / 3)
            else:
                neigh = structure.get_neighbors(num_neighbors=58)
                a_0 = (2 * v)**(1 / 3)
            vecs = np.round(neigh.vecs[0] / a_0 * 2).astype(int)
            indices = np.unique(np.einsum("nx,mxy->mny", vecs, rotations).reshape(-1, 3), axis=0, return_inverse=True)[1]
            try:
                m = job["output/generic/dft/atom_spins"][-1]
            except TypeError:
                m = np.zeros(len(structure))
            prod = m[neigh.indices[0]] * m[0]
            indices = np.argsort(tree.query(np.einsum("nx,mxy->mny", vecs, rotations).reshape(-1, 3))[1].reshape(len(rotations), len(vecs)))
            forces = np.einsum("i,nij->jn", job.content["output/generic/forces"][-1][0], rotations)
            for tag, ff in zip(["f_x", "f_y", "f_z"], forces):
                output_data[tag].extend(ff)
            for key, mm in zip(keys, prod[indices].T):
                output_data[key].extend(mm)
            output_data["volume"].extend(len(mm) * [v])

        df = pd.DataFrame(output_data)
        df.volume = np.round(df.volume, decimals=6)
        df = df[~df.duplicated()]
        result_df[cs] = df
    return result_df
