import numpy as np
import pandas as pd
from magnetic_qha.src.collect_data import get_job_table


def get_heisenberg_data(project, max_exp=5):
    data_dict = {}
    df = get_job_table(project)
    for cs, s_count in zip(["bcc", "fcc"], [5, 6]):
        m_lst = []
        A_lst = []
        E_lst = []
        v_lst = []
        n_lst = []
        for job_name in df.job:
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
            E_lst.extend([job["output/generic/energy_pot"][-1]] + len(m) * [0])
            m_lst.extend(np.transpose([[-0.5 * m.dot(ss.dot(m))] + (-ss.dot(m)).tolist() for ss in s_lst[:s_count]]))
            A_lst.extend(np.transpose([[np.sum(m**(2 * (ii + 1)))] + (2 * (ii + 1) * m**(2 * ii + 1)).tolist() for ii in range(max_exp)]))
            v_lst.extend([job.structure.get_volume(per_atom=True)] + len(m) * [0])
            n_lst.extend([len(job.structure)] + len(m) * [0])
        data_dict[cs] = pd.DataFrame(m_lst, columns=[f"J_{ii}" for ii in range(s_count)])
        for ii, AA in enumerate(np.transpose(A_lst)):
            data_dict[cs][f"A_{ii}"] = AA
        data_dict[cs]["n"] = n_lst
        data_dict[cs]["v"] = v_lst
        data_dict[cs]["E"] = E_lst
    return data_dict
