import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree


def get_job_table(pr):
    df = pr.job_table()
    df = df[
        (df.status != "running")
        & (df.status != "submitted")
        & df.chemicalformula.str.isalnum()
    ]
    df["job_stem"] = ["_".join(job_name.split("_")[:2]) for job_name in df.job]
    return df[~df.job_stem.duplicated(keep="last")]


def _repeat(repeat, arr):
    return np.array(repeat.prod() * arr.tolist())


def collect_data(project):
    data_dict = {"fcc": defaultdict(list), "bcc": defaultdict(list)}
    df = get_job_table(project)
    for job_name in df.job:
        job = project.inspect(job_name)
        structure = job["input/structure"].to_object()
        repeat = 8 // (
            structure.analyse.get_layers(distance_threshold=0.1).max(axis=0) + 1
        )
        E = job["output/generic/energy_pot"][-1] * repeat.prod()
        try:
            m = job["output/generic/dft"]["atom_spins"][-1]
        except KeyError:
            m = np.zeros(len(structure))
        try:
            f = job["output/generic/forces"][-1]
            cs = job_name.split("_")[0]
            data_dict[cs]["positions"].append(structure.repeat(repeat).positions)
            data_dict[cs]["cell"].append(structure.repeat(repeat).cell)
            data_dict[cs]["energy"].append(E)
            data_dict[cs]["magmoms"].append(_repeat(repeat, m))
            data_dict[cs]["forces"].append(_repeat(repeat, f))
        except TypeError:
            pass
    for cs in ["fcc", "bcc"]:
        for key, value in data_dict[cs].items():
            data_dict[cs][key] = np.array(value)

    for cs in ["fcc", "bcc"]:
        x = np.einsum(
            "nji,nmj->nmi",
            np.linalg.inv(data_dict[cs]["cell"]),
            data_dict[cs]["positions"],
        )
        tree = cKDTree(x[0])
        indices = np.array([tree.query(xx)[1] for xx in x])
        data_dict[cs]["positions"] = np.array(
            [xx[np.argsort(ii)] for xx, ii in zip(x, indices)]
        )
        data_dict[cs]["forces"] = np.array(
            [f[np.argsort(ii)] for f, ii in zip(data_dict[cs]["forces"], indices)]
        )
        data_dict[cs]["magmoms"] = np.array(
            [f[np.argsort(ii)] for f, ii in zip(data_dict[cs]["magmoms"], indices)]
        )
    return data_dict


def combine_fits(project):
    x_list, y_list = [], []
    for job in project.iter_jobs(job="fit_parameters*", convert_to_object=False):
        X, Y = job.content["storage/output/result"]
        Y = [yyy for yy in Y for yyy in yy]
        X = [xxx for xx in X for xxx in xx]
        for xx, yy in zip(X, Y):
            if (
                len(x_list) > 0
                and np.linalg.norm(xx) == 0
                and np.linalg.norm(x_list, axis=-1).min() == 0
            ):
                continue
            if (
                len(x_list) > 0
                and np.isclose(
                    np.abs(
                        np.einsum("x,nx->n", xx, x_list)
                        / np.linalg.norm(xx)
                        / np.linalg.norm(x_list, axis=-1)
                    ),
                    1.0,
                ).any()
            ):
                continue
            x_list.append(xx)
            y_list.append(yy)
    return x_list, y_list
