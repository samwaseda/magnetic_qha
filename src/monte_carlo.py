import numpy as np
from mamonca import MC
from magnetic_qha.src.fitting import get_structure


def run_simple_mc(temperature, n_steps, cs, data_dict, metadynamics=False, n_repeat=20):
    structure = get_structure(cs, n_repeat=n_repeat)
    mc = MC(len(structure))
    mc.set_magnetic_moments(
        structure.get_initial_magnetic_moments()[:, None] * [1, 0, 0]
    )
    if cs == "bcc":
        neigh = structure.get_neighbors(num_neighbors=58)
    else:
        neigh = structure.get_neighbors(num_neighbors=86)
    for ii, mat in enumerate(neigh.get_shell_matrix()):
        mc.set_heisenberg_coeff(data_dict[f"J_{ii}"] * mat)
    for ii in range(10):
        if f"A_{ii}" not in data_dict:
            break
        mc.set_landau_coeff(data_dict[f"A_{ii}"], 2 * (ii + 1))
    if metadynamics:
        mc.set_metadynamics(max_range=3)
    mc.run(temperature=temperature, number_of_iterations=n_steps)
    if metadynamics:
        return mc.get_magnetic_moments(), mc.get_magnetization(), mc.get_metadynamics_free_energy()
    else:
        return mc.get_magnetic_moments(), mc.get_magnetization()


def run_thermodynamic_integration(
    temperature, n_steps, data_dict_bcc, data_dict_fcc, lambda_, n_repeat=20
):
    fcc = get_structure("fcc", n_repeat=n_repeat)
    bcc = fcc.copy()
    cell = bcc.cell.copy()
    cell[-1, -1] *= 1 / np.sqrt(2)
    bcc.set_cell(cell, scale_atoms=True)
    mc = MC(len(fcc))
    mc.set_magnetic_moments(np.ones(len(fcc))[:, None] * [2.3, 0, 0])
    mc.set_lambda(lambda_)
    neigh_fcc = fcc.get_neighbors(num_neighbors=86)
    neigh_bcc = bcc.get_neighbors(num_neighbors=58)
    for ii, mat in enumerate(neigh_fcc.get_shell_matrix()):
        mc.set_heisenberg_coeff(data_dict_fcc[f"J_{ii}"] * mat, index=0)
    for ii, mat in enumerate(neigh_bcc.get_shell_matrix()):
        mc.set_heisenberg_coeff(data_dict_bcc[f"J_{ii}"] * mat, index=1)
    for ii in range(10):
        if f"A_{ii}" not in data_dict_fcc:
            break
        mc.set_landau_coeff(data_dict_fcc[f"A_{ii}"], 2 * (ii + 1), index=0)
    for ii in range(10):
        if f"A_{ii}" not in data_dict_bcc:
            break
        mc.set_landau_coeff(data_dict_bcc[f"A_{ii}"], 2 * (ii + 1), index=1)
    mc.run(temperature=temperature, number_of_iterations=n_steps)
    return mc.get_magnetic_moments(), mc.get_magnetization(), mc.get_energy(index=0), mc.get_energy(index=1)
