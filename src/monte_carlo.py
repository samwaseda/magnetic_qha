from mamonca import MC
from magnetic_qha.src.fitting import get_structure


def run_simple_mc(temperature, n_steps, cs, data_dict, metadynamics=False, n_repeat=20):
    structure = get_structure(cs, n_repeat=n_repeat)
    mc = MC(len(structure))
    if cs == "bcc":
        neigh = structure.get_neighbors(num_neighbors=58)
    else:
        neigh = structure.get_neighbors(num_neighbors=86)
    shell_matrices = neigh.get_shell_matrix()
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
