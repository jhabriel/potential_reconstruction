import porepy as pp
import numpy as np
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from model import ManufacturedModel,  manu_incomp_fluid, manu_incomp_solid

# Declare material parameters
solid = pp.SolidConstants(manu_incomp_solid)
fluid = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid, "fluid": fluid}

# Declare mesh types
mesh_types = [
    "regular_structured",
    "irregular_structured",
    "unstructured",
    "perturbed_unstructured"
]

# Declare solution types
solutions = ["parabolic", "trigonometric"]

# Declare permeability types
permeability = [np.eye(2), np.array([[7.7500, 3.8971], [3.8971, 3.2500]])]

# Run analysis
for sol, perm in zip(solutions, permeability):
    for mesh_type in mesh_types:

        # Model parameters
        params = {
            "material_constants": material_constants,
            "dim": 2,
            "domain_size": np.array([1.0, 1.0]),
            "mesh_type": mesh_type,
            "meshing_arguments": {"cell_size": 0.1},
            "plot_results": False,
            "save_grid": False,
            "save_exact_pressure": False,
            "pressure_solution": sol,
            "permeability": perm,
        }

        # Run convergence analysis
        convergence_analysis = ConvergenceAnalysis(
            model_class=ManufacturedModel,
            model_params=params,
            levels=7,
            spatial_refinement_rate=2,
        )
        list_of_results = convergence_analysis.run_analysis()
        convergence_analysis.export_errors_to_txt(
            list_of_results=list_of_results,
            file_name=f"error_{mesh_type}_{sol[:4]}.txt"
        )
        order_of_convergence = convergence_analysis.order_of_convergence(
            list_of_results=list_of_results,
            x_axis="cell_diameter",
            base_log_x_axis=2,
            base_log_y_axis=2,
        )
        print(order_of_convergence)
