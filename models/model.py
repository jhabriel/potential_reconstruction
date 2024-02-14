"""
This module contains a code verification implementation using a manufactured solution
for the two-dimensional, incompressible, single phase flow.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mdamr
import numpy as np
import porepy as pp
import porepy.models.geometry
import quadpy
import sympy as sym
from mdamr.estimates.pressure_reconstruction import (
    keilegavlen_p1, patchwise_p1, vohralik_p2
)
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.utils.examples_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving

from exact_solution import ExactSolution
from grids import MeshGenerator

# PorePy typings
number = pp.number
grid = pp.GridLike

# Material constants for the verification setup. Constants with (**) cannot be
# changed since the manufactured solution implicitly assume such values.
manu_incomp_fluid: dict[str, number] = {
    "compressibility": 0,  # (**)
    "density": 1.0,  # (**)
    "viscosity": 1.0,  # (**)
}

manu_incomp_solid: dict[str, number] = {
    "residual_aperture": 1.0,  # (**)
}


class Postprocessing:
    """
    Mixin for various postprocessing steps, including flux extension using RT0,
    potential reconstructions, and error computation.
    """

    mdg: pp.MixedDimensionalGrid
    pressure: Callable
    darcy_flux: Callable
    equation_system: pp.EquationSystem
    exact_sol: ExactSolution

    def postprocess_solution(self) -> None:
        """Caller for postprocessing methods.

        These methods include: Transferring of solutions to the data dictionary,
        extension of normal finite volume fluxes, reconstruction of potentials using
        different techniques, and the computation of errors.

        """
        self.transfer_solution()
        self.extend_normal_fluxes()
        self.reconstruct_potentials()
        self.compute_errors()

    def transfer_solution(self):
        """Evaluate and transfer FV solutions to data dictionary."""

        for sd, d in self.mdg.subdomains(return_data=True):
            # Create key if it does not exist
            if d.get("estimates") is None:
                d["estimates"] = {}

            # Save FV P0 pressures
            d["estimates"]["fv_sd_pressure"] = self.pressure([sd]).value(
                self.equation_system
            )

            # Save FV P0 normal fluxes
            d["estimates"]["fv_sd_flux"] = self.darcy_flux([sd]).value(
                self.equation_system
            )

    def extend_normal_fluxes(self) -> None:
        """Extend FV fluxes using RT0-basis functions."""
        mdamr.extend_fv_fluxes(self.mdg)

    def reconstruct_potentials(self) -> None:
        """Reconstruct potentials using three different techniques."""

        # Retrieve subdomain grid and its dictionary
        sd = self.mdg.subdomains()[0]
        sd_data = self.mdg.subdomain_data(sd)

        # Retrieve boundary grid and its dictionary
        bg = self.mdg.subdomain_to_boundary_grid(sd)
        bg_data = self.mdg.boundary_grid_data(bg)

        # Method 1: Cochez-Dhondt
        point_val, point_coo = patchwise_p1(sd, sd_data, bg_data)
        sd_data["estimates"]["p_recon_avg_p1"] = mdamr.utils.interpolate_p1(
            point_val, point_coo
        )

        # Method 2: Keilegavlen-Varela
        point_val, point_coo = keilegavlen_p1(sd, sd_data, bg_data)
        sd_data["estimates"]["p_recon_rt0_p1"] = mdamr.utils.interpolate_p1(
            point_val, point_coo
        )

        # Method 3: Vohralik-Ern
        point_val, point_coo = vohralik_p2(sd, sd_data, bg_data)
        sd_data["estimates"]["p_recon_neu_p2"] = (
            mdamr.utils.interpolate_p2(point_val, point_coo)
        )

    def compute_errors(self) -> None:
        """Compute errors for the different reconstruction schemes."""

        # Retrieve subdomain and data
        sd = self.mdg.subdomains()[0]
        d = self.mdg.subdomain_data(sd)

        # Potential reconstruction methods
        reconstructions = ["avg_p1", "rt0_p1", "neu_p2", "post_p2"]

        # Exact pressure and pressure gradients
        x, y = sym.symbols("x y")
        ex = ExactSolution(self)
        p = ex.p
        p_fun = sym.lambdify((x, y), p, "numpy")
        gradp = [sym.diff(ex.p, x), sym.diff(ex.p, y)]
        gradp_fun = [sym.lambdify((x, y), grad, "numpy") for grad in gradp]

        # Obtain elements and declare integration method
        method = quadpy.t2.get_good_scheme(10)
        elements = mdamr.utils.get_quadpy_elements(sd)

        for reconstruction in reconstructions:

            # Retrieve reconstructed pressures
            recon_p = d["estimates"]["p_recon_" + reconstruction]
            pr = mdamr.utils.poly2col(recon_p)

            def integrand_h1_error(x):
                # Exact pressure gradient in x and y
                gradp_exact_x = gradp_fun[0](x[0], x[1])
                gradp_exact_y = gradp_fun[1](x[0], x[1])

                # Reconstructed pressure gradient in x and y
                ones = np.ones_like(x[0])
                if reconstruction in ["avg_p1", "rt0_p1"]:  # P1(K) methods
                    gradp_recon_x = pr[0] * ones
                    gradp_recon_y = pr[1] * ones
                elif reconstruction in ["neu_p2", "post_p2"]:  # P2(K) methods
                    gradp_recon_x = 2 * pr[0] * x[0] + pr[1] * x[1] + pr[2] * ones
                    gradp_recon_y = pr[1] * x[0] + 2 * pr[3] * x[1] + pr[4] * ones
                else:
                    raise NotImplementedError()

                # Integral in x and y
                int_x = (gradp_exact_x - gradp_recon_x) ** 2
                int_y = (gradp_exact_y - gradp_recon_y) ** 2

                return int_x + int_y

            integral = method.integrate(integrand_h1_error, elements)
            d["estimates"]["local_error_h1_" + reconstruction] = integral
            d["estimates"]["error_h1_" + reconstruction] = integral.sum() ** 0.5

            def integrand_l2_error(x):
                # Exact pressure
                p_exact = p_fun(x[0], x[1])
                # Reconstructed pressures
                ones = np.ones_like(x[0])
                if reconstruction in ["avg_p1", "rt0_p1"]:  # P1(K) methods
                    p_recon = pr[0] * x[0] + pr[1] * x[1] + pr[2] * ones
                elif reconstruction in ["neu_p2", "post_p2"]:  # P2(K) methods
                    p_recon = (
                        pr[0] * x[0] ** 2
                        + pr[1] * x[0] * x[1]
                        + pr[2] * x[0]
                        + pr[3] * x[1] ** 2
                        + pr[4] * x[1]
                        + pr[5]
                    )
                return (p_exact - p_recon) ** 2

            integral = method.integrate(integrand_l2_error, elements)
            d["estimates"]["local_error_l2" + reconstruction] = integral
            d["estimates"]["error_l2_" + reconstruction] = integral.sum() ** 0.5


@dataclass
class SaveData:
    """Data class to save relevant results from the verification setup."""

    # error_l2_avg: float
    # """Error in the L2 broken norm obtained with average of cell-centered potentials."""
    #
    # error_l2_rt0: float
    # """Error in the L2 broken norm obtained with RT0-based reconstruction."""
    #
    # error_l2_neu: float
    # """Error in the L2 broken norm obtained by solving a local Neumann problem."""

    # error_l2_postp: float
    # """Error in the L2 broken norm for the post-processed potential."""

    error_h1_avg: float
    """Error in the H1 broken norm obtained with average of cell-centered potentials."""

    error_h1_rt0: float
    """Error in the H1 broken norm obtained with RT0-based reconstruction."""

    error_h1_neu: float
    """Error in the H1 broken norm obtained by solving a local Neumann problem."""

    # error_h1_postp: float
    # """Error in the H1 broken norm for the post-processed potential."""



class DataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    def collect_data(self) -> SaveData:
        """Collect data from the verification setup.

        Returns:
            ManuIncompSaveData object containing the results of the verification.

        """

        sd: pp.Grid = self.mdg.subdomains()[0]
        d: dict = self.mdg.subdomain_data(sd)

        # Store collected data in data class
        collected_data = SaveData(
            error_h1_rt0=d["estimates"]["error_h1_rt0_p1"],
            error_h1_avg=d["estimates"]["error_h1_avg_p1"],
            error_h1_neu=d["estimates"]["error_h1_neu_p2"],
            # error_h1_postp=d["estimates"]["error_h1_post_p2"],
            # error_l2_rt0=d["estimates"]["error_l2_rt0_p1"],
            # error_l2_avg=d["estimates"]["error_l2_avg_p1"],
            # error_l2_neu=d["estimates"]["error_l2_neu_p2"],
            # error_l2_postp=d["estimates"]["error_l2_post_p2"],
        )

        return collected_data


class Utils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[SaveData]
    """List of ManuIncompSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_pressure()

    def plot_errors(self) -> None:
        """Plotting (square) of local errors."""
        reconstructions = ["avg_p1", "rt0_p1", "neu_p2", "post_p2"]
        sd = self.mdg.subdomains()[0]
        d = self.mdg.subdomain_data(sd)
        for reconstruction in reconstructions:
            pp.plot_grid(
                sd,
                d["estimates"]["local_error" + reconstruction],
                plot_2d=True,
                linewidth=0,
                title=f"Error {reconstruction}",
            )

    def _plot_pressure(self) -> None:
        """Plots exact and numerical pressures in the matrix."""
        sd = self.mdg.subdomains()[0]
        d = self.mdg.subdomain_data(sd)
        cc = sd.cell_centers
        ex = ExactSolution()

        # Exact pressure
        p_ex = ex.pressure(sd)
        pp.plot_grid(sd, p_ex, plot_2d=True, linewidth=0, title="Exact")

        # P0 pressure
        p0 = self.pressure([sd]).value(self.equation_system)
        pp.plot_grid(sd, p0, plot_2d=True, linewidth=0, title="P0")

        # AVG_P1 pressure
        p_avg = d["estimates"]["p_recon_avg_p1"]
        cc_p_avg = p_avg[:, 0] * cc[0] + p_avg[:, 1] * cc[1] + p_avg[:, 2]
        pp.plot_grid(sd, cc_p_avg, plot_2d=True, linewidth=0, title="AVG P1")

        # RTO_P1 pressure
        p_rt0 = d["estimates"]["p_recon_rt0_p1"]
        cc_p_rt0 = p_rt0[:, 0] * cc[0] + p_rt0[:, 1] * cc[1] + p_rt0[:, 2]
        pp.plot_grid(sd, cc_p_rt0,  plot_2d=True, linewidth=0, title="RT0 P1")

        # NEU_P2 pressure
        p_neu = d["estimates"]["p_recon_neu_p2"]
        cc_p_neu = (
            p_neu[:, 0] * cc[0] ** 2
            + p_neu[:, 1] * cc[0] * cc[1]
            + p_neu[:, 2] * cc[0]
            + p_neu[:, 3] * cc[1] ** 2
            + p_neu[:, 4] * cc[1]
            + p_neu[:, 5]
        )
        pp.plot_grid(sd, cc_p_neu, plot_2d=True, linewidth=0, title="NEU P2")

        # POS_P2 pressure
        p_post = d["estimates"]["p_recon_post_p2"]
        cc_p_post = (
                p_post[:, 0] * cc[0] ** 2
                + p_post[:, 1] * cc[0] * cc[1]
                + p_post[:, 2] * cc[0]
                + p_post[:, 3] * cc[1] ** 2
                + p_post[:, 4] * cc[1]
                + p_post[:, 5]
        )
        pp.plot_grid(sd, cc_p_post, plot_2d=True, linewidth=0, title="POSTP P2")


class Geometry(porepy.models.geometry.ModelGeometry):
    """Class for setting up the geometry of the unit square domain."""

    params: dict
    """Simulation model parameters."""

    def set_geometry(self) -> None:
        """Define geometry and create a mixed-dimensional grid.

        The default values provided in set_domain, set_fractures, grid_type and
        meshing_arguments produce a 2d unit square domain with no fractures and a four
        Cartesian cells.

        """
        # Retrieve information from data parameter
        domain_size: np.ndarray = self.params.get("domain_size", np.array([1.0, 1.0]))
        mesh_size: float = self.params["meshing_arguments"].get("cell_size", 0.1)
        dim: int = self.params.get("dim", 2)

        # Set domain
        if dim == 2:
            x_max = domain_size[0]
            y_max = domain_size[1]
            self._domain = pp.Domain({"xmax": x_max, "ymax": y_max})
        else:
            x_max = domain_size[0]
            y_max = domain_size[1]
            z_max = domain_size[2]
            self._domain = pp.Domain({"xmax": x_max, "ymax": y_max, "zmax": z_max})

        # Set fractures
        self.set_fractures()

        # Create a fracture network.
        self.fracture_network = pp.create_fracture_network(self.fractures, self.domain)

        # Create grid from the mesh generator
        me = MeshGenerator(
            domain=domain_size,
            dim=dim,
            mesh_size=mesh_size,
        )
        default_mesh_type = "regular_structured"
        mesh_type = self.params.get("mesh_type", default_mesh_type)
        if mesh_type == "regular_structured":
            sd = me.regular_structured_simplex()
        elif mesh_type == "irregular_structured":
            sd = me.irregular_structured_simplex()
        elif mesh_type == "unstructured":
            sd = me.unstructured_simplex(perturb_nodes=False)
        else:
            sd = me.unstructured_simplex(perturb_nodes=True)

        # Create mixed-dimensional grid from subdomain grid
        self.mdg = pp.meshing.subdomains_to_mdg([[sd]])
        self.nd: int = self.mdg.dim_max()

        # Create projections between local and global coordinates for fracture grids.
        pp.set_local_coordinate_projections(self.mdg)

        self.set_well_network()
        if len(self.well_network.wells) > 0:
            # Compute intersections
            assert isinstance(self.fracture_network, FractureNetwork3d)
            pp.compute_well_fracture_intersections(
                self.well_network, self.fracture_network
            )
            # Mesh wells and add fracture + intersection grids to mixed-dimensional
            # grid along with these grids' new interfaces to fractures.
            self.well_network.mesh(self.mdg)


class BoundaryConditions(pp.fluid_mass_balance.BoundaryConditionsSinglePhaseFlow):
    """Set boundary conditions for the simulation model."""

    exact_sol: ExactSolution
    """Exact solution object."""

    def bc_type_darcy_flux(self, sd: pp.Grid) -> pp.BoundaryCondition:
        """Set boundary condition type."""
        boundary_faces = self.domain_boundary_sides(sd).all_bf
        return pp.BoundaryCondition(sd, boundary_faces, "dir")

    def bc_values_pressure(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Analytical boundary condition values for Darcy flux.

        Parameters:
            boundary_grid: Boundary grid for which to define boundary conditions.

        Returns:
            Boundary condition values array.

        """
        return self.exact_sol.boundary_values(boundary_grid=boundary_grid)


class BalanceEquation(pp.fluid_mass_balance.MassBalanceEquations):
    """Modify balance equation to account for external sources."""

    exact_sol: ExactSolution
    """Exact solution object."""

    def fluid_source(self, subdomains: list[pp.Grid]) -> pp.ad.Operator:
        """Contribution of mass fluid sources to the mass balance equation.

        Parameters:
            subdomains: List of subdomains.

        Returns:
            Cell-wise Ad operator containing the fluid source contributions.

        """
        # Retrieve internal sources (jump in mortar fluxes) from the base class
        internal_sources: pp.ad.Operator = super().fluid_source(subdomains)

        # Retrieve external (integrated) sources from the exact solution.
        sd = self.mdg.subdomains()[0]
        external_sources = pp.wrap_as_dense_ad_array(
            self.exact_sol.integrated_source(sd)
        )

        # Add up both contributions
        source = internal_sources + external_sources
        source.set_name("fluid sources")

        return source


class SolutionStrategy(pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow):
    """Modified solution strategy for the verification setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    exact_sol: ExactSolution
    """Exact solution object."""

    fluid: pp.FluidConstants
    """Object containing the fluid constants."""

    plot_results: Callable
    """Method to plot results of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    plot_errors: Callable
    """Method to plot errors of the verification setup. Usually provided by the
    mixin class :class:`SetupUtilities`.

    """

    postprocess_solution: Callable
    """Postprocess solutions to reconstruct potentials and compute errors. Method 
    provided by the mixin class :class:`Postprocessing`.
    
    """

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    results: list[SaveData]
    """List of SaveData objects."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: ExactSolution
        """Exact solution object."""

        self.results: list[SaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.density() == 1
        assert self.fluid.viscosity() == 1
        assert self.fluid.compressibility() == 0

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ExactSolution(self)

    def set_discretization_parameters(self) -> None:
        """Set anisotropic permeability"""
        super().set_discretization_parameters()

        # Retrieve subdomain and data dictionary
        sd = self.mdg.subdomains()[0]
        d = self.mdg.subdomain_data(sd)
        kw = self.darcy_keyword

        # Declare permeability matrix. Identity by default.
        perm = self.params.get("permeability", np.eye(2))
        perm_xx = perm[0][0]
        perm_xy = perm[0][1]
        perm_yx = perm[1][0]
        perm_yy = perm[1][1]
        assert perm_xy == perm_yx, "Permeability matrix not symmetric."

        d[pp.PARAMETERS][kw]["second_order_tensor"] = pp.SecondOrderTensor(
            kxx=perm_xx * np.ones(sd.num_cells),
            kyy=perm_yy * np.ones(sd.num_cells),
            kxy=perm_xy * np.ones(sd.num_cells),
        )

    def after_nonlinear_convergence(
        self, solution: np.ndarray, errors: float, iteration_counter: int
    ) -> None:
        solution = self.equation_system.get_variable_values(iterate_index=0)
        self.equation_system.shift_time_step_values()
        self.equation_system.set_variable_values(
            values=solution, time_step_index=0, additive=False
        )
        self.convergence_status = True
        self.postprocess_solution()
        self.save_data_time_step()

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        # Plot solutions
        if self.params.get("plot_results", False):
            self.plot_results()
        # Plot errors
        if self.params.get("plot_errors", False):
            self.plot_errors()

    def _is_nonlinear_problem(self) -> bool:
        """The problem is linear."""
        return False

    def _is_time_dependent(self) -> bool:
        """The problem is stationary."""
        return False


# -----> Mixer
class ManufacturedModel(  # type: ignore[misc]
    Geometry,
    BalanceEquation,
    BoundaryConditions,
    SolutionStrategy,
    Utils,
    DataSaving,
    Postprocessing,
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d incompressible flow setup with a single fracture.
    """