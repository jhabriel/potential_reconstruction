"""
This module contains a code verification implementation using a manufactured solution
for the two-dimensional, incompressible, single phase flow.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import porepy as pp
import porepy.models.geometry
from porepy.applications.convergence_analysis import ConvergenceAnalysis
from porepy.fracs.fracture_network_3d import FractureNetwork3d
from porepy.utils.examples_utils import VerificationUtils
from porepy.viz.data_saving_model_mixin import VerificationDataSaving
from grids import MeshGenerator

from exact_solution import ExactSolution

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
    "permeability": 1.0,  # (**)
}


# -----> Data-saving
@dataclass
class ManuIncompSaveData:
    """Data class to save relevant results from the verification setup."""

    approx_flux: np.ndarray
    """Numerical flux."""

    approx_pressure: np.ndarray
    """Numerical pressure."""

    error_pressure: number
    """L2-discrete relative error for the pressure."""

    error_flux: number
    """L2-discrete relative error for the flux."""

    exact_pressure: np.ndarray
    """Exact pressure."""

    exact_flux: np.ndarray
    """Exact flux."""


class DataSaving(VerificationDataSaving):
    """Mixin class to save relevant data."""

    darcy_flux: Callable[[list[pp.Grid]], pp.ad.Operator]
    """Method that returns the Darcy fluxes in the form of an Ad operator. Usually
    provided by the mixin class :class:`porepy.models.constitutive_laws.DarcysLaw`.

    """

    exact_sol: ExactSolution
    """Exact solution object."""

    pressure: Callable[[list[pp.Grid]], pp.ad.MixedDimensionalVariable]
    """Pressure variable. Normally defined in a mixin instance of
    :class:`~porepy.models.fluid_mass_balance.VariablesSinglePhaseFlow`.

    """

    def collect_data(self) -> ManuIncompSaveData:
        """Collect data from the verification setup.

        Returns:
            ManuIncompSaveData object containing the results of the verification.

        """

        sd: pp.Grid = self.mdg.subdomains()[0]
        exact_sol: ExactSolution = self.exact_sol

        # Collect data
        exact_pressure = exact_sol.pressure(sd)
        pressure_ad = self.pressure([sd])
        approx_pressure = pressure_ad.value(self.equation_system)
        error_pressure = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_pressure,
            approx_array=approx_pressure,
            is_scalar=True,
            is_cc=True,
            relative=True,
        )

        exact_flux = exact_sol.flux(sd)
        flux_ad = self.darcy_flux([sd])
        approx_flux = flux_ad.value(self.equation_system)
        error_flux = ConvergenceAnalysis.l2_error(
            grid=sd,
            true_array=exact_flux,
            approx_array=approx_flux,
            is_scalar=True,
            is_cc=False,
            relative=True,
        )

        # Store collected data in data class
        collected_data = ManuIncompSaveData(
            exact_pressure=exact_pressure,
            exact_flux=exact_flux,
            approx_pressure=approx_pressure,
            approx_flux=approx_flux,
            error_pressure=error_pressure,
            error_flux=error_flux,
        )

        return collected_data


class Utils(VerificationUtils):
    """Mixin class containing useful utility methods for the setup."""

    mdg: pp.MixedDimensionalGrid
    """Mixed-dimensional grid."""

    results: list[ManuIncompSaveData]
    """List of ManuIncompSaveData objects."""

    def plot_results(self) -> None:
        """Plotting results."""
        self._plot_pressure()

    def _plot_pressure(self) -> None:
        """Plots exact and numerical pressures in the matrix."""
        sd = self.mdg.subdomains()[0]
        p_num = self.results[-1].approx_pressure
        p_ex = self.results[-1].exact_pressure
        pp.plot_grid(
            sd, p_ex, plot_2d=True, linewidth=0, title="Matrix pressure (Exact)"
        )
        pp.plot_grid(
            sd, p_num, plot_2d=True, linewidth=0, title="Matrix pressure (MPFA)"
        )


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
        mesh_size: float = self.params.get("mesh_size", 0.1)
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


class SolutionStrategy(
    pp.fluid_mass_balance.SolutionStrategySinglePhaseFlow
):
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

    solid: pp.SolidConstants
    """Object containing the solid constants."""

    results: list[ManuIncompSaveData]
    """List of SaveData objects."""

    error_estimates_data_saving: Callable
    """Method to save solution data to be used in a posteriori error estimation."""

    def __init__(self, params: dict):
        """Constructor for the class."""

        super().__init__(params)

        self.exact_sol: ExactSolution
        """Exact solution object."""

        self.results: list[ManuIncompSaveData] = []
        """Results object that stores exact and approximated solutions and errors."""

    def set_materials(self):
        """Set material constants for the verification setup."""
        super().set_materials()

        # Sanity checks to guarantee the validity of the manufactured solution
        assert self.fluid.density() == 1
        assert self.fluid.viscosity() == 1
        assert self.fluid.compressibility() == 0
        assert self.solid.permeability() == 1

        # Instantiate exact solution object after materials have been set
        self.exact_sol = ExactSolution()

    def after_simulation(self) -> None:
        """Method to be called after the simulation has finished."""
        if self.params.get("plot_results", False):
            self.plot_results()

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
    pp.fluid_mass_balance.SinglePhaseFlow,
):
    """
    Mixer class for the 2d incompressible flow setup with a single fracture.
    """

#%% Runner
solid_constants = pp.SolidConstants(manu_incomp_solid)
fluid_constants = pp.FluidConstants(manu_incomp_fluid)
material_constants = {"solid": solid_constants, "fluid": fluid_constants}
params = {
    "material_constants": material_constants,
    "plot_results": True,
    "mesh_size": 0.1,
    "dim": 2,
    "domain_size": np.array([1.0, 1.0]),
}
model = ManufacturedModel(params=params)
pp.run_time_dependent_model(model, {})