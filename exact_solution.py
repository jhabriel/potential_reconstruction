"""Module containing the exact solution."""
import numpy as np
import porepy as pp
import quadpy
import sympy as sym

from utils import get_quadpy_elements


class ExactSolution:
    """Class containing the exact manufactured solution."""

    def __init__(self, model):
        """Constructor of the class."""

        # Physical parameters
        perm = model.params.get("permeability", np.eye(2))

        # Symbolic variables
        x, y = sym.symbols("x y")
        K_xx = perm[0][0]
        K_xy = perm[0][1]
        K_yx = perm[1][0]
        K_yy = perm[1][1]

        # Pressure
        pressure_solution = model.params.get("pressure_solution", "parabolic")
        if pressure_solution == "parabolic":
            p = x * (1 - x) * y * (1 - y)
        elif pressure_solution == "trigonometric":
            p = sym.cos(2 * sym.pi * x) * sym.cos(2 * sym.pi * y)
        else:
            raise NotImplementedError()

        # Pressure gradient
        gradp_x = sym.diff(p, x)
        gradp_y = sym.diff(p, y)

        # Darcy velocity
        q_x = -K_xx * gradp_x - K_xy * gradp_y
        q_y = -K_yx * gradp_x - K_yy * gradp_y

        # Divergence of velocity
        div_q = sym.diff(q_x, x) + sym.diff(q_y, y)

        # Source term
        f = div_q

        # Public attributes
        self.p = p
        self.gradp_x = gradp_x
        self.gradp_y = gradp_y
        self.q_x = q_x
        self.q_y = q_y
        self.f = f

    def pressure(self, sd: pp.Grid) -> np.ndarray:
        """Evaluate exact pressure at the cell centers.

        Parameters:
            sd: pp.Grid.
                Subdomain grid.

        Returns:
            Array of ``shape=(sd.num_cells, )`` containing the exact pressures at
            the cell centers.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get cell centers
        cc = sd.cell_centers

        # Lambdify expression
        p_fun = sym.lambdify((x, y), self.p, "numpy")

        # Cell-centered pressures
        p_cc = p_fun(cc[0], cc[1])

        return p_cc

    def flux(self, sd: pp.Grid) -> np.ndarray:
        """

        :param sd:
        :return:
        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdyfy expressions
        qx_fun = sym.lambdify((x, y), self.q_x, "numpy")
        qy_fun = sym.lambdify((x, y), self.q_y, "numpy")

        # Evaluate
        fc = sd.face_centers
        fn = sd.face_normals
        q = qx_fun(fc[0], fc[1]) * fn[0] + qy_fun(fc[0], fc[1]) * fn[1]

        return q

    def boundary_values(self, boundary_grid: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            boundary_grid: Boundary grid.

        Returns:
            Array of ``shape=(boundary_grid.num_cells, )`` with the exact pressure
            values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdify expression
        p_fun = sym.lambdify((x, y), self.p, "numpy")

        # Boundary pressures
        fc = boundary_grid.cell_centers
        p_bf = p_fun(fc[0], fc[1])

        return p_bf

    def integrated_source(self, sd: pp.Grid):
        """

        :param sd_matrix:
        :return:
        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Lambdify expression
        f_fun = sym.lambdify((x, y), self.f, "numpy")

        # Declare integration method and get hold of elements in QuadPy format
        int_method = quadpy.t2.get_good_scheme(10)
        elements = get_quadpy_elements(sd)

        # Declare integrand
        def integrand(x):
            return f_fun(x[0], x[1]) * np.ones_like(x[0])

        # Perform numerical integration
        integral = int_method.integrate(integrand, elements)

        return integral
