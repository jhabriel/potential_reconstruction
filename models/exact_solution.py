"""Module containing the exact solution."""

import numpy as np
import porepy as pp
import quadpy
import sympy as sym

import mdamr as amr


class ExactSolution:
    """Class containing the exact manufactured solution."""

    def __init__(self):
        """Constructor of the class."""

        # Symbolic variables
        x, y = sym.symbols("x y")
        K_x, K_y = sym.Float(1.0), sym.Float(1000.0)

        # Pressure
        p = sym.sin(sym.pi / sym.sqrt(K_x) * x) * sym.sin(sym.pi / sym.sqrt(K_y) * y)

        # Darcy velocity
        q = [- K_x * sym.diff(p, x), - K_y * sym.diff(p, y)]

        # Divergence of velocity
        div_q = sym.diff(q[0], x) + sym.diff(q[1], y)

        # Source term
        f = div_q

        # Public attributes
        self.p = p
        self.q = q
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

    def boundary_values(self, boundary_grid_matrix: pp.BoundaryGrid) -> np.ndarray:
        """Exact pressure at the boundary faces.

        Parameters:
            boundary_grid_matrix: Matrix boundary grid.

        Returns:
            Array of ``shape=(boundary_grid_matrix.num_cells, )`` with the exact
            pressure values at the exterior boundary faces.

        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of face indices
        fc = boundary_grid_matrix.cell_centers
        bot = fc[1] < 0.25
        mid = (fc[1] >= 0.25) & (fc[1] <= 0.75)
        top = fc[1] > 0.75
        face_idx = [bot, mid, top]

        # Lambdify expression
        p_fun = [sym.lambdify((x, y), p, "numpy") for p in self.p_matrix]

        # Boundary pressures
        p_bf = np.zeros(boundary_grid_matrix.num_cells)
        for p, idx in zip(p_fun, face_idx):
            p_bf += p(fc[0], fc[1]) * idx

        return p_bf

    def integrated_matrix_source(self, sd_matrix: pp.Grid):
        """

        :param sd_matrix:
        :return:
        """
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]

        # Declare integration method and get hold of elements in QuadPy format
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd_matrix)

        integral = np.zeros(sd_matrix.num_cells)
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x):
                return f(x[0], x[1]) * np.ones_like(x[0])

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return integral

    def integrated_fracture_source(self, sd_frac):
        """

        :param sd_frac:
        :return:
        """
        # Symbolic variable
        y = sym.symbols("y")

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        method = quadpy.c1.newton_cotes_closed(10)
        elements = amr.utils.get_quadpy_elements(sd_frac)
        elements *= -1  # we have to use the real `y` coordinates here

        def integrand(y):
            return f_fun(y)

        integral = method.integrate(integrand, elements)

        return integral

    def residual_error_matrix(self, sd_matrix: pp.Grid, d_matrix: dict) -> np.ndarray:
        """Compute square of residual errors for 2D (only the norm)"""
        # Symbolic variables
        x, y = sym.symbols("x y")

        # Get list of cell indices
        cc = sd_matrix.cell_centers
        bot = cc[1] < 0.25
        mid = (cc[1] >= 0.25) & (cc[1] <= 0.75)
        top = cc[1] > 0.75
        cell_idx = [bot, mid, top]

        # Lambdify expression
        f_fun = [sym.lambdify((x, y), f, "numpy") for f in self.f_matrix]

        # Retrieve reconstructed velocity and compute divergence
        recon_u = d_matrix["estimates"]["recon_sd_flux"].copy()
        u = amr.utils.poly2col(recon_u)
        div_u = 2 * u[0]

        # Integration method and retrieving elements
        int_method = quadpy.t2.get_good_scheme(10)
        elements = amr.utils.get_quadpy_elements(sd_matrix)

        integral = np.zeros(sd_matrix.num_cells)
        weights = (sd_matrix.cell_diameters() / np.pi) ** 2
        for f, idx in zip(f_fun, cell_idx):
            # Declare integrand
            def integrand(x):
                return (f(x[0], x[1]) * np.ones_like(x[0]) - div_u) ** 2

            # Integrate, and add the contribution of each subregion
            integral += int_method.integrate(integrand, elements) * idx

        return weights * integral

    def residual_error_fracture(self, sd_frac: pp.Grid, d_frac: dict) -> np.ndarray:
        """Compute square of residual errors for 2D (only the norm)"""

        # Retrieve reconstructed velocity and compute its divergence
        recon_u = d_frac["estimates"]["recon_sd_flux"].copy()
        u = amr.utils.poly2col(recon_u)
        div_u = u[0]

        # Contribution from interface fluid fluxes to mass balance equation
        sources_from_intf = d_frac["estimates"]["sources_from_intf"].copy()

        # Integration method and retrieving elements
        y = sym.symbols("y")

        # Lambdify expression
        f_fun = sym.lambdify(y, self.f_frac, "numpy")

        method = quadpy.c1.newton_cotes_closed(10)
        elements = amr.utils.get_quadpy_elements(sd_frac)
        elements *= -1  # we have to use the real `y` coordinates here

        weights = (sd_frac.cell_diameters() / np.pi) ** 2

        def integrand(y):
            return (f_fun(y) - div_u + sources_from_intf) ** 2

        integral = method.integrate(integrand, elements)

        return weights * integral


#%%
nx = np.array([10, 10])
physdims = np.ones(2)
g = pp.StructuredTriangleGrid(nx, physdims, name="Regular structured triangular grid")
g.compute_geometry()

ex = ExactSolution()
ptrue = ex.pressure(g)
pp.plot_grid(g, ptrue, plot_2d=True)