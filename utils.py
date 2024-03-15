import numpy as np
import porepy as pp
import scipy.sparse as sps


def get_quadpy_elements(sd: pp.Grid) -> np.ndarray:
    """
    Assembles the elements of a given grid in quadpy format: https://pypi.org/project/quadpy/.

    Parameters
    ----------
        sd (pp.Grid): PorePy grid.

    Returns
    --------
    quadpy_elements (np.ndarray): Elements in QuadPy format.

    Example
    --------
    >>> # shape (3, 5, 2), i.e., (corners, num_triangles, xy_coords)
    >>> triangles = np.stack([
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[1.2, 0.6], [1.3, 0.7], [1.4, 0.8]],
            [[26.0, 31.0], [24.0, 27.0], [33.0, 28]],
            [[0.1, 0.3], [0.4, 0.4], [0.7, 0.1]],
            [[8.6, 6.0], [9.4, 5.6], [7.5, 7.4]]
            ], axis=-2)
    """
    # Renaming variables
    nc = sd.num_cells

    # Getting node coordinates for each cell
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(nc, sd.dim + 1)
    nodes_coor_cell = sd.nodes[:, nodes_of_cell]

    # Stacking node coordinates
    cnc_stckd = np.empty([nc, (sd.dim + 1) * sd.dim])
    col = 0
    for vertex in range(sd.dim + 1):
        for dim in range(sd.dim):
            cnc_stckd[:, col] = nodes_coor_cell[dim][:, vertex]
            col += 1
    element_coord = np.reshape(cnc_stckd, np.array([nc, sd.dim + 1, sd.dim]))

    # Reshaping to please quadpy format i.e., (corners, num_elements, coords)
    elements = np.stack(element_coord, axis=-2)  # type:ignore

    # For some reason, quadpy needs a different formatting for line segments
    if sd.dim == 1:
        elements = elements.reshape(sd.dim + 1, sd.num_cells)

    return elements

def interpolate_p1(point_val: np.ndarray, point_coo: np.ndarray) -> np.ndarray:
    """
    Performs a linear local interpolation of a P1 FEM element given
    the pressure values and the coordinates at the Lagrangian nodes.

    Parameters
    ----------
    point_val : NumPy nd-array of shape (g.num_cells x num_Lagr_nodes)
        Pressures values at the Lagrangian nodes.
    point_coo : NumPy nd-array of shape (g.dim x g.num_cells x num_Lagr_nodes)
        Coordinates of the Lagrangian nodes. In the case of embedded entities,
        the points should correspond to the rotated coordinates.

    Raises
    ------
    Value Error
        If the number of columns of point_val is different from 4 (3D), 3 (2d),
        or 2 (1D)

    Returns
    -------
    coeff : Numpy nd-array of shape (g.num_cells x (g.dim+1))
        Coefficients of the cell-wise P1 polynomial satisfying:
        c0 x + c1                   (1D),
        c0 x + c1 y + c2            (2D),
        c0 x + c1 y + c3 z + c4     (3D).

    """

    # Get rows, cols, and dimensionality
    rows = point_val.shape[0]  # number of cells
    cols = point_val.shape[1]  # number of Lagrangian nodes per cell
    if cols == 4:
        dim = 3
    elif cols == 3:
        dim = 2
    elif cols == 2:
        dim = 1
    else:
        raise ValueError("P1 reconstruction only valid for 1d, 2d, and 3d.")

    if dim == 3:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        z = point_coo[2].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, z, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif dim == 2:
        x = point_coo[0].flatten()
        y = point_coo[1].flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, y, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:
        x = point_coo.flatten()
        ones = np.ones(rows * (dim + 1))

        lcl = np.column_stack([x, ones])
        lcl = np.reshape(lcl, newshape=[rows, dim + 1, dim + 1])

        p_vals = np.reshape(point_val, newshape=[rows, dim + 1, 1])

        coeff = np.empty([rows, dim + 1])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff


def interpolate_p2(point_val: np.ndarray, point_coo: np.ndarray) -> np.ndarray:
    """Performs a linear local interpolation of a P2 FEM element given.

    Parameters:
        point_val : NumPy nd-array of shape (sd.num_cells x dof)
            Pressures values at the Lagrangian nodes.
        point_coo : NumPy nd-array of shape (sd.dim x sd.num_cells x dof)
            Coordinates of the Lagrangian nodes. In the case of embedded entities,
            the points should correspond to the rotated coordinates.

    Raises:
        - ValueError if:
            Number of cols of point_val is different from 10 (3D), 6 (2D), or 3 (1D).

    Returns:
        Numpy nd-array of shape (sd.num_cells x dof)

            Coefficients of the cell-wise P2 polynomial satisfying:
            c0x^2 + c1x + c2                                                    (1D),
            c0x^2 + c1xy + c2x + c3y^2 + c4y + c5                               (2D),
            c0x^2 + c1xy + c2xz + c3x + c4y^2 + c5yz + c6y + c7z^2 + c8z + c9   (3D).

    """
    # Local degrees of freedom according to simplex dimensionality
    DOF_3D: int = 10
    DOF_2D: int = 6
    DOF_1D: int = 3

    # Get rows, cols, and dimensionality
    rows = point_val.shape[0]  # number of cells
    cols = point_val.shape[1]  # number of Lagrangian nodes per cell
    if cols == DOF_3D:
        dim = 3
    elif cols == DOF_2D:
        dim = 2
    elif cols == DOF_1D:
        dim = 1
    else:
        raise ValueError("P2 reconstruction only valid for 1D, 2D, and 3D.")

    if dim == 3:
        x = point_coo[0].flatten()  # (nc * 10,)
        y = point_coo[1].flatten()  # (nc * 10,)
        z = point_coo[2].flatten()  # (nc * 10,)
        ones = np.ones(rows * DOF_3D)

        lcl = np.column_stack([x ** 2, x * y, x * z, x, y ** 2, y * z, y, z ** 2, z, ones])
        lcl = np.reshape(lcl, [rows, DOF_3D, DOF_3D])

        p_vals = np.reshape(point_val, [rows, DOF_3D, 1])

        coeff = np.empty([rows, DOF_3D])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    elif dim == 2:
        x = point_coo[0].flatten()  # (nc * 6,)
        y = point_coo[1].flatten()  # (nc * 6,)
        ones = np.ones(rows * DOF_2D)

        lcl = np.column_stack([x ** 2, x * y, x, y ** 2, y, ones])
        lcl = np.reshape(lcl, [rows, DOF_2D, DOF_2D])

        p_vals = np.reshape(point_val, [rows, DOF_2D, 1])

        coeff = np.empty([rows, DOF_2D])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    else:
        x = point_coo.flatten()  # (nc * 3,)
        ones = np.ones(rows * DOF_1D)

        lcl = np.column_stack([x ** 2, x, ones])
        lcl = np.reshape(lcl, [rows, DOF_1D, DOF_1D])

        p_vals = np.reshape(point_val, [rows, DOF_1D, 1])

        coeff = np.empty([rows, DOF_1D])
        for cell in range(rows):
            coeff[cell] = (np.dot(np.linalg.inv(lcl[cell]), p_vals[cell])).T

    return coeff

def poly2col(polynomial: np.ndarray) -> list:
    """
    Returns the coefficients (columns) of a polynomial in the form of a list.

    Parameters
    ----------
        polynomial (np.ndarray): Coefficients, i.e., the ones obtained from interpolate_P1. The
            expected shape is: rows x num_lagrangian_nodes.

    Returns
    -------
        List
            Coefficients stored in column-wise format.

    """
    rows = polynomial.shape[0]
    cols = polynomial.shape[1]
    coeff_list = []

    for col in range(cols):
        coeff_list.append(polynomial[:, col].reshape(rows, 1))

    return coeff_list
