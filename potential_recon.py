"""
This module contains functions that performs pressure reconstructions. The basic idea is
to apply an interpolator of the type G: P_0 -> P_1 to enhance the regularity of the
cell-centered (P0) pressures and obtain P_1, energy-conforming potentials.
"""
from __future__ import annotations

import numpy as np

import porepy as pp
import quadpy
import scipy.sparse as sps

from utils import get_quadpy_elements


def r2c(array: np.ndarray) -> np.ndarray:
    """Reshape a 1d array into a column vector"""
    return array.reshape(-1, 1)


def patchwise_p1(
    sd: pp.Grid, sd_data: dict, bg_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pressure reconstruction using average of P0(K) potentials over patches [1].

    Reference:
        Cochez-Dhondt, S., Nicaise, S., & Repin, S. (2009). A posteriori error estimates
        for finite volume approximations. Mathematical Modelling of Natural Phenomena,
        4(1), 106-122.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        sd_data: dict
            Subdomain data dictionary.
        bg_data: dict
            Boundary grid data dictionary.

    Returns:
        2-tuple of numpy arrays containing the values and Lagrangian coordinates of
        the reconstructed pressure for all elements of the subdomain grid.

    """
    # Retrieve finite volume cell-centered pressures
    p_cc = sd_data["estimates"]["fv_sd_pressure"]

    # Retrieving topological data
    nc = sd.num_cells
    nf = sd.num_faces

    # Perform reconstruction
    cell_nodes = sd.cell_nodes()
    cell_nodes_volume = cell_nodes * sps.dia_matrix((sd.cell_volumes, 0), (nc, nc))
    cell_nodes_pressure = cell_nodes * sps.dia_matrix((p_cc, 0), (nc, nc))
    numerator_sparse = cell_nodes_volume.multiply(cell_nodes_pressure)
    numerator = np.array(numerator_sparse.sum(axis=1)).flatten()
    denominator = np.array(cell_nodes_volume.sum(axis=1)).flatten()
    nodal_pressures = numerator / denominator

    # Treatment of boundary conditions
    bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

    bc_dir_values = np.zeros(sd.num_faces)
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, sd.tags["domain_boundary_faces"]
    )
    bc_pressure = bg_data[pp.ITERATE_SOLUTIONS]["pressure"][0]
    bg_dir_filter = bg_data[pp.ITERATE_SOLUTIONS]["bc_values_darcy_filter_dir"][0] == 1
    bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = sd.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_dir_values[external_dirichlet_boundary]
    node_val_dir = sd.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Export Lagrangian nodes and coordinates
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(sd.num_cells, sd.dim + 1)
    point_val = nodal_pressures[nodes_of_cell]
    point_coo = sd.nodes[:, nodes_of_cell]

    return point_val, point_coo


def keilegavlen_p1(
    sd: pp.Grid, sd_data: dict, bg_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Pressure reconstruction using the inverse of the numerical fluxes.

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        sd_data: dict
            Subdomain data dictionary.
        bg_data: dict
            Boundary grid data dictionary.

    Returns:
        2-tuple of numpy arrays containing the values and Lagrangian coordinates of
        the reconstructed pressure for all elements of the subdomain grid.

    """
    # Retrieve finite volume cell-centered pressures
    p_cc = sd_data["estimates"]["fv_sd_pressure"]
    assert p_cc.size == sd.num_cells

    # Retrieve topological data
    nc = sd.num_cells
    nf = sd.num_faces
    nn = sd.num_nodes

    # Perform reconstruction
    cell_nodes = sd.cell_nodes()
    cell_node_volumes = cell_nodes * sps.dia_matrix(
        arg1=(sd.cell_volumes, 0), shape=(nc, nc)
    )
    sum_cell_nodes = cell_node_volumes * np.ones(nc)
    cell_nodes_scaled = (
        sps.dia_matrix(arg1=(1.0 / sum_cell_nodes, 0), shape=(nn, nn))
        * cell_node_volumes
    )

    # Retrieve reconstructed velocities
    coeff = sd_data["estimates"]["recon_sd_flux"]
    if sd.dim == 3:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
                coeff[:, 0] * sd.cell_centers[1] + coeff[:, 2],
                coeff[:, 0] * sd.cell_centers[2] + coeff[:, 3],
            ]
        )
    elif sd.dim == 2:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
                coeff[:, 0] * sd.cell_centers[1] + coeff[:, 2],
            ]
        )
    else:
        proj_flux = np.array(
            [
                coeff[:, 0] * sd.cell_centers[0] + coeff[:, 1],
            ]
        )

    # Obtain local gradients
    loc_grad = np.zeros((sd.dim, nc))
    perm = sd_data[pp.PARAMETERS]["flow"]["second_order_tensor"].values
    for ci in range(nc):
        loc_grad[: sd.dim, ci] = -np.linalg.inv(perm[: sd.dim, : sd.dim, ci]).dot(
            proj_flux[:, ci]
        )

    # Obtaining nodal pressures
    cell_nodes_map = sps.find(sd.cell_nodes().T)[1]
    cell_node_matrix = cell_nodes_map.reshape(np.array([sd.num_cells, sd.dim + 1]))
    nodal_pressures = np.zeros(nn)

    for col in range(sd.dim + 1):
        nodes = cell_node_matrix[:, col]
        dist = sd.nodes[: sd.dim, nodes] - sd.cell_centers[: sd.dim]
        scaling = cell_nodes_scaled[nodes, np.arange(nc)]
        contribution = (
            np.asarray(scaling) * (p_cc + np.sum(dist * loc_grad, axis=0))
        ).ravel()
        nodal_pressures += np.bincount(nodes, weights=contribution, minlength=nn)

    # Treatment of boundary conditions
    bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

    bc_dir_values = np.zeros(sd.num_faces)
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, sd.tags["domain_boundary_faces"]
    )
    bc_pressure = bg_data[pp.ITERATE_SOLUTIONS]["pressure"][0]
    bg_dir_filter = bg_data[pp.ITERATE_SOLUTIONS]["bc_values_darcy_filter_dir"][0] == 1
    bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]

    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = sd.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_dir_values[external_dirichlet_boundary]
    node_val_dir = sd.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    nodal_pressures[is_dir_node] = node_val_dir[is_dir_node]

    # Export Lagrangian nodes and coordinates
    nodes_of_cell = sps.find(sd.cell_nodes().T)[1].reshape(sd.num_cells, sd.dim + 1)
    point_val = nodal_pressures[nodes_of_cell]
    point_coo = sd.nodes[:, nodes_of_cell]

    return point_val, point_coo


def vohralik_p2(
    sd: pp.Grid, sd_data: dict, bg_data: dict
) -> tuple[np.ndarray, np.ndarray]:
    """TODO ---->

    Parameters:
        sd: pp.Grid
            Subdomain grid.
        sd_data: dict
            Subdomain data dictionary.
        bg_data:
            Boundary grid data dictionary.

    Returns:
        2-tuple of numpy arrays containing the values and Lagrangian coordinates of
        the reconstructed pressure for all elements of the subdomain grid.
    """
    # Retrieve cell-centered pressures
    p_cc = sd_data["estimates"]["fv_sd_pressure"]

    # Retrieve RT0 flux and obtain coefficients
    u = sd_data["estimates"]["recon_sd_flux"]

    # Retrieve permeability and compute its inverse
    perm = sd_data[pp.PARAMETERS]["flow"]["second_order_tensor"].values

    # Loop through all cells and compute the vector r
    s = np.zeros((sd.num_cells, 6))
    for ci in range(sd.num_cells):
        # Local permeability tensor
        K = perm[: sd.dim, : sd.dim, ci]
        Kxx = K[0][0]
        Kxy = K[0][1]
        Kyy = K[1][1]

        # Retrieve components of the RT0 local flux field
        a = u[ci][0]
        b = u[ci][1]
        c = u[ci][2]

        # Compute components of vector post-processed pressure
        s[ci][0] = (a * Kyy) / (2 * (Kxy ** 2 - Kxx * Kyy))  # x^2
        s[ci][1] = (a * Kxy) / (Kxx * Kyy - Kxy ** 2)  # xy
        s[ci][2] = (Kxy * c - Kyy * b) / (Kxx * Kyy - Kxy ** 2)  # x
        s[ci][3] = (a * Kxx) / (2 * (Kxy ** 2 - Kxx * Kyy))  # y^2
        s[ci][4] = (Kxx * c - Kxy * b) / (Kxy ** 2 - Kxx * Kyy)  # y

    # Get quadpy elements and declare integration method
    elements = get_quadpy_elements(sd)
    method = quadpy.t2.get_good_scheme(10)

    # To obtain the constant c_5, we solve  c_5 = p_h - 1/|K| (gamma(x, y), 1)_K,
    # where s(x, y) = gamma(x, y) + c_5.
    def integrand(x):
        int_0 = r2c(s[:, 0]) * x[0] ** 2
        int_1 = r2c(s[:, 1]) * x[0] * x[1]
        int_2 = r2c(s[:, 2]) * x[0]
        int_3 = r2c(s[:, 3]) * x[1] ** 2
        int_4 = r2c(s[:, 4]) * x[1]
        return int_0 + int_1 + int_2 + int_3 + int_4
    integral = method.integrate(integrand, elements)

    # Now, we can compute the constant C, one per cell.
    s[:, 5] = p_cc - integral / sd.cell_volumes

    sd_data["estimates"]["p_recon_post_p2"] = s

    # The following step is now to apply the Oswald interpolator

    # Sanity check
    if not s.shape == (sd.num_cells, 6):
        raise ValueError("Wrong shape of P2 polynomial.")

    # Abbreviations
    dim = sd.dim
    nn = sd.num_nodes
    nf = sd.num_faces
    nc = sd.num_cells

    # Mappings
    cell_faces_map = sps.find(sd.cell_faces.T)[1]
    cell_nodes_map = sps.find(sd.cell_nodes().T)[1]
    faces_of_cell = cell_faces_map.reshape(nc, dim + 1)
    nodes_of_cell = cell_nodes_map.reshape(nc, dim + 1)

    # Treatment of the nodes
    # Evaluate post-processed pressure at the nodes
    nodes_p = np.zeros([nc, 3])
    nx = sd.nodes[0][nodes_of_cell]  # local node x-coordinates
    ny = sd.nodes[1][nodes_of_cell]  # local node y-coordinates

    # Compute node pressures
    for col in range(dim + 1):
        nodes_p[:, col] = (
                s[:, 0] * nx[:, col] ** 2  # c0x^2
                + s[:, 1] * nx[:, col] * ny[:, col]  # c1xy
                + s[:, 2] * nx[:, col]  # c2x
                + s[:, 3] * ny[:, col] ** 2  # c3y^2
                + s[:, 4] * ny[:, col]  # c4y
                + s[:, 5]  # c5
        )

    # Average nodal pressure
    node_cardinality = np.bincount(cell_nodes_map)
    node_pressure = np.zeros(nn)
    for col in range(dim + 1):
        node_pressure += np.bincount(
            nodes_of_cell[:, col], weights=nodes_p[:, col], minlength=nn
        )
    node_pressure /= node_cardinality

    # Treatment of the faces
    # Evaluate post-processed pressure at the face-centers
    faces_p = np.zeros([nc, 3])
    fx = sd.face_centers[0][faces_of_cell]  # local face-center x-coordinates
    fy = sd.face_centers[1][faces_of_cell]  # local face-center y-coordinates

    for col in range(3):
        faces_p[:, col] = (
                s[:, 0] * fx[:, col] ** 2  # c0x^2
                + s[:, 1] * fx[:, col] * fy[:, col]  # c1xy
                + s[:, 2] * fx[:, col]  # c2x
                + s[:, 3] * fy[:, col] ** 2  # c3y^2
                + s[:, 4] * fy[:, col]  # c4x
                + s[:, 5]  # c5
        )

    # Average face pressure
    face_cardinality = np.bincount(cell_faces_map)
    face_pressure = np.zeros(nf)
    for col in range(3):
        face_pressure += np.bincount(
            faces_of_cell[:, col], weights=faces_p[:, col], minlength=nf
        )
    face_pressure /= face_cardinality

    # Treatment of the boundary points
    bc = sd_data[pp.PARAMETERS]["flow"]["bc"]

    bc_dir_values = np.zeros(nf)
    external_dirichlet_boundary = np.logical_and(
        bc.is_dir, sd.tags["domain_boundary_faces"]
    )
    bc_pressure = bg_data[pp.ITERATE_SOLUTIONS]["pressure"][0]
    bg_dir_filter = bg_data[pp.ITERATE_SOLUTIONS]["bc_values_darcy_filter_dir"][0] == 1
    bc_dir_values[external_dirichlet_boundary] = bc_pressure[bg_dir_filter]
    face_pressure[external_dirichlet_boundary] = (
        bc_dir_values[external_dirichlet_boundary]
    )

    # Boundary values at the nodes
    face_vec = np.zeros(nf)
    face_vec[external_dirichlet_boundary] = 1
    num_dir_face_of_node = sd.face_nodes * face_vec
    is_dir_node = num_dir_face_of_node > 0
    face_vec *= 0
    face_vec[external_dirichlet_boundary] = bc_dir_values[external_dirichlet_boundary]
    node_val_dir = sd.face_nodes * face_vec
    node_val_dir[is_dir_node] /= num_dir_face_of_node[is_dir_node]
    node_pressure[is_dir_node] = node_val_dir[is_dir_node]

    # Prepare for exporting
    point_val = np.column_stack(
        [node_pressure[nodes_of_cell], face_pressure[faces_of_cell]]
    )
    point_coo = np.empty([dim, nc, 6])
    point_coo[0] = np.column_stack([nx, fx])
    point_coo[1] = np.column_stack([ny, fy])

    return point_val, point_coo

