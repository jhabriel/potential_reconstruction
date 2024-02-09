import numpy as np
import porepy as pp
from scipy.spatial import Delaunay


class MeshGenerator:
    def __init__(
        self,
        domain: np.ndarray = np.array([1.0, 1.0]),
        mesh_size: float = 0.1,
        dim: int = 2,
    ):
        """Class constructor.

        :param domain:
        :param mesh_size:
        :param dim:
        """
        self.mesh_size = mesh_size
        self.domain = domain
        self.dim = dim

    def regular_structured_simplex(self) -> pp.Grid:
        nx = int(1 / self.mesh_size) * np.ones(self.dim, dtype=int)
        physdims = self.domain
        g = pp.StructuredTriangleGrid(nx, physdims)
        g.compute_geometry()

        return g

    def irregular_structured_simplex(self) -> pp.Grid:
        nx = int(1 / self.mesh_size) * np.ones(self.dim, dtype=int)
        physdims = self.domain

        x = np.linspace(0, physdims[0], nx[0] + 1)
        y = np.linspace(0, physdims[1], nx[1] + 1)

        # Node coordinates
        x_coord, y_coord = np.meshgrid(x, y)
        points = np.vstack((x_coord.ravel(), y_coord.ravel()))

        # Create triangulation
        triangulation = Delaunay(points.T)
        g = pp.TriangleGrid(
            triangulation.points.T,
            triangulation.simplices.T,
        )
        g.compute_geometry()

        return g

    def unstructured_simplex(self, perturb_nodes: bool = True) -> pp.Grid:
        domain = pp.Domain({"xmax": self.domain[0], "ymax": self.domain[1]})
        fn = pp.create_fracture_network([], domain=domain)
        mdg = pp.create_mdg("simplex", {"cell_size": self.mesh_size}, fn)
        sd = mdg.subdomains()[0]

        # Perturb nodes in the x and y coordinates
        if perturb_nodes:
            epsilon = self.mesh_size / 2

            np.random.seed(42)
            int_x_nodes = np.where((sd.nodes[0] != 0) & (sd.nodes[0] != 1))[0]
            sd.nodes[0][int_x_nodes] += epsilon * np.random.rand(int_x_nodes.size)

            np.random.seed(42)
            int_y_nodes = np.where((sd.nodes[1] != 0) & (sd.nodes[1] != 1))[0]
            sd.nodes[1][int_y_nodes] += epsilon * np.random.rand(int_y_nodes.size)

            sd.compute_geometry()

        return sd
