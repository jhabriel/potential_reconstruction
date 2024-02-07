import porepy as pp
import numpy as np
from scipy.spatial import Delaunay
from porepy import TriangleGrid
from typing import Optional


#%% Regular structured grid

nx = np.array([10, 10])
physdims = np.ones(2)
g = pp.StructuredTriangleGrid(nx, physdims, name="Regular structured triangular grid")
g.compute_geometry()
pp.plot_grid(g, plot_2d=True, alpha=0, title="", )
pp.save_img(grid=g, name="regular_grid", plot_2d=True, alpha=0.75, title="")

#%% Irregular structured grid

nx = np.array([10, 10])
physdims = np.ones(2)

# x and y linspaces
x = np.linspace(0, physdims[0], nx[0] + 1)
y = np.linspace(0, physdims[1], nx[1] + 1)

# Node coordinates
x_coord, y_coord = np.meshgrid(x, y)
points = np.vstack((x_coord.ravel(), y_coord.ravel()))

# Create triangulation
triangulation = Delaunay(points.T)

sd = pp.TriangleGrid(
    triangulation.points.T,
    triangulation.simplices.T,
    name="Irregular structured triangular grid"
)
pp.plot_grid(sd, plot_2d=True)
pp.save_img(grid=sd, name="irregular_grid", plot_2d=True, alpha=0.75, title="")


#%% Perturbed grid

cell_size = 0.1
epsilon = cell_size / 2

domain = pp.Domain({"xmax": 1.0, "ymax": 1.0})
fn = pp.create_fracture_network([], domain=domain)
mdg = pp.create_mdg("simplex", {"cell_size": cell_size}, fn)
sd = mdg.subdomains()[0]

# Perturb nodes in the x-coordinates
np.random.seed(42)
int_x_nodes = np.where((sd.nodes[0] != 0) & (sd.nodes[0] != 1))[0]
sd.nodes[0][int_x_nodes] += epsilon * np.random.rand(int_x_nodes.size)

np.random.seed(42)
int_y_nodes = np.where((sd.nodes[1] != 0) & (sd.nodes[1] != 1))[0]
sd.nodes[1][int_y_nodes] += epsilon * np.random.rand(int_y_nodes.size)

sd.compute_geometry()
pp.save_img(grid=sd, name="pertubed_unstructured", plot_2d=True, alpha=0.75, title="")
