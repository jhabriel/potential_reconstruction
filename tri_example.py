import numpy as np
import matplotlib.pyplot as plt


def generate_triangular_grid(n):
    """
    Generate a structured triangular grid within a square domain.

    Parameters:
        n (int): Number of divisions along each side of the square domain.

    Returns:
        x (ndarray): x-coordinates of the vertices.
        y (ndarray): y-coordinates of the vertices.
        triangles (ndarray): Indices of vertices forming triangles.
    """
    x = np.linspace(0, 1, n + 1)
    y = np.linspace(0, 1, n + 1)
    X, Y = np.meshgrid(x, y)

    x = X.flatten()
    y = Y.flatten()

    triangles = []
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                v0 = i * (n + 1) + j
                v1 = v0 + n + 1
                v2 = v1 + 1
                triangles.append([v0, v1, v2])
            else:
                v0 = i * (n + 1) + j
                v1 = v0 + 1
                v2 = v1 + n
                triangles.append([v0, v1, v2])

                v0 = i * (n + 1) + j
                v1 = v0 + n + 1
                v2 = v1 + 1
                triangles.append([v0, v1, v2])

    return x, y, np.array(triangles)


def plot_triangular_grid(x, y, triangles):
    """
    Plot the triangular grid.

    Parameters:
        x (ndarray): x-coordinates of the vertices.
        y (ndarray): y-coordinates of the vertices.
        triangles (ndarray): Indices of vertices forming triangles.
    """
    plt.figure(figsize=(8, 8))
    for triangle in triangles:
        plt.fill(x[triangle], y[triangle], edgecolor='k', fill=False)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Structured Triangular Grid')
    plt.show()


# Example usage:
n = 5  # Number of divisions along each side of the square domain
x, y, triangles = generate_triangular_grid(n)
plot_triangular_grid(x, y, triangles)
