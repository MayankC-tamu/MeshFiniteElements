import meshio
import numpy as np

n = 20

vertices = [(np.cos(2*np.pi*i/n), np.sin(2*np.pi*i/n), 0) for i in range(n)]
vertices.append((0, 0, 0.1))

faces = []
for i in range(n):
    faces.append([i, (i+1) % n, n])

meshio.write_points_cells(
    f"pyramid-{n}.obj",
    vertices,
    {"triangle": faces}
)

