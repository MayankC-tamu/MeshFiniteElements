import numpy as np
import matplotlib.pyplot as plt
import tqdm

from fem_tester import FEM_V, FEM_V2, FEM_Q3

vertices = [9, 25, 81, 289, 1089, 4225, 16641]
vertex_sqrt = [3, 5, 9, 17, 33, 65, 129]

division = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]


# vertices = [16, 49, 169, 625, 2401]
 # vertex_sqrt = [4,7,13,25,49]
# nx_initial = 3
# num_subdivisions = [0, 1, 2, 3, 4]  # Corresponding to mesh refinement levels 0 to 4
#division = 2.0 / (nx_initial * 2**np.array(num_subdivisions))

for i in range(6):

    X, Y = np.meshgrid(
        np.linspace(0, 1, vertex_sqrt[i]),
        np.linspace(0, 1, vertex_sqrt[i])
    )

    print("Mesh Refinement Level:", i)

    V = FEM_V2('./solutions/planar-franke-O2/' + str(i) + '.csv', division[i])

    grid = np.linspace(0, 1, 401)
    x, y = np.meshgrid(grid, grid)

    u = np.zeros_like(x)

    with tqdm.tqdm(total=u.size, desc=f"creating points {i}") as pbar:
        for j in range(x.shape[0]):
            for k in range(x.shape[1]):
                u[j, k] = V((x[j, k], y[j, k]))
                pbar.update(1)

    Vertices = np.zeros_like(X)

    for j in range(X.shape[0]):
        for k in range(X.shape[1]):
            Vertices[j, k] = V((X[j, k], Y[j, k]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    surf = ax.plot_surface(
        x, y, u,
        cmap='viridis',
        rstride=1,
        cstride=1,
        alpha=0.9
    )

    wire = ax.plot_wireframe(
        X, Y, Vertices,
        color='black',
        linewidth=1
    )

    ax.set_title(f'FEM Solution Wireframe Plot - Vertices: {vertices[i]}')

    plt.savefig('./solutions/planar-franke-O2/surface_plot_' + str(i) + '.png')
    plt.close()