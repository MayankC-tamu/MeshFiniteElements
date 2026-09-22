import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from math_surfaces_2 import *


# ============================================================
# Abstract implicit surface container
# ============================================================

@dataclass
class ImplicitSurface:
    """
    Represents an implicit surface:

        f(p) = level

    where p = np.array([x, y, z]).
    """
    name: str
    function: callable
    bounds: tuple
    level: float = 0.0


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray


# ============================================================
# Numerical gradient and projection
# ============================================================

def numerical_gradient(surface_fn, p, eps=1e-6):
    """
    Central-difference numerical gradient of f at point p.
    """
    grad = np.zeros(3, dtype=float)

    for i in range(3):
        dp = np.zeros(3)
        dp[i] = eps

        grad[i] = (
            surface_fn(p + dp) - surface_fn(p - dp)
        ) / (2.0 * eps)

    return grad


def project_point_to_implicit(
    p,
    surface_fn,
    level=0.0,
    max_iters=20,
    tol=1e-8,
    max_step=0.05,
):
    """
    Project a point p onto the implicit surface:

        f(p) = level

    using damped Newton projection.
    """
    p = np.asarray(p, dtype=float).copy()

    for _ in range(max_iters):
        value = surface_fn(p) - level

        if abs(value) < tol:
            break

        grad = numerical_gradient(surface_fn, p)
        grad_norm_sq = np.dot(grad, grad)

        if grad_norm_sq < 1e-14:
            break

        step = value * grad / grad_norm_sq

        step_norm = np.linalg.norm(step)
        if step_norm > max_step:
            step *= max_step / step_norm

        p -= step

    return p


def project_mesh_to_implicit(
    mesh,
    surface,
    max_iters=20,
    max_step=0.05,
):
    """
    Project all mesh vertices back onto the implicit surface.
    """
    vertices = mesh.vertices.copy()

    for i in range(len(vertices)):
        vertices[i] = project_point_to_implicit(
            vertices[i],
            surface.function,
            level=surface.level,
            max_iters=max_iters,
            max_step=max_step,
        )

    return Mesh(vertices, mesh.faces.copy())


# ============================================================
# Marching Cubes: implicit surface to triangle mesh
# ============================================================

def implicit_surface_to_triangle_mesh(
    surface,
    resolution=96,
):
    """
    Converts an implicit surface f(x,y,z)=level into a triangle mesh
    using Marching Cubes.

    Parameters
    ----------
    surface:
        ImplicitSurface object.

    resolution:
        Either an int or a tuple (nx, ny, nz).

    Returns
    -------
    Mesh with triangular faces.
    """
    if isinstance(resolution, int):
        resolution = (resolution, resolution, resolution)

    nx, ny, nz = resolution

    (xmin, xmax), (ymin, ymax), (zmin, zmax) = surface.bounds

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(zmin, zmax, nz)

    values = np.empty((nx, ny, nz), dtype=float)

    print(f"Sampling implicit surface '{surface.name}'...")
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                values[i, j, k] = surface.function(np.array([x, y, z]))

    vmin = values.min()
    vmax = values.max()

    if not (vmin <= surface.level <= vmax):
        raise ValueError(
            f"Surface level {surface.level} was not found inside the sampled box. "
            f"Value range was [{vmin}, {vmax}]. Try larger bounds."
        )

    spacing = (
        (xmax - xmin) / (nx - 1),
        (ymax - ymin) / (ny - 1),
        (zmax - zmin) / (nz - 1),
    )

    print("Running Marching Cubes...")
    vertices, faces, normals, vals = measure.marching_cubes(
        values,
        level=surface.level,
        spacing=spacing,
    )

    # Marching Cubes coordinates start at zero, so shift to real coordinates.
    vertices[:, 0] += xmin
    vertices[:, 1] += ymin
    vertices[:, 2] += zmin

    return Mesh(vertices=vertices, faces=faces.astype(int))


# ============================================================
# Catmull-Clark subdivision
# ============================================================

def catmull_clark(mesh):
    """
    Catmull-Clark subdivision.

    Input faces can be triangles, quads, or general polygons.
    Output faces are always quads.

    One Catmull-Clark step converts a triangle mesh into a quadrilateral mesh.
    """
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = [list(map(int, f)) for f in mesh.faces]

    # Face points
    face_points = np.array([
        np.mean(vertices[face], axis=0)
        for face in faces
    ])

    # Edge to adjacent faces
    edge_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)

        for i in range(n):
            a = face[i]
            b = face[(i + 1) % n]
            e = tuple(sorted((a, b)))
            edge_faces[e].append(fi)

    # Check for nonmanifold edges
    nonmanifold_edges = [
        e for e, adj in edge_faces.items()
        if len(adj) > 2
    ]

    if nonmanifold_edges:
        raise ValueError(
            f"Mesh has {len(nonmanifold_edges)} nonmanifold edges. "
            "Cannot safely apply Catmull-Clark."
        )

    # Edge points
    edge_points = []
    edge_key_to_index = {}

    for idx, (edge, adjacent_faces) in enumerate(edge_faces.items()):
        a, b = edge

        if len(adjacent_faces) == 2:
            f0 = adjacent_faces[0]
            f1 = adjacent_faces[1]

            edge_point = (
                vertices[a]
                + vertices[b]
                + face_points[f0]
                + face_points[f1]
            ) / 4.0
        else:
            # Boundary case. A closed surface should not need this,
            # but this makes the function more general.
            edge_point = 0.5 * (vertices[a] + vertices[b])

        edge_key_to_index[edge] = idx
        edge_points.append(edge_point)

    edge_points = np.asarray(edge_points)

    # Vertex adjacency
    vertex_faces = [[] for _ in range(len(vertices))]
    vertex_edges = [[] for _ in range(len(vertices))]

    for fi, face in enumerate(faces):
        for v in face:
            vertex_faces[v].append(fi)

    for edge in edge_faces:
        a, b = edge
        vertex_edges[a].append(edge)
        vertex_edges[b].append(edge)

    # Updated old vertices
    updated_vertices = np.zeros_like(vertices)

    for vi, V in enumerate(vertices):
        incident_faces = vertex_faces[vi]
        incident_edges = vertex_edges[vi]

        n = len(incident_edges)

        if n == 0:
            updated_vertices[vi] = V
            continue

        F = np.mean(face_points[incident_faces], axis=0)

        edge_midpoints = []
        for edge in incident_edges:
            a, b = edge
            edge_midpoints.append(0.5 * (vertices[a] + vertices[b]))

        R = np.mean(edge_midpoints, axis=0)

        updated_vertices[vi] = (
            F + 2.0 * R + (n - 3.0) * V
        ) / n

    # New vertex indexing
    face_offset = 0
    edge_offset = len(face_points)
    vertex_offset = len(face_points) + len(edge_points)

    new_vertices = np.vstack([
        face_points,
        edge_points,
        updated_vertices,
    ])

    # New quad faces
    new_faces = []

    for fi, face in enumerate(faces):
        m = len(face)
        face_point_index = face_offset + fi

        for i in range(m):
            v_current = face[i]
            v_prev = face[(i - 1) % m]
            v_next = face[(i + 1) % m]

            current_vertex_index = vertex_offset + v_current

            edge_prev = tuple(sorted((v_prev, v_current)))
            edge_next = tuple(sorted((v_current, v_next)))

            edge_prev_index = edge_offset + edge_key_to_index[edge_prev]
            edge_next_index = edge_offset + edge_key_to_index[edge_next]

            # Quad around the original vertex
            new_faces.append([
                current_vertex_index,
                edge_next_index,
                face_point_index,
                edge_prev_index,
            ])

    return Mesh(
        vertices=np.asarray(new_vertices, dtype=float),
        faces=np.asarray(new_faces, dtype=int),
    )


# ============================================================
# Mesh topology helpers
# ============================================================

def mesh_topology_stats(mesh):
    """
    Compute basic mesh topology information.
    For a closed orientable connected surface:

        genus = (2 - chi) / 2
    """
    vertices = mesh.vertices
    faces = mesh.faces

    edge_faces = defaultdict(list)

    for fi, face in enumerate(faces):
        n = len(face)

        for i in range(n):
            a = int(face[i])
            b = int(face[(i + 1) % n])
            e = tuple(sorted((a, b)))
            edge_faces[e].append(fi)

    V = len(vertices)
    E = len(edge_faces)
    F = len(faces)

    chi = V - E + F

    boundary_edges = [
        e for e, adj in edge_faces.items()
        if len(adj) == 1
    ]

    nonmanifold_edges = [
        e for e, adj in edge_faces.items()
        if len(adj) > 2
    ]

    is_closed = len(boundary_edges) == 0
    is_edge_manifold = len(nonmanifold_edges) == 0

    genus = None
    if is_closed and is_edge_manifold:
        genus_float = (2 - chi) / 2

        if abs(genus_float - round(genus_float)) < 1e-8:
            genus = int(round(genus_float))
        else:
            genus = genus_float

    face_sizes = [len(f) for f in faces]

    return {
        "vertices": V,
        "edges": E,
        "faces": F,
        "euler_characteristic": chi,
        "genus": genus,
        "is_closed": is_closed,
        "is_edge_manifold": is_edge_manifold,
        "boundary_edges": len(boundary_edges),
        "nonmanifold_edges": len(nonmanifold_edges),
        "all_quads": all(s == 4 for s in face_sizes),
        "min_face_size": min(face_sizes),
        "max_face_size": max(face_sizes),
    }


def print_mesh_stats(mesh, label):
    stats = mesh_topology_stats(mesh)

    print(f"\n{label}")
    print("-" * len(label))

    for key, value in stats.items():
        print(f"{key}: {value}")


# ============================================================
# Plotting
# ============================================================

def plot_mesh(
    mesh,
    title="Mesh",
    face_color="cyan",
    edge_color="black",
    alpha=0.75,
    line_width=0.2,
):
    """
    Plot triangle or quadrilateral mesh.
    """
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    polygons = [vertices[face] for face in faces]

    collection = Poly3DCollection(
        polygons,
        alpha=alpha,
        linewidths=line_width,
    )

    collection.set_facecolor(face_color)
    collection.set_edgecolor(edge_color)

    ax.add_collection3d(collection)

    # Equal-ish axes
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)

    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    #plt.show()


# ============================================================
# Abstract mesh-generation pipeline
# ============================================================

def create_mesh_from_implicit_surface(
    surface,
    resolution=96,
    make_quad_mesh=True,
    project_after_quad=True,
    projection_max_step=0.05,
):
    """
    General pipeline:

        implicit surface
            -> Marching Cubes triangle mesh
            -> optional Catmull-Clark quad mesh
            -> optional projection to exact surface

    This function does not know about torus-specific geometry.
    It only knows about generic implicit surfaces.
    """
    tri_mesh = implicit_surface_to_triangle_mesh(
        surface,
        resolution=resolution,
    )

    print_mesh_stats(tri_mesh, "Initial triangle mesh")

    if not make_quad_mesh:
        return tri_mesh

    quad_mesh = catmull_clark(tri_mesh)

    print_mesh_stats(quad_mesh, "After Catmull-Clark quad conversion")

    if project_after_quad:
        quad_mesh = project_mesh_to_implicit(
            quad_mesh,
            surface,
            max_step=projection_max_step,
        )

        print_mesh_stats(quad_mesh, "After projection back to implicit surface")

    return quad_mesh


def run_surface_tests(
    resolution=48,
    make_quad_mesh=True,
    project_after_quad=True,
    plot=True,
):
    """
    Test multiple implicit surfaces using the new abstract pipeline.
    """

    meshes = {}

    for item in test_surfaces:
        print()
        print("=" * 80)
        print(f"Generating surface: {item['name']}")
        print(f"Expected genus: {item['expected_genus']}")
        print("=" * 80)

        surface = ImplicitSurface(
            name=item["name"],
            function=item["function"],
            bounds=item["bounds"],
            level=0.0,
        )

        mesh = create_mesh_from_implicit_surface(
            surface,
            resolution=item["resolution"],
            make_quad_mesh=make_quad_mesh,
            project_after_quad=project_after_quad,
            projection_max_step=item["projection_max_step"],
        )

        if plot:
            plot_mesh(
                mesh,
                title=f"{item['name']}",
                face_color="cyan",
                edge_color="black",
                alpha=0.75,
                line_width=0.25,
            )

        meshes[item["name"]] = mesh

    return meshes

#
#
#

def to_obj(mesh, filename):
    """
    Save mesh to a simple .obj file.
    """
    with open(filename, "w") as f:
        for v in mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for face in mesh.faces:
            # OBJ format uses 1-based indexing
            face_str = " ".join(str(idx + 1) for idx in face)
            f.write(f"f {face_str}\n")

    print(f"Mesh saved to {filename}")

# ============================================================
# Main
# ============================================================

def main():
    # # The only torus-specific part of the code is here.
    # torus = ImplicitSurface(
    #     name="tanglecube",
    #     function=X_func(),
    #     bounds=(
    #             (-1.5, 1.5),
    #             (-1.5, 1.5),
    #             (-1.5, 1.5),
    #     ),
    #     level=0.0,
    # )

    # mesh = create_mesh_from_implicit_surface(
    #     torus,
    #     resolution=32,
    #     make_quad_mesh=True,
    #     project_after_quad=True,
    #     projection_max_step=0.03,
    # )

    # plot_mesh(
    #     mesh,
    #     title="Quadrilateral Mesh of Torus",
    #     face_color="cyan",
    #     edge_color="black",
    #     alpha=0.75,
    #     line_width=0.25,
    # )

    # vertices = mesh.vertices

    # save_filename = f"X_{len(vertices)}.obj"
    # to_obj(mesh, save_filename)

    # return mesh

    meshes = run_surface_tests(
        resolution=96,
        make_quad_mesh=True,
        project_after_quad=True,
        plot=True,
    )

    # return meshes
    return meshes




if __name__ == "__main__":
    mesh = main()