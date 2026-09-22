"""
reconstruct_visual_error.py

Given:
    1. Original coarse quad OBJ
    2. Bicubic solution CSV containing x,y,z,u but no Faces16

This script can:
    - reconstruct bicubic 16-node connectivity Faces16
    - save reconstructed mesh as NPZ
    - export colored OBJ visualization
    - compute nodal and integrated L2 error

Important:
    If the original OBJ has no quad faces, reconstruction is not guaranteed.
    Vertices/edges alone do not uniquely define surface elements.

Dependencies:
    pip install numpy scipy matplotlib
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from numpy.polynomial.legendre import leggauss


# ============================================================
# USER CONFIG
# ============================================================




ORIGINAL_OBJ_PATH = "./meshes/cube/cube2.obj"
BICUBIC_CSV_PATH = "./meshes/cube/cube2_sol.csv"

OUT_NPZ_PATH = "./meshes/cube/cube2_sol.npz"
OUT_COLOR_OBJ_PATH = "./meshes/cube/cube2_sol.obj"

# Main toggles
DO_RECONSTRUCT = True
DO_SAVE_NPZ = True
DO_EXPORT_VISUAL = True
DO_ERROR_TEST = False

# If DO_RECONSTRUCT=False, load this NPZ instead.
LOAD_NPZ_PATH = OUT_NPZ_PATH

# Matching tolerance between reconstructed expected Q3 nodes and CSV nodes.
MATCH_TOL = 1e-8

# Colored OBJ config
COLOR_MAP = "turbo"

# Options:
#   "solution" -> color by numerical U
#   "exact"    -> color by exact solution
#   "error"    -> color by abs(U - exact)
COLOR_BY = "solution"

# Error config
QUAD_ORDER = 10

# For closed-surface Poisson, solution is often unique only up to constant.
NORMALIZE_MEAN_FOR_ERROR = False

# Optional fallback if OBJ has edges but no faces.
# This is not reliable for general meshes.
TRY_INFER_QUADS_FROM_EDGES = False


# ============================================================
# EXACT SOLUTION
# Edit this for your problem
# ============================================================

def exact_u(x, y, z):
    """
    Replace this with your exact/manufactured solution.

    Example:
        return np.sin(np.pi*x) * np.sin(np.pi*y)

    For now this returns 0.
    """
    return 0.0


# ============================================================
# CSV READER
# ============================================================

def read_bicubic_csv(path):
    """
    Reads CSV with one of these common formats:

        x,y,z,u
        x,y,z,u,boundary
        x,y,u
        x,y,u,boundary

    Returns:
        vertices: (N,3)
        U:        (N,)
    """
    vertices = []
    values = []

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError(f"CSV is empty: {path}")

    header = [s.strip().lower() for s in rows[0]]
    has_header = any(h in header for h in ["x", "y", "z", "u"])

    if has_header:
        data_rows = rows[1:]

        def get_col(name):
            return header.index(name) if name in header else None

        ix = get_col("x")
        iy = get_col("y")
        iz = get_col("z")
        iu = get_col("u")

        if ix is None or iy is None or iu is None:
            raise ValueError("CSV header must contain at least x,y,u or x,y,z,u.")

        for row in data_rows:
            if len(row) == 0:
                continue

            try:
                x = float(row[ix])
                y = float(row[iy])
                z = float(row[iz]) if iz is not None else 0.0
                u = float(row[iu])
            except Exception:
                continue

            vertices.append([x, y, z])
            values.append(u)

    else:
        for row in rows:
            numeric = []

            for item in row:
                try:
                    numeric.append(float(item))
                except Exception:
                    pass

            if len(numeric) >= 4:
                x, y, z, u = numeric[:4]
            elif len(numeric) >= 3:
                x, y, u = numeric[:3]
                z = 0.0
            else:
                continue

            vertices.append([x, y, z])
            values.append(u)

    vertices = np.asarray(vertices, dtype=float)
    values = np.asarray(values, dtype=float)

    if len(vertices) == 0:
        raise ValueError(f"No valid rows found in CSV: {path}")

    return vertices, values


# ============================================================
# OBJ READER
# ============================================================

def parse_obj_vertices_faces_edges(path):
    vertices = []
    faces = []
    edges = set()

    def parse_obj_index(token):
        return int(token.split("/")[0]) - 1

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == "v":
                vertices.append([
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3])
                ])

            elif parts[0] == "f":
                face = [parse_obj_index(tok) for tok in parts[1:]]
                faces.append(face)

                for a, b in zip(face, face[1:] + face[:1]):
                    edges.add(tuple(sorted((a, b))))

            elif parts[0] == "l":
                ids = [parse_obj_index(tok) for tok in parts[1:]]

                for a, b in zip(ids[:-1], ids[1:]):
                    edges.add(tuple(sorted((a, b))))

    return np.asarray(vertices, dtype=float), faces, sorted(edges)


def extract_quad_faces(faces):
    quads = []

    for face in faces:
        if len(face) == 4:
            quads.append(face)

    return np.asarray(quads, dtype=int)


def infer_quads_from_edges(vertices, edges):
    """
    Very limited fallback for edge-only OBJ files.
    Finds chordless 4-cycles.

    Warning:
        This is not reliable for general surfaces.
    """
    n = len(vertices)
    adj = [set() for _ in range(n)]

    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    cycles = set()

    def canonical(cyc):
        cyc = list(cyc)
        candidates = []

        for k in range(4):
            candidates.append(tuple(cyc[k:] + cyc[:k]))

        rcyc = list(reversed(cyc))

        for k in range(4):
            candidates.append(tuple(rcyc[k:] + rcyc[:k]))

        return min(candidates)

    for a in range(n):
        for b in adj[a]:
            for c in adj[b]:
                if c == a:
                    continue

                for d in adj[c]:
                    if d == b or d == a:
                        continue

                    if a not in adj[d]:
                        continue

                    if len({a, b, c, d}) != 4:
                        continue

                    # chordless check
                    if c in adj[a]:
                        continue
                    if d in adj[b]:
                        continue

                    cycles.add(canonical([a, b, c, d]))

    return np.asarray(sorted(cycles), dtype=int)


# ============================================================
# RECONSTRUCTION
# ============================================================

def bilinear_quad_point(q0, q1, q2, q3, s, t):
    """
    Quad corner convention:

        q3 ---- q2
        |       |
        q0 ---- q1

    s,t in [0,1].
    """
    return (
        (1.0 - s) * (1.0 - t) * q0
        + s * (1.0 - t) * q1
        + s * t * q2
        + (1.0 - s) * t * q3
    )


def expected_q3_points_from_quad(coarse_vertices, quad):
    """
    Local Q3 node layout:

        12 -- 13 -- 14 -- 15
         |     |     |     |
         8 --  9 -- 10 -- 11
         |     |     |     |
         4 --  5 --  6 --  7
         |     |     |     |
         0 --  1 --  2 --  3

    This gives the same A = i + 4*j ordering used by the bicubic basis.
    """
    q0, q1, q2, q3 = coarse_vertices[quad]

    coords = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    pts = []

    for j in range(4):
        t = coords[j]

        for i in range(4):
            s = coords[i]
            p = bilinear_quad_point(q0, q1, q2, q3, s, t)
            pts.append(p)

    return np.asarray(pts, dtype=float)


def reconstruct_faces16_from_csv(coarse_vertices, quads, csv_vertices, tol):
    tree = cKDTree(csv_vertices)

    faces16 = []
    failed = []

    for e, quad in enumerate(quads):
        expected = expected_q3_points_from_quad(coarse_vertices, quad)

        dist, idx = tree.query(expected, k=1)

        if np.max(dist) > tol:
            failed.append((e, float(np.max(dist)), quad.tolist()))

        if len(set(idx.tolist())) != 16:
            failed.append((e, "duplicate local match", quad.tolist()))

        faces16.append(idx)

    if failed:
        msg = []
        msg.append("Failed to reconstruct all Faces16.")
        msg.append(f"MATCH_TOL = {tol}")
        msg.append(f"Number failed = {len(failed)}")
        msg.append("First failures:")

        for item in failed[:10]:
            msg.append(str(item))

        msg.append("")
        msg.append("Possible fixes:")
        msg.append("  1. Increase MATCH_TOL, e.g. 1e-6.")
        msg.append("  2. Make sure ORIGINAL_OBJ_PATH is the exact coarse quad mesh.")
        msg.append("  3. If the bicubic CSV was projected to a surface, edit reconstruction_project(p).")
        msg.append("  4. If original OBJ has no quad faces, save Faces16 during the solver instead.")

        raise RuntimeError("\n".join(msg))

    return np.asarray(faces16, dtype=int)


def reconstruct_or_load():
    if DO_RECONSTRUCT:
        print("Reading bicubic CSV...")
        vertices, U = read_bicubic_csv(BICUBIC_CSV_PATH)
        print(f"  CSV vertices: {len(vertices)}")

        print("Reading original OBJ...")
        coarse_vertices, obj_faces, obj_edges = parse_obj_vertices_faces_edges(
            ORIGINAL_OBJ_PATH
        )

        print(f"  Original vertices: {len(coarse_vertices)}")
        print(f"  Original faces: {len(obj_faces)}")
        print(f"  Original edges: {len(obj_edges)}")

        quads = extract_quad_faces(obj_faces)

        if len(quads) == 0:
            if TRY_INFER_QUADS_FROM_EDGES:
                print("No quad faces found. Trying edge-only quad inference...")
                quads = infer_quads_from_edges(coarse_vertices, obj_edges)
                print(f"  Inferred quads: {len(quads)}")
            else:
                raise RuntimeError(
                    "Original OBJ has no quad faces. "
                    "Cannot reliably reconstruct bicubic Faces16."
                )

        print(f"Using coarse quads: {len(quads)}")

        print("Reconstructing Faces16...")
        Faces16 = reconstruct_faces16_from_csv(
            coarse_vertices,
            quads,
            vertices,
            MATCH_TOL
        )

        print(f"  Faces16 shape: {Faces16.shape}")

        if DO_SAVE_NPZ:
            np.savez(
                OUT_NPZ_PATH,
                Vertices=vertices,
                U=U,
                Faces16=Faces16
            )
            print(f"Saved reconstructed NPZ: {OUT_NPZ_PATH}")

        return vertices, U, Faces16

    else:
        print(f"Loading reconstructed NPZ: {LOAD_NPZ_PATH}")

        data = np.load(LOAD_NPZ_PATH)

        vertices = data["Vertices"]
        U = data["U"]
        Faces16 = data["Faces16"]

        print(f"  Vertices: {vertices.shape}")
        print(f"  U: {U.shape}")
        print(f"  Faces16: {Faces16.shape}")

        return vertices, U, Faces16


# ============================================================
# COLORED OBJ VISUALIZATION
# ============================================================

def scalar_to_rgb(scalars, cmap_name="turbo"):
    scalars = np.asarray(scalars, dtype=float)

    smin = np.min(scalars)
    smax = np.max(scalars)

    cmap = plt.get_cmap(cmap_name)

    if np.isclose(smin, smax):
        normalized = np.zeros_like(scalars)
    else:
        normalized = (scalars - smin) / (smax - smin)

    return cmap(normalized)[:, :3]


def bicubic_faces_to_subquads(Faces16):
    """
    Converts each Q3 16-node face into 9 ordinary linear quads for OBJ viewing.
    """
    subfaces = []

    for face in Faces16:
        face = np.asarray(face, dtype=int)
        grid = face.reshape((4, 4))

        for j in range(3):
            for i in range(3):
                v0 = grid[j, i]
                v1 = grid[j, i + 1]
                v2 = grid[j + 1, i + 1]
                v3 = grid[j + 1, i]

                subfaces.append([v0, v1, v2, v3])

    return np.asarray(subfaces, dtype=int)


def get_visual_scalars(vertices, U):
    if COLOR_BY.lower() == "solution":
        return U

    if COLOR_BY.lower() == "exact":
        return np.array(
            [exact_u(x, y, z) for x, y, z in vertices],
            dtype=float
        )

    if COLOR_BY.lower() == "error":
        Ue = np.array(
            [exact_u(x, y, z) for x, y, z in vertices],
            dtype=float
        )
        return np.abs(U - Ue)

    raise ValueError(
        f"Unknown COLOR_BY={COLOR_BY}. Use 'solution', 'exact', or 'error'."
    )


def write_colored_obj(filename, vertices, faces, scalars, cmap_name="turbo"):
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    scalars = np.asarray(scalars, dtype=float)

    if len(scalars) != len(vertices):
        raise ValueError(
            f"len(scalars)={len(scalars)} but len(vertices)={len(vertices)}"
        )

    colors = scalar_to_rgb(scalars, cmap_name=cmap_name)

    with open(filename, "w") as f:
        f.write("# Colorized reconstructed bicubic FEM OBJ\n")
        f.write("# vertex format: v x y z r g b\n")
        f.write(f"# color_by: {COLOR_BY}\n")
        f.write(f"# colormap: {cmap_name}\n")

        for p, c in zip(vertices, colors):
            x, y, z = p
            r, g, b = c

            f.write(
                f"v {x:.17g} {y:.17g} {z:.17g} "
                f"{r:.6f} {g:.6f} {b:.6f}\n"
            )

        for face in faces:
            ids = face + 1
            f.write("f " + " ".join(str(i) for i in ids) + "\n")

    print(f"Wrote colored OBJ: {filename}")
    print(f"  scalar min = {np.min(scalars):.16e}")
    print(f"  scalar max = {np.max(scalars):.16e}")


def export_visual(vertices, U, Faces16):
    SubFaces = bicubic_faces_to_subquads(Faces16)
    scalars = get_visual_scalars(vertices, U)

    write_colored_obj(
        OUT_COLOR_OBJ_PATH,
        vertices,
        SubFaces,
        scalars,
        cmap_name=COLOR_MAP
    )


# ============================================================
# Q3 BASIS AND QUADRATURE
# ============================================================

def q3_lagrange_basis_1d(t):
    return np.array([
        -9.0 / 16.0  * (t + 1.0 / 3.0) * (t - 1.0 / 3.0) * (t - 1.0),
         27.0 / 16.0 * (t + 1.0)       * (t - 1.0 / 3.0) * (t - 1.0),
        -27.0 / 16.0 * (t + 1.0)       * (t + 1.0 / 3.0) * (t - 1.0),
          9.0 / 16.0 * (t + 1.0)       * (t + 1.0 / 3.0) * (t - 1.0 / 3.0)
    ], dtype=float)


def q3_lagrange_basis_derivative_1d(t):
    return np.array([
        -9.0 / 16.0  * (3.0 * t**2 - 2.0 * t - 1.0 / 9.0),
         27.0 / 16.0 * (3.0 * t**2 - 2.0 * t / 3.0 - 1.0),
        -27.0 / 16.0 * (3.0 * t**2 + 2.0 * t / 3.0 - 1.0),
          9.0 / 16.0 * (3.0 * t**2 + 2.0 * t - 1.0 / 9.0)
    ], dtype=float)


def q3_basis_and_derivatives(xi, eta):
    lx = q3_lagrange_basis_1d(xi)
    ly = q3_lagrange_basis_1d(eta)

    dlx = q3_lagrange_basis_derivative_1d(xi)
    dly = q3_lagrange_basis_derivative_1d(eta)

    phi = np.zeros(16)
    dphi_dxi = np.zeros(16)
    dphi_deta = np.zeros(16)

    for j in range(4):
        for i in range(4):
            a = i + 4 * j

            phi[a] = lx[i] * ly[j]
            dphi_dxi[a] = dlx[i] * ly[j]
            dphi_deta[a] = lx[i] * dly[j]

    return phi, dphi_dxi, dphi_deta


def tensor_legendre_quadrature(n):
    pts, wts = leggauss(n)

    qpts = []
    qwts = []

    for i in range(n):
        for j in range(n):
            qpts.append((pts[i], pts[j]))
            qwts.append(wts[i] * wts[j])

    return qpts, qwts


# ============================================================
# ERROR TESTING
# ============================================================

def nodal_error(vertices, U, normalize_mean=False):
    Ue = np.array(
        [exact_u(x, y, z) for x, y, z in vertices],
        dtype=float
    )

    Uh = U.copy()

    if normalize_mean:
        Uh = Uh - np.mean(Uh)
        Ue = Ue - np.mean(Ue)

    err = Uh - Ue

    abs_l2 = np.linalg.norm(err)
    exact_l2 = np.linalg.norm(Ue)
    rel_l2 = abs_l2 / exact_l2 if exact_l2 > 0.0 else np.nan
    linf = np.max(np.abs(err))

    return {
        "nodal_abs_l2": abs_l2,
        "nodal_rel_l2": rel_l2,
        "nodal_linf": linf,
    }


def compute_area_weighted_means(vertices, U, Faces16, quad_order):
    qpts, qwts = tensor_legendre_quadrature(quad_order)

    int_uh = 0.0
    int_ue = 0.0
    area = 0.0

    for face in Faces16:
        P = vertices[face]
        Uloc = U[face]

        for (xi, eta), w in zip(qpts, qwts):
            phi, dphi_dxi, dphi_deta = q3_basis_and_derivatives(xi, eta)

            X = phi @ P
            uh = float(phi @ Uloc)
            ue = exact_u(X[0], X[1], X[2])

            g1 = dphi_dxi @ P
            g2 = dphi_deta @ P
            dS = np.linalg.norm(np.cross(g1, g2))

            int_uh += w * uh * dS
            int_ue += w * ue * dS
            area += w * dS

    return int_uh / area, int_ue / area, area


def integrated_l2_error(vertices, U, Faces16, quad_order=10, normalize_mean=False):
    qpts, qwts = tensor_legendre_quadrature(quad_order)

    if normalize_mean:
        mean_uh, mean_ue, _ = compute_area_weighted_means(
            vertices,
            U,
            Faces16,
            quad_order
        )
    else:
        mean_uh = 0.0
        mean_ue = 0.0

    err_l2_sq = 0.0
    exact_l2_sq = 0.0
    area_total = 0.0
    linf_quad = 0.0

    for face in Faces16:
        P = vertices[face]
        Uloc = U[face]

        for (xi, eta), w in zip(qpts, qwts):
            phi, dphi_dxi, dphi_deta = q3_basis_and_derivatives(xi, eta)

            X = phi @ P
            uh = float(phi @ Uloc)
            ue = exact_u(X[0], X[1], X[2])

            if normalize_mean:
                uh = uh - mean_uh
                ue = ue - mean_ue

            g1 = dphi_dxi @ P
            g2 = dphi_deta @ P
            dS = np.linalg.norm(np.cross(g1, g2))

            e = uh - ue

            err_l2_sq += w * e**2 * dS
            exact_l2_sq += w * ue**2 * dS
            area_total += w * dS
            linf_quad = max(linf_quad, abs(e))

    abs_l2 = np.sqrt(err_l2_sq)
    exact_l2 = np.sqrt(exact_l2_sq)
    rel_l2 = abs_l2 / exact_l2 if exact_l2 > 0.0 else np.nan

    return {
        "area": area_total,
        "L2_abs": abs_l2,
        "L2_rel": rel_l2,
        "Linf_quadrature": linf_quad,
    }


def run_error_test(vertices, U, Faces16):
    print("")
    print("Computing nodal error...")

    nodal = nodal_error(
        vertices,
        U,
        normalize_mean=NORMALIZE_MEAN_FOR_ERROR
    )

    for k, v in nodal.items():
        print(f"  {k}: {v:.16e}")

    print("")
    print("Computing integrated L2 error...")

    integ = integrated_l2_error(
        vertices,
        U,
        Faces16,
        quad_order=QUAD_ORDER,
        normalize_mean=NORMALIZE_MEAN_FOR_ERROR
    )

    for k, v in integ.items():
        print(f"  {k}: {v:.16e}")


# ============================================================
# MAIN
# ============================================================

def main():
    vertices, U, Faces16 = reconstruct_or_load()

    if DO_EXPORT_VISUAL:
        print("")
        print("Exporting colored visualization OBJ...")
        export_visual(vertices, U, Faces16)
    else:
        print("")
        print("Skipping visual export because DO_EXPORT_VISUAL=False")

    if DO_ERROR_TEST:
        run_error_test(vertices, U, Faces16)
    else:
        print("")
        print("Skipping error test because DO_ERROR_TEST=False")

    print("")
    print("Done.")


if __name__ == "__main__":
    main()