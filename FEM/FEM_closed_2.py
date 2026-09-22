"""
arbitrary_surface_fem_solver.py

FEM solver for arbitrary non-parametric OBJ surface meshes.

Solves either:

    screened Poisson:
        -Delta_Gamma u + alpha u = f

or:

    zero-mean Poisson:
        -Delta_Gamma u = f,   integral_Gamma u dS = 0

using P1 linear triangular surface FEM.

This is intended for non-mathematical meshes such as scanned/artist OBJ files.

Dependencies:
    pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix, csr_matrix, bmat
from scipy.sparse.linalg import spsolve


# ============================================================
# USER CONFIG
# ============================================================

OBJ_PATH = "./meshes/Cow/Cow.obj"

OUT_COLOR_OBJ = "./meshes/Cow/Mesh_Cow_colored.obj"
OUT_NPZ = "./meshes/Cow/cow_fem_solution.npz"

# Your Beagle file says coordinates are in centimeters.
# Use SCALE = 1.0 to keep centimeters.
# Use SCALE = 0.01 to convert cm -> meters.
SCALE = 1.0

# PDE_TYPE options:
#   "screened_poisson"
#   "poisson_zero_mean"
PDE_TYPE = "screened_poisson"

# Used only for screened Poisson:
# Larger alpha makes solution more localized/smoother-conditioned.
REACTION_ALPHA = 1.0e-2

# RHS_TYPE options:
#   "gaussian"
#   "height"
#   "x_coordinate"
#   "y_coordinate"
#   "z_coordinate"
#   "two_gaussians"
RHS_TYPE = "gaussian"

# Gaussian source settings.
# CENTER_MODE options:
#   "bbox_center"
#   "max_x"
#   "max_y"
#   "max_z"
#   "manual"
CENTER_MODE = "max_z"

MANUAL_CENTER = np.array([0.0, 0.0, 0.0], dtype=float)

# In mesh units. If your mesh is in cm, this is cm.
GAUSSIAN_SIGMA = 10.0

# For "two_gaussians"
SECOND_CENTER_MODE = "min_z"
SECOND_GAUSSIAN_SIGMA = 10.0

# Optional Dirichlet pin.
# Usually False for closed surfaces.
# If True, sets one vertex value to PIN_VALUE.
USE_DIRICHLET_PIN = False
PIN_MODE = "min_y"
PIN_VALUE = 0.0

# Visualization
COLOR_MAP = "turbo"

# COLOR_BY options:
#   "solution"
#   "rhs"
#   "abs_solution"
COLOR_BY = "solution"

# Optional exact solution testing.
# For non-mathematical meshes this is usually False.
DO_EXACT_ERROR_TEST = False


# ============================================================
# OPTIONAL EXACT SOLUTION
# ============================================================

def exact_u(x, y, z):
    """
    Only used if DO_EXACT_ERROR_TEST=True.

    For arbitrary meshes there usually is no exact solution.
    You can still define a manufactured coordinate function if desired.
    """
    return np.sin(0.1 * x) * np.cos(0.1 * y)


# ============================================================
# OBJ PARSER
# ============================================================

def parse_obj(path, scale=1.0):
    """
    Minimal OBJ parser.

    Reads:
        v x y z
        f i j k ...
        f i/t/n j/t/n k/t/n ...

    Ignores:
        vn, vt, mtllib, usemtl, etc.

    Returns:
        V:     (N,3) float
        faces: list of lists of vertex indices, zero-based
    """
    vertices = []
    faces = []

    def parse_face_index(tok, n_vertices_so_far):
        """
        OBJ indices are 1-based.
        Negative indices are relative to current vertex list.
        """
        raw = tok.split("/")[0]
        idx = int(raw)

        if idx > 0:
            return idx - 1
        else:
            return n_vertices_so_far + idx

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            if parts[0] == "v":
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                vertices.append([scale * x, scale * y, scale * z])

            elif parts[0] == "f":
                n = len(vertices)
                face = [parse_face_index(tok, n) for tok in parts[1:]]

                if len(face) >= 3:
                    faces.append(face)

    V = np.asarray(vertices, dtype=float)

    if len(V) == 0:
        raise ValueError("OBJ contains no vertices.")

    if len(faces) == 0:
        raise ValueError(
            "OBJ contains no faces. FEM cannot be assembled from vertices/normals only. "
            "You need face connectivity."
        )

    return V, faces


def triangulate_faces(faces):
    """
    Fan-triangulates arbitrary polygonal OBJ faces.

    Quad:
        [0,1,2,3] -> [0,1,2], [0,2,3]

    Polygon:
        [0,1,2,3,4] -> [0,1,2], [0,2,3], [0,3,4]
    """
    triangles = []

    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        else:
            v0 = face[0]
            for i in range(1, len(face) - 1):
                triangles.append([v0, face[i], face[i + 1]])

    return np.asarray(triangles, dtype=int)


def remove_degenerate_triangles(V, T, eps=1e-14):
    good = []

    for tri in T:
        p0, p1, p2 = V[tri]
        area2 = np.linalg.norm(np.cross(p1 - p0, p2 - p0))

        if area2 > eps:
            good.append(tri)

    return np.asarray(good, dtype=int)


# ============================================================
# SURFACE FEM ASSEMBLY
# ============================================================

def triangle_gradients(p0, p1, p2):
    """
    Computes gradients of P1 basis functions on a 3D triangle.

    Returns:
        grads: (3,3), where grads[i] is surface gradient of phi_i
        area: triangle area
    """
    n = np.cross(p1 - p0, p2 - p0)
    n2 = np.dot(n, n)

    if n2 <= 0.0:
        raise ValueError("Degenerate triangle.")

    area = 0.5 * np.sqrt(n2)

    # grad phi_i formulas in embedded 3D
    g0 = np.cross(n, p2 - p1) / n2
    g1 = np.cross(n, p0 - p2) / n2
    g2 = np.cross(n, p1 - p0) / n2

    grads = np.vstack([g0, g1, g2])

    return grads, area


def assemble_p1_surface_fem(V, T):
    """
    Assembles stiffness K and mass M.

    Weak form:
        K_ij = integral_Gamma grad phi_i . grad phi_j dS
        M_ij = integral_Gamma phi_i phi_j dS
    """
    n_vertices = len(V)

    K_rows = []
    K_cols = []
    K_vals = []

    M_rows = []
    M_cols = []
    M_vals = []

    total_area = 0.0

    for tri in T:
        p0, p1, p2 = V[tri]

        grads, area = triangle_gradients(p0, p1, p2)
        total_area += area

        Kloc = area * (grads @ grads.T)

        Mloc = (area / 12.0) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0]
        ])

        for a in range(3):
            A = tri[a]

            for b in range(3):
                B = tri[b]

                K_rows.append(A)
                K_cols.append(B)
                K_vals.append(Kloc[a, b])

                M_rows.append(A)
                M_cols.append(B)
                M_vals.append(Mloc[a, b])

    K = coo_matrix(
        (K_vals, (K_rows, K_cols)),
        shape=(n_vertices, n_vertices)
    ).tocsr()

    M = coo_matrix(
        (M_vals, (M_rows, M_cols)),
        shape=(n_vertices, n_vertices)
    ).tocsr()

    return K, M, total_area


# ============================================================
# RHS / SOURCE TERM
# ============================================================

def choose_point(V, mode):
    mode = mode.lower()

    if mode == "bbox_center":
        return 0.5 * (np.min(V, axis=0) + np.max(V, axis=0))

    if mode == "max_x":
        return V[np.argmax(V[:, 0])]

    if mode == "min_x":
        return V[np.argmin(V[:, 0])]

    if mode == "max_y":
        return V[np.argmax(V[:, 1])]

    if mode == "min_y":
        return V[np.argmin(V[:, 1])]

    if mode == "max_z":
        return V[np.argmax(V[:, 2])]

    if mode == "min_z":
        return V[np.argmin(V[:, 2])]

    if mode == "manual":
        return np.asarray(MANUAL_CENTER, dtype=float)

    raise ValueError(f"Unknown point mode: {mode}")


def gaussian_on_vertices(V, center, sigma):
    d2 = np.sum((V - center[None, :])**2, axis=1)
    return np.exp(-d2 / (2.0 * sigma**2))


def normalize_to_unit_range(a):
    a = np.asarray(a, dtype=float)
    amin = np.min(a)
    amax = np.max(a)

    if np.isclose(amin, amax):
        return np.zeros_like(a)

    return (a - amin) / (amax - amin)


def compute_rhs_vertex_values(V):
    """
    Returns nodal values f_i for source f.
    The load vector is then b = M f.
    """
    rhs_type = RHS_TYPE.lower()

    if rhs_type == "gaussian":
        c = choose_point(V, CENTER_MODE)
        f = gaussian_on_vertices(V, c, GAUSSIAN_SIGMA)
        return f

    if rhs_type == "two_gaussians":
        c1 = choose_point(V, CENTER_MODE)
        c2 = choose_point(V, SECOND_CENTER_MODE)

        f1 = gaussian_on_vertices(V, c1, GAUSSIAN_SIGMA)
        f2 = gaussian_on_vertices(V, c2, SECOND_GAUSSIAN_SIGMA)

        return f1 - f2

    if rhs_type == "height":
        return normalize_to_unit_range(V[:, 2]) - 0.5

    if rhs_type == "x_coordinate":
        return normalize_to_unit_range(V[:, 0]) - 0.5

    if rhs_type == "y_coordinate":
        return normalize_to_unit_range(V[:, 1]) - 0.5

    if rhs_type == "z_coordinate":
        return normalize_to_unit_range(V[:, 2]) - 0.5

    raise ValueError(f"Unknown RHS_TYPE: {RHS_TYPE}")


def subtract_mass_mean(f, M):
    """
    Makes f have zero surface mean:
        integral f dS = 0

    Needed for pure closed-surface Poisson.
    """
    ones = np.ones_like(f)
    m = M @ ones

    integral_f = float(m @ f)
    area = float(m @ ones)

    mean_f = integral_f / area

    return f - mean_f


# ============================================================
# SOLVERS
# ============================================================

def apply_dirichlet_pin(A, b, pin_index, pin_value):
    """
    Applies u[pin_index] = pin_value.
    """
    A = A.tolil()
    b = b.copy()

    # Adjust RHS for fixed value.
    col = A[:, pin_index].toarray().ravel()
    b -= col * pin_value

    # Zero row and column.
    A[pin_index, :] = 0.0
    A[:, pin_index] = 0.0

    A[pin_index, pin_index] = 1.0
    b[pin_index] = pin_value

    return A.tocsr(), b


def solve_screened_poisson(K, M, f):
    """
    Solves:
        -Delta u + alpha u = f

    Weak form:
        (K + alpha M) u = M f
    """
    A = K + REACTION_ALPHA * M
    b = M @ f

    if USE_DIRICHLET_PIN:
        pin_point = choose_point(V_GLOBAL_FOR_PIN, PIN_MODE)
        pin_index = int(np.argmin(np.linalg.norm(V_GLOBAL_FOR_PIN - pin_point, axis=1)))
        A, b = apply_dirichlet_pin(A, b, pin_index, PIN_VALUE)

    u = spsolve(A, b)

    residual = np.linalg.norm(A @ u - b) / max(np.linalg.norm(b), 1e-30)

    return u, residual


def solve_poisson_zero_mean(K, M, f):
    """
    Solves:
        -Delta u = f
        integral u dS = 0

    Augmented system:
        [K  m][u] = [M f]
        [mT 0][l]   [ 0 ]

    where m_i = integral phi_i dS.
    """
    f = subtract_mass_mean(f, M)

    b = M @ f

    ones = np.ones(K.shape[0])
    m = M @ ones
    m_col = csr_matrix(m.reshape(-1, 1))
    m_row = csr_matrix(m.reshape(1, -1))

    zero = csr_matrix((1, 1))

    A_aug = bmat([
        [K,     m_col],
        [m_row, zero]
    ], format="csr")

    b_aug = np.concatenate([b, np.array([0.0])])

    sol = spsolve(A_aug, b_aug)

    u = sol[:-1]

    residual = np.linalg.norm(A_aug @ sol - b_aug) / max(np.linalg.norm(b_aug), 1e-30)

    return u, residual


# ============================================================
# ERROR TESTING
# ============================================================

def exact_error_test(V, M, u):
    ue = np.array([exact_u(x, y, z) for x, y, z in V], dtype=float)

    e = u - ue

    nodal_l2 = np.linalg.norm(e)
    nodal_linf = np.max(np.abs(e))

    l2_sq = float(e @ (M @ e))
    exact_l2_sq = float(ue @ (M @ ue))

    l2 = np.sqrt(max(l2_sq, 0.0))
    rel_l2 = l2 / np.sqrt(exact_l2_sq) if exact_l2_sq > 0.0 else np.nan

    return {
        "nodal_l2": nodal_l2,
        "nodal_linf": nodal_linf,
        "surface_L2": l2,
        "surface_relative_L2": rel_l2,
    }


# ============================================================
# COLORED OBJ EXPORT
# ============================================================

def scalar_to_rgb(scalars, cmap_name="turbo"):
    scalars = np.asarray(scalars, dtype=float)

    smin = np.min(scalars)
    smax = np.max(scalars)

    cmap = plt.get_cmap(cmap_name)

    if np.isclose(smin, smax):
        t = np.zeros_like(scalars)
    else:
        t = (scalars - smin) / (smax - smin)

    return cmap(t)[:, :3]


def write_colored_obj(path, V, T, scalars, cmap_name="turbo"):
    colors = scalar_to_rgb(scalars, cmap_name=cmap_name)

    with open(path, "w") as f:
        f.write("# Colorized FEM solution OBJ\n")
        f.write("# vertex format: v x y z r g b\n")

        for p, c in zip(V, colors):
            x, y, z = p
            r, g, b = c

            f.write(
                f"v {x:.17g} {y:.17g} {z:.17g} "
                f"{r:.6f} {g:.6f} {b:.6f}\n"
            )

        for tri in T:
            ids = tri + 1
            f.write(f"f {ids[0]} {ids[1]} {ids[2]}\n")

    print(f"Wrote colored OBJ: {path}")
    print(f"  scalar min = {np.min(scalars):.16e}")
    print(f"  scalar max = {np.max(scalars):.16e}")


def get_visual_scalars(u, f):
    mode = COLOR_BY.lower()

    if mode == "solution":
        return u

    if mode == "rhs":
        return f

    if mode == "abs_solution":
        return np.abs(u)

    raise ValueError(f"Unknown COLOR_BY: {COLOR_BY}")


# ============================================================
# MAIN
# ============================================================

V_GLOBAL_FOR_PIN = None


def main():
    global V_GLOBAL_FOR_PIN

    print("Reading OBJ...")
    V, faces = parse_obj(OBJ_PATH, scale=SCALE)
    V_GLOBAL_FOR_PIN = V

    print(f"  vertices: {len(V)}")
    print(f"  polygon faces: {len(faces)}")

    print("Triangulating faces...")
    T = triangulate_faces(faces)
    T = remove_degenerate_triangles(V, T)

    print(f"  triangles: {len(T)}")

    print("Assembling FEM matrices...")
    K, M, area = assemble_p1_surface_fem(V, T)

    print(f"  surface area: {area:.16e}")
    print(f"  K shape: {K.shape}")
    print(f"  M shape: {M.shape}")

    print("Computing RHS...")
    f = compute_rhs_vertex_values(V)

    if PDE_TYPE.lower() == "screened_poisson":
        print("Solving screened Poisson:")
        print(f"  -Delta_Gamma u + {REACTION_ALPHA} u = f")
        u, residual = solve_screened_poisson(K, M, f)

    elif PDE_TYPE.lower() == "poisson_zero_mean":
        print("Solving zero-mean Poisson:")
        print("  -Delta_Gamma u = f, integral u dS = 0")
        u, residual = solve_poisson_zero_mean(K, M, f)

    else:
        raise ValueError(f"Unknown PDE_TYPE: {PDE_TYPE}")

    print(f"  relative linear residual: {residual:.16e}")
    print(f"  solution min: {np.min(u):.16e}")
    print(f"  solution max: {np.max(u):.16e}")
    print(f"  solution mean nodal: {np.mean(u):.16e}")

    mass_ones = M @ np.ones(len(V))
    surface_mean_u = float(mass_ones @ u) / float(mass_ones @ np.ones(len(V)))

    print(f"  solution mean surface: {surface_mean_u:.16e}")

    if DO_EXACT_ERROR_TEST:
        print("Computing exact error...")
        errs = exact_error_test(V, M, u)

        for k, val in errs.items():
            print(f"  {k}: {val:.16e}")

    print("Saving NPZ...")
    np.savez(
        OUT_NPZ,
        V=V,
        T=T,
        K=K,
        M=M,
        f=f,
        u=u,
        area=area
    )
    print(f"  saved: {OUT_NPZ}")

    print("Exporting colored OBJ...")
    scalars = get_visual_scalars(u, f)

    write_colored_obj(
        OUT_COLOR_OBJ,
        V,
        T,
        scalars,
        cmap_name=COLOR_MAP
    )

    print("Done.")


if __name__ == "__main__":
    main()