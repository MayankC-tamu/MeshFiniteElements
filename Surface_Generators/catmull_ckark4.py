import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from catmull_clark3 import Mesh, catmull_clark  # Assuming these are defined in catmull_clark3.py

# ============================================================
# 1. SPHERE CONTROL MESH WITH EXTRAORDINARY POLES
# ============================================================
def make_sphere_control_net(n_lat=8, n_lon=6, radius=1.0):
    """
    UV-sphere quad control net.
    Interior vertices have valence 4 (regular).
    The two POLES have valence n_lon (extraordinary if n_lon != 4).
    """
    verts = []
    # north pole
    north = len(verts); verts.append([0, 0, radius])
    ring_idx = np.zeros((n_lat - 1, n_lon), dtype=int)
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            verts.append([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
            ring_idx[i - 1, j] = len(verts) - 1
    south = len(verts); verts.append([0, 0, -radius])
    verts = np.array(verts, dtype=float)

    faces = []
    # north cap: triangles fan -> but we want quads, so use degenerate-free
    # quad fan by pairing pole with consecutive ring edges (triangles here).
    for j in range(n_lon):
        a = ring_idx[0, j]
        b = ring_idx[0, (j + 1) % n_lon]
        faces.append([north, a, b])          # triangle at pole
    # middle quad band
    for i in range(n_lat - 2):
        for j in range(n_lon):
            a = ring_idx[i, j]
            b = ring_idx[i, (j + 1) % n_lon]
            c = ring_idx[i + 1, (j + 1) % n_lon]
            d = ring_idx[i + 1, j]
            faces.append([a, b, c, d])
    for j in range(n_lon):
        a = ring_idx[n_lat - 2, j]
        b = ring_idx[n_lat - 2, (j + 1) % n_lon]
        faces.append([south, b, a])          # triangle at pole
    return verts, faces, north, south


# ============================================================
# 2. CUBIC B-SPLINE BASIS (regular Catmull-Clark limit patch)
# ============================================================
def cubic_bspline_1d(t):
    return np.array([
        (1 - t) ** 3,
        3 * t**3 - 6 * t**2 + 4,
        -3 * t**3 + 3 * t**2 + 3 * t + 1,
        t**3,
    ]) / 6.0


def regular_patch_basis(u, v):
    Bu = cubic_bspline_1d(u)
    Bv = cubic_bspline_1d(v)
    N = np.zeros(16)
    for j in range(4):
        for i in range(4):
            N[4 * j + i] = Bu[i] * Bv[j]
    return N


# ============================================================
# 3. CATMULL-CLARK SUBDIVISION (to reveal the true pole surface)
#    Reuse your catmull_clark from catmull_clark3.py [3].
# ============================================================
# from catmull_clark3 import Mesh, catmull_clark


def subdivide_n_times(verts, faces, k, catmull_clark, Mesh):
    m = Mesh(vertices=np.asarray(verts, float),
             faces=faces)
    for _ in range(k):
        m = catmull_clark(m)   # your generator's routine [3]
    return np.asarray(m.vertices), [list(map(int, f)) for f in m.faces]


# ============================================================
# 4. VISUALIZE: regular B-spline patch vs. subdivided pole
# ============================================================
def evaluate_regular_patch(stencil, n=30):
    us = np.linspace(0, 1, n); vs = np.linspace(0, 1, n)
    surf = np.zeros((n, n, 3))
    for a, u in enumerate(us):
        for b, v in enumerate(vs):
            surf[a, b] = regular_patch_basis(u, v) @ stencil
    return surf


def plot_comparison(verts, faces, pole_idx, regular_stencil,
                    reg_surf, sub_verts, sub_faces):
    fig = plt.figure(figsize=(14, 6))

    # --- LEFT: regular region = exact bicubic B-spline ---
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(reg_surf[:, :, 0], reg_surf[:, :, 1],
                     reg_surf[:, :, 2], color="cyan", alpha=0.85)
    ax1.scatter(regular_stencil[:, 0], regular_stencil[:, 1],
                regular_stencil[:, 2], color="red", s=25)
    ax1.set_title("Regular patch (valence 4)\n= exact bicubic B-spline (C2)")

    # --- RIGHT: pole region = true CC limit (via subdivision) ---
    ax2 = fig.add_subplot(122, projection="3d")
    # plot subdivided faces near the pole
    pole_pos = verts[pole_idx]
    for f in sub_faces:
        pts = sub_verts[np.array(f)]
        if np.linalg.norm(pts.mean(axis=0) - pole_pos) < 0.6:
            poly = np.vstack([pts, pts[0]])
            ax2.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                     color="darkorange", lw=0.6)
    ax2.scatter(*pole_pos, color="black", s=60, label="extraordinary pole")
    ax2.set_title("Pole patch (valence != 4)\nNOT a bicubic B-spline (Stam needed)")
    ax2.legend()

    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # n_lon = 4 -> poles have valence 4 (regular)
    verts, faces, north, south = make_sphere_control_net(n_lat=8, n_lon=4)

    # ---- pick a REGULAR quad from the middle band ----
    quad_faces = [f for f in faces if len(f) == 4]
    reg_quad = quad_faces[len(quad_faces) // 2]
    # crude 16-pt stencil: reuse the 4 corners repeated as placeholder ring
    # (for a real stencil, gather one-ring via topology helpers [3])
    corners = verts[np.array(reg_quad)]
    stencil = np.zeros((16, 3))
    for j in range(4):
        for i in range(4):
            s, t = i / 3.0, j / 3.0
            stencil[4 * j + i] = (
                (1 - s) * (1 - t) * corners[0] + s * (1 - t) * corners[1]
                + s * t * corners[2] + (1 - s) * t * corners[3]
            )
    reg_surf = evaluate_regular_patch(stencil)

    # ---- reveal the true pole surface via subdivision ----
    # Uncomment once catmull_clark / Mesh are imported [3]:
    sub_verts, sub_faces = subdivide_n_times(verts, faces, 4,
                                             catmull_clark, Mesh)
    plot_comparison(verts, faces, north, stencil, reg_surf,
                    sub_verts, sub_faces)

    print("Set n_lon != 4 to make the poles extraordinary.")
    print(f"North pole valence = {sum(1 for f in faces if north in f)}")