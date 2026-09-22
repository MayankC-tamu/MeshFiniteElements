import numpy as np


# ============================================================
# Test surface functions
# Each function returns scalar f(p), where surface is f(p) = 0
# ============================================================


def sphere_function(radius=1.0):
    def f(p):
        x, y, z = p
        return x*x + y*y + z*z - radius*radius
    return f

def bean():
    def f(p):
        x, y, z = p
        return (x**2 + y**2 + z**2)**2 - x - z**2
    return f

def superellipsoid_function(a=1.0, b=0.8, c=1.2, n=4):
    def f(p):
        x, y, z = p
        return (x / a) ** n + (y / b) ** n + (z / c) ** n - 1.0
    return f


def torus_function(R=1.0, r=0.35):
    def f(p):
        x, y, z = p
        rho = np.sqrt(x*x + y*y)
        return (rho - R) ** 2 + z*z - r*r
    return f


def wavy_torus_function(R=1.0, r=0.28, amp=0.25, waves=6):
    def f(p):
        x, y, z = p
        rho = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)

        local_r = r * (1.0 + amp * np.cos(waves * theta))

        return (rho - R) ** 2 + z*z - local_r**2
    return f


def asymmetric_torus_function(R=1.0, r=0.32):
    def f(p):
        x, y, z = p

        # Coordinate deformation
        x2 = x
        y2 = 0.75 * y
        z2 = 1.25 * z + 0.15 * x * y

        rho = np.sqrt(x2*x2 + y2*y2)

        return (rho - R) ** 2 + z2*z2 - r*r
    return f

def X_func():
    def f(p):
        x, y, z = p
        return (x**2 + y**2 - 1)**2 + (z**2 - 1/2)**2 + (y**2 + z**2 - 1)**2 - 1/2
    return f


def smooth_min(values, k=12.0):
    """
    Smooth approximation of min(values).
    Useful for smooth unions of implicit solids.
    """
    values = np.asarray(values, dtype=float)
    a = -k * values
    amax = np.max(a)

    return -(amax + np.log(np.sum(np.exp(a - amax)))) / k


def torus_field(p, center, R=0.65, r=0.30):
    x, y, z = p
    cx, cy, cz = center

    X = x - cx
    Y = y - cy
    Z = z - cz

    rho = np.sqrt(X*X + Y*Y)

    return (rho - R) ** 2 + Z*Z - r*r


def multi_torus_chain_function(
    genus=2,
    spacing=0.95,
    R=0.65,
    r=0.32,
    smoothness=12.0,
):
    """
    Approximate genus-g surface using a smooth union of g toroidal solids.

    The actual genus should still be verified from the extracted mesh because
    topology depends on resolution, spacing, and tube radius.
    """
    centers = []

    offset = 0.5 * (genus - 1) * spacing

    for i in range(genus):
        centers.append(np.array([i * spacing - offset, 0.0, 0.0]))

    def f(p):
        vals = [
            torus_field(p, center=c, R=R, r=r)
            for c in centers
        ]

        return smooth_min(vals, k=smoothness)

    return f


def tanglecube_function(c=11.8):
    """
    Classic closed implicit surface with multiple tunnels.
    Often used as a higher-genus stress test.
    """
    def f(p):
        x, y, z = p

        return (
            x**4 + y**4 + z**4
            - 5.0 * x**2
            - 5.0 * y**2
            - 5.0 * z**2
            + c
        )

    return f

resolution = 96

# test_surfaces = [
#         {
#             "name": "sphere_genus_0",
#             "function": sphere_function(radius=1.0),
#             "bounds": (
#                 (-1.3, 1.3),
#                 (-1.3, 1.3),
#                 (-1.3, 1.3),
#             ),
#             "expected_genus": 0,
#             "resolution": resolution,
#             "projection_max_step": 0.04,
#         },
#         {
#             "name": "superellipsoid_genus_0",
#             "function": superellipsoid_function(a=1.0, b=0.8, c=1.2, n=4),
#             "bounds": (
#                 (-1.4, 1.4),
#                 (-1.2, 1.2),
#                 (-1.6, 1.6),
#             ),
#             "expected_genus": 0,
#             "resolution": resolution,
#             "projection_max_step": 0.04,
#         },
#         {
#             "name": "torus_genus_1",
#             "function": torus_function(R=1.0, r=0.35),
#             "bounds": (
#                 (-1.6, 1.6),
#                 (-1.6, 1.6),
#                 (-0.8, 0.8),
#             ),
#             "expected_genus": 1,
#             "resolution": resolution,
#             "projection_max_step": 0.03,
#         },
#         {
#             "name": "wavy_torus_genus_1",
#             "function": wavy_torus_function(R=1.0, r=0.28, amp=0.25, waves=6),
#             "bounds": (
#                 (-1.6, 1.6),
#                 (-1.6, 1.6),
#                 (-0.8, 0.8),
#             ),
#             "expected_genus": 1,
#             "resolution": resolution,
#             "projection_max_step": 0.025,
#         },
#         {
#             "name": "asymmetric_torus_genus_1",
#             "function": asymmetric_torus_function(R=1.0, r=0.32),
#             "bounds": (
#                 (-1.8, 1.8),
#                 (-1.8, 1.8),
#                 (-1.0, 1.0),
#             ),
#             "expected_genus": 1,
#             "resolution": resolution,
#             "projection_max_step": 0.03,
#         },
#         {
#             "name": "double_torus_genus_2",
#             "function": multi_torus_chain_function(
#                 genus=2,
#                 spacing=0.95,
#                 R=0.65,
#                 r=0.32,
#                 smoothness=12.0,
#             ),
#             "bounds": (
#                 (-2.0, 2.0),
#                 (-1.4, 1.4),
#                 (-1.0, 1.0),
#             ),
#             "expected_genus": 2,
#             "resolution": resolution,
#             "projection_max_step": 0.025,
#         },
#         {
#             "name": "triple_torus_genus_3",
#             "function": multi_torus_chain_function(
#                 genus=3,
#                 spacing=0.95,
#                 R=0.65,
#                 r=0.32,
#                 smoothness=12.0,
#             ),
#             "bounds": (
#                 (-2.6, 2.6),
#                 (-1.4, 1.4),
#                 (-1.0, 1.0),
#             ),
#             "expected_genus": 3,
#             "resolution": resolution,
#             "projection_max_step": 0.02,
#         },
#         {
#             "name": "tanglecube_high_genus",
#             "function": tanglecube_function(c=11.8),
#             "bounds": (
#                 (-3.0, 3.0),
#                 (-3.0, 3.0),
#                 (-3.0, 3.0),
#             ),
#             "expected_genus": "usually around 5, verify numerically",
#             "resolution": max(resolution, 64),
#             "projection_max_step": 0.02,
#         },
#     ]

test_surfaces = [
        {
            "name": "tanglecube_genus_5",
            "function": tanglecube_function(c=11.8),
            "bounds": (
                (-3.0, 3.0),
                (-3.0, 3.0),
                (-3.0, 3.0),
            ),
            "expected_genus": "usually around 5, verify numerically",
            "resolution": max(resolution, 64),
            "projection_max_step": 0.02,
        },
    ]