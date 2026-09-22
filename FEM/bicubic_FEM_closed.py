import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.polynomial.legendre import leggauss
import meshio
import tqdm as tqdm

# Source term
def f(x, y, z):
    #return x**2 + y**2 + z**2
    #return 12*x*y*z+1.8*(3*z**2-1)+1.2*(x**2-y**2)
    return np.sin(2*np.pi*(x**2+y**2+z**2))


Q3_NODES = np.array([-1.0, -1.0/3.0, 1.0/3.0, 1.0])


def lagrange_1d(t):
    """
    Cubic Lagrange basis and derivatives at t
    for nodes [-1, -1/3, 1/3, 1].
    """
    x = Q3_NODES
    L = np.ones(4)
    dL = np.zeros(4)

    for i in range(4):
        for j in range(4):
            if j != i:
                L[i] *= (t - x[j]) / (x[i] - x[j])

        # Derivative
        total = 0.0
        for k in range(4):
            if k == i:
                continue

            term = 1.0 / (x[i] - x[k])
            for j in range(4):
                if j != i and j != k:
                    term *= (t - x[j]) / (x[i] - x[j])

            total += term

        dL[i] = total

    return L, dL


class BicubicElement:
    def __init__(self, Vertices, face):
        """
        face should contain 16 global node indices.

        Ordering assumed:

        eta = -1:
            0, 1, 2, 3

        eta = -1/3:
            4, 5, 6, 7

        eta = 1/3:
            8, 9, 10, 11

        eta = 1:
            12, 13, 14, 15
        """
        self.face = np.asarray(face, dtype=int)
        self.p = Vertices[self.face]  # shape: 16 x 3

    def basis_functions(self, xi, eta):
        Lxi, dLxi = lagrange_1d(xi)
        Leta, dLeta = lagrange_1d(eta)

        N = np.zeros(16)
        dN_dxi = np.zeros(16)
        dN_deta = np.zeros(16)

        for j in range(4):
            for i in range(4):
                a = 4*j + i
                N[a] = Lxi[i] * Leta[j]
                dN_dxi[a] = dLxi[i] * Leta[j]
                dN_deta[a] = Lxi[i] * dLeta[j]

        return N, dN_dxi, dN_deta

    def local_to_global(self, xi, eta):
        N, _, _ = self.basis_functions(xi, eta)
        return N @ self.p

    def geometry(self, xi, eta):
        """
        Returns:
            x       physical point
            g1      dx/dxi
            g2      dx/deta
            area    |g1 x g2|
        """
        N, dN_dxi, dN_deta = self.basis_functions(xi, eta)

        x = N @ self.p
        g1 = dN_dxi @ self.p
        g2 = dN_deta @ self.p

        normal_cross = np.cross(g1, g2)
        area = np.linalg.norm(normal_cross)

        return x, g1, g2, area

    def surface_gradients(self, xi, eta):
        """
        Compute physical surface gradients of the 16 bicubic basis functions.
        """
        N, dN_dxi, dN_deta = self.basis_functions(xi, eta)
        x, g1, g2, area = self.geometry(xi, eta)

        G = np.array([
            [np.dot(g1, g1), np.dot(g1, g2)],
            [np.dot(g2, g1), np.dot(g2, g2)]
        ])

        G_inv = np.linalg.inv(G)

        # Contravariant tangent vectors
        g_contra_1 = G_inv[0, 0] * g1 + G_inv[0, 1] * g2
        g_contra_2 = G_inv[1, 0] * g1 + G_inv[1, 1] * g2

        grad = np.zeros((16, 3))

        for a in range(16):
            grad[a] = dN_dxi[a] * g_contra_1 + dN_deta[a] * g_contra_2

        return grad, N, x, area



def quadrature_points(n=6):
    pts, wts = leggauss(n)

    GP = []
    weights = []

    for i in range(n):
        for j in range(n):
            xi = pts[i]
            eta = pts[j]
            weight = wts[i] * wts[j]

            GP.append((xi, eta))
            weights.append(weight)

    return GP, weights


def assemble_bicubic_system(Vertices, Faces, nq=6):
    """
    Vertices: N x 3 array
    Faces:    M x 16 array of bicubic quad connectivity

    Returns:
        K: stiffness matrix
        b: load vector
        m: mean constraint vector, m_i = int_Gamma Psi_i dS
    """
    Nverts = len(Vertices)

    rows = []
    cols = []
    data = []

    b = np.zeros(Nverts)
    m = np.zeros(Nverts)

    GP, weights = quadrature_points(nq)

    for face in tqdm.tqdm(Faces, desc="Assembling Bicubic System"):
        element = BicubicElement(Vertices, face)

        Ke = np.zeros((16, 16))
        be = np.zeros(16)
        me = np.zeros(16)

        for (xi, eta), weight in zip(GP, weights):
            grad, basis, x, area = element.surface_gradients(xi, eta)

            fx = f(x[0], x[1], x[2])

            for i in range(16):
                be[i] += weight * fx * basis[i] * area
                me[i] += weight * basis[i] * area

                for j in range(16):
                    Ke[i, j] += weight * np.dot(grad[i], grad[j]) * area

        # Scatter local contributions into global arrays
        for i_local in range(16):
            i_global = face[i_local]

            b[i_global] += be[i_local]
            m[i_global] += me[i_local]

            for j_local in range(16):
                j_global = face[j_local]

                rows.append(i_global)
                cols.append(j_global)
                data.append(Ke[i_local, j_local])

    K = sp.coo_matrix((data, (rows, cols)), shape=(Nverts, Nverts)).tocsr()

    return K, b, m


def solve_closed_surface_poisson(K, b, m):
    N = K.shape[0]

    Mcol = sp.csr_matrix(m.reshape(-1, 1))
    Mrow = sp.csr_matrix(m.reshape(1, -1))
    zero = sp.csr_matrix((1, 1))

    K_aug = sp.bmat([
        [K,    Mcol],
        [Mrow, zero]
    ], format="csr")

    rhs = np.concatenate([b, np.array([0.0])])

    sol = spla.spsolve(K_aug, rhs)

    U = sol[:-1]
    lam = sol[-1]

    return U, lam


def make_bicubic_mesh(Vertices, Quads, project=None):
    """
    Convert a 4-node quad mesh into a 16-node bicubic quad mesh.

    Vertices: N x 3
    Quads:    M x 4, ordered [q0, q1, q2, q3]

    Local ordering:
        12, 13, 14, 15
         8,  9, 10, 11
         4,  5,  6,  7
         0,  1,  2,  3
    """

    Vertices = np.asarray(Vertices, dtype=float)
    Quads = np.asarray(Quads, dtype=int)

    vertices_list = [v.copy() for v in Vertices]
    edge_to_nodes = {}

    def maybe_project(p):
        if project is None:
            return np.asarray(p, dtype=float)
        return np.asarray(project(p), dtype=float)

    def add_vertex(p):
        vertices_list.append(maybe_project(p))
        return len(vertices_list) - 1

    def get_edge_node(a, b, m):
        """
        m = 1 gives point 1/3 along edge a -> b.
        m = 2 gives point 2/3 along edge a -> b.
        Shared between neighboring quads.
        """
        key = tuple(sorted((int(a), int(b))))

        if key not in edge_to_nodes:
            v0 = vertices_list[key[0]]
            v1 = vertices_list[key[1]]

            p1 = (2.0/3.0) * v0 + (1.0/3.0) * v1
            p2 = (1.0/3.0) * v0 + (2.0/3.0) * v1

            idx1 = add_vertex(p1)
            idx2 = add_vertex(p2)

            edge_to_nodes[key] = [idx1, idx2]

        nodes = edge_to_nodes[key]

        if (int(a), int(b)) == key:
            return nodes[m - 1]
        else:
            return nodes[2 - m]

    def bilinear_point(q0, q1, q2, q3, xi, eta):
        p0 = vertices_list[q0]
        p1 = vertices_list[q1]
        p2 = vertices_list[q2]
        p3 = vertices_list[q3]

        N0 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N1 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N3 = 0.25 * (1.0 - xi) * (1.0 + eta)

        return N0*p0 + N1*p1 + N2*p2 + N3*p3

    new_faces = []

    for quad in Quads:
        q0, q1, q2, q3 = map(int, quad)

        face = np.empty(16, dtype=int)

        # Corners
        face[0] = q0
        face[3] = q1
        face[15] = q2
        face[12] = q3

        # Bottom edge q0 -> q1
        face[1] = get_edge_node(q0, q1, 1)
        face[2] = get_edge_node(q0, q1, 2)

        # Right edge q1 -> q2
        face[7] = get_edge_node(q1, q2, 1)
        face[11] = get_edge_node(q1, q2, 2)

        # Top edge q3 -> q2
        face[13] = get_edge_node(q3, q2, 1)
        face[14] = get_edge_node(q3, q2, 2)

        # Left edge q0 -> q3
        face[4] = get_edge_node(q0, q3, 1)
        face[8] = get_edge_node(q0, q3, 2)

        # Interior nodes
        face[5] = add_vertex(bilinear_point(q0, q1, q2, q3, -1.0/3.0, -1.0/3.0))
        face[6] = add_vertex(bilinear_point(q0, q1, q2, q3,  1.0/3.0, -1.0/3.0))
        face[9] = add_vertex(bilinear_point(q0, q1, q2, q3, -1.0/3.0,  1.0/3.0))
        face[10] = add_vertex(bilinear_point(q0, q1, q2, q3, 1.0/3.0,  1.0/3.0))

        new_faces.append(face)

    return np.asarray(vertices_list, dtype=float), np.asarray(new_faces, dtype=int)




OBJ_PATH = "./meshes/cube/cube2.obj"
SOLUTION_PATH = "./meshes/cube/cube2_sol.csv"

# Load ordinary quad mesh
mesh = meshio.read(OBJ_PATH)

Vertices0 = np.asarray(mesh.points, dtype=float)
Quads = np.asarray(mesh.cells_dict["quad"], dtype=int)

# Upgrade 4-node quads to 16-node bicubic quads
Vertices, Faces = make_bicubic_mesh(Vertices0, Quads)

print("Number of vertices after bicubic upgrade:", len(Vertices))
print("Number of bicubic faces:", len(Faces))

# Assemble bicubic closed-surface FEM system
K, b, m = assemble_bicubic_system(Vertices, Faces, nq=6)

area = np.sum(m)
print("Approximate surface area:", area)
print("Approximate mean of f:", np.sum(b) / area)

# Solve with zero-mean constraint
U, lam = solve_closed_surface_poisson(K, b, m)

print("Lagrange multiplier:", lam)
print("Mean of solution:", np.dot(m, U) / area)
print("min(U), max(U):", np.min(U), np.max(U))

N = len(Vertices)

File = open(SOLUTION_PATH,'w')
File.write('x,y,z,u\n')

for i in range(N):
  (x,y,z) = Vertices[i]
  File.write(str(x)+', '+str(y)+', '+str(z)+', '+str(U[i])+'\n')

File.close()