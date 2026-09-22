import numpy as np
import trimesh
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

#u(x,y,z)= x*y*z+0.3(3*z**2-1)+0.2*(x**2-y**2)

OBJECT_PATH = "./solutions/closed/sphere/sphere_3158.obj"
SOLUTION_PATH = "./solutions/closed/sphere/sphere_3158_sol.csv"

# Source term
def f(x, y, z):
    #return x**2 + y**2 + z**2
    return 12*x*y*z+1.8*(3*z**2-1)+1.2*(x**2-y**2)


class Element:
    def __init__(self, Vertices, face):
        self.face = face
        self.p = Vertices[face]

        # Edge vectors
        self.e = [
            self.p[1] - self.p[0],
            self.p[2] - self.p[0]
        ]

        # Normal and area
        self.N = np.cross(self.e[0], self.e[1])
        self.A = np.linalg.norm(self.N) / 2.0
        self.n = self.N / np.linalg.norm(self.N)

        # Compute surface gradients of linear basis functions
        M = np.vstack([self.e[0], self.e[1], self.n])
        M_inv = np.linalg.inv(M)

        inv_e1 = M_inv[:, 0]
        inv_e2 = M_inv[:, 1]

        grad_0 = -inv_e1 - inv_e2
        grad_1 = inv_e1
        grad_2 = inv_e2

        self.grad = [grad_0, grad_1, grad_2]


def assemble_system(Vertices, Faces):
    N = len(Vertices)

    rows = []
    cols = []
    data = []

    b = np.zeros(N)
    m = np.zeros(N)   

    for face in Faces:
        element = Element(Vertices, face)
        Atri = element.A

        # Local stiffness matrix
        for i in range(3):
            for j in range(3):
                rows.append(face[i])
                cols.append(face[j])
                data.append(Atri * np.dot(element.grad[i], element.grad[j]))

        # Evaluate f at vertices
        f0 = f(*element.p[0])
        f1 = f(*element.p[1])
        f2 = f(*element.p[2])

        # Correct linear FEM load vector
        b_local = (Atri / 12.0) * np.array([
            2.0 * f0 + f1 + f2,
            f0 + 2.0 * f1 + f2,
            f0 + f1 + 2.0 * f2
        ])

        for i in range(3):
            b[face[i]] += b_local[i]

        # Integral of each P1 basis function over a triangle is A/3
        for i in range(3):
            m[face[i]] += Atri / 3.0

    K = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

    return K, b, m


# Load closed tangle cube mesh
mesh = trimesh.load(
    OBJECT_PATH,
    force="mesh"
)

Vertices = np.asarray(mesh.vertices)
Faces = np.asarray(mesh.faces)
N = len(Vertices)

print("Number of vertices:", N)
print("Number of faces:", len(Faces))
print("Watertight:", mesh.is_watertight)

K, b, m = assemble_system(Vertices, Faces)

# Total surface area
area = np.sum(m)

# Average value of f over the surface, approximately
f_mean = np.sum(b) / area

print("Approximate mean of f over surface:", f_mean)

# Augmented system:
#
# [ K   m ] [ u      ] = [ b ]
# [ m^T 0 ] [ lambda ]   [ 0 ]
#
# This enforces int_Gamma u dS = 0.
#
# If f has nonzero mean, lambda removes the incompatible constant part.
zero = sp.csr_matrix((1, 1))
Mcol = sp.csr_matrix(m.reshape(-1, 1))
Mrow = sp.csr_matrix(m.reshape(1, -1))

K_aug = sp.bmat([
    [K,    Mcol],
    [Mrow, zero]
], format="csr")

rhs = np.concatenate([b, np.array([0.0])])

sol = spla.spsolve(K_aug, rhs)

U = sol[:-1]
lam = sol[-1]

print("Lagrange multiplier:", lam)
print("Mean of solution:", np.dot(m, U) / area)
print("min(U), max(U):", np.min(U), np.max(U))

#Save Results
File = open(SOLUTION_PATH,'w')
File.write('x,y,z,u\n')

for i in range(N):
  (x,y,z) = Vertices[i]
  File.write(str(x)+', '+str(y)+', '+str(z)+', '+str(U[i])+'\n')

File.close()