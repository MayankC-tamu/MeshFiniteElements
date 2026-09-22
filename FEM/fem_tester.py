import numpy as np

def find_triangle(point, division):

    x, y = point

    grid_size = int(round(2 / division))

    x_index = int(np.floor((x + 1) / division))
    y_index = int(np.floor((y + 1) / division))

    x_index = max(0, min(x_index, grid_size - 1))
    y_index = max(0, min(y_index, grid_size - 1))

    x_min = -1 + x_index * division
    x_max = min(x_min + division, 1)
    y_min = -1 + y_index * division
    y_max = min(y_min + division, 1)

    #print(f"Point: {point}, x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}")

    p_i = np.array([x_min, y_min])
    p_j = np.array([x_max, y_min])
    p_k = np.array([x_min, y_max])
    p_m = np.array([x_max, y_max])

    #print(f"Point: {point}, Triangle vertices: {p_i}, {p_j}, {p_k}, {p_m}")

    distances = [np.linalg.norm(point - p_i), np.linalg.norm(point - p_j), np.linalg.norm(point - p_k), np.linalg.norm(point - p_m)]

    max_distance = max(distances)
    max_index = distances.index(max_distance)

    if max_index == 0:
        return p_m, p_j, p_k
    elif max_index == 1:
        return p_i, p_k, p_m
    elif max_index == 2:
        return p_j, p_m, p_i
    else:
        return p_k, p_i, p_j
    

class FEM_V:
    def __init__(self,domain_path,division):
        self.domain = self.get_domain(domain_path)
        self.division = division

    def get_domain(self,domain_path):
        domain = {}
        with open(domain_path, 'r') as f:
            lines = f.readlines()
            lines.pop(0)  # Remove the header line
            for line in lines:
                x, y, u = map(float, line.strip().split(',')[:3])
                domain[(x, y)] = u
        return domain
    
    def get_value(self, point):
        if point in self.domain:
            return self.domain[point]
        else:
            p_i, p_j, p_k = find_triangle(point, self.division)
            triangles = [[point, p_j, p_k], [p_i, point, p_k], [p_i, p_j, point]]
            areas = [self.triangle_area(triangle) for triangle in triangles]
            total_area = sum(areas)
            areas = [area / total_area for area in areas]
            U = [self.domain[tuple(p)] for p in [p_i, p_j, p_k]]
            return sum(area * u for area, u in zip(areas, U))
        
    def triangle_area(self, triangle):
        p1, p2, p3 = triangle
        #cross product formula for area of triangle given by three points
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
    
    def __call__(self, point):
        return self.get_value(point)
    

def round_key(p, ndigits=12):
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits))


def find_triangle_2(point, division):

    point = np.asarray(point, dtype=float)
    x, y = point

    grid_size = int(round(2 / division))

    x_index = int(np.floor((x + 1) / division))
    y_index = int(np.floor((y + 1) / division))

    x_index = max(0, min(x_index, grid_size - 1))
    y_index = max(0, min(y_index, grid_size - 1))

    x_min = -1 + x_index * division
    y_min = -1 + y_index * division
    x_max = x_min + division
    y_max = y_min + division

    p_i = np.array([x_min, y_min], dtype=float)  # bottom-left
    p_j = np.array([x_max, y_min], dtype=float)  # bottom-right
    p_k = np.array([x_min, y_max], dtype=float)  # top-left
    p_m = np.array([x_max, y_max], dtype=float)  # top-right

    # Local coordinates inside square cell
    sx = (x - x_min) / division
    sy = (y - y_min) / division

    # Same diagonal as your old farthest-corner method: p_j -- p_k
    if sx + sy <= 1.0:
        # lower-left triangle
        return p_i, p_j, p_k
    else:
        # upper-right triangle
        return p_m, p_k, p_j
    

def barycentric_coords(point, p1, p2, p3):

    point = np.asarray(point, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    A = np.column_stack((p2 - p1, p3 - p1))
    xi_eta = np.linalg.solve(A, point - p1)

    xi = xi_eta[0]
    eta = xi_eta[1]

    L1 = 1.0 - xi - eta
    L2 = xi
    L3 = eta

    return L1, L2, L3


def p2_basis(L1, L2, L3):

    return np.array([
        L1 * (2 * L1 - 1),   # vertex i
        L2 * (2 * L2 - 1),   # vertex j
        L3 * (2 * L3 - 1),   # vertex k
        4 * L2 * L3,         # midpoint j-k
        4 * L1 * L3,         # midpoint i-k
        4 * L1 * L2          # midpoint i-j
    ])

class FEM_V2:
    def __init__(self, domain_path, division):
        self.domain = self.get_domain(domain_path)
        self.division = division

    def get_domain(self, domain_path):
        domain = {}

        with open(domain_path, 'r') as f:
            lines = f.readlines()

            # remove header if present
            if not lines[0][0].isdigit() and not lines[0].startswith("-"):
                lines.pop(0)

            for line in lines:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue

                x = float(parts[0])
                y = float(parts[1])
                u = float(parts[2])

                domain[round_key((x, y))] = u

        return domain

    def node_value(self, p):
        key = round_key(p)

        if key not in self.domain:
            raise KeyError(
                f"Node {key} not found in CSV. "
                "This usually means you are using a P1 solution file, "
                "or the midpoint nodes were not written to the CSV."
            )

        return self.domain[key]

    def get_value(self, point):
        point = np.asarray(point, dtype=float)

        # If the point is exactly a stored FEM node, return it directly
        key = round_key(point)
        if key in self.domain:
            return self.domain[key]

        # Find physical triangle
        p_i, p_j, p_k = find_triangle_2(point, self.division)

        # P2 midpoint nodes, matching ordering [i, j, k, m_jk, m_ik, m_ij]
        m_jk = 0.5 * (p_j + p_k)
        m_ik = 0.5 * (p_i + p_k)
        m_ij = 0.5 * (p_i + p_j)

        nodes = [
            p_i,
            p_j,
            p_k,
            m_jk,
            m_ik,
            m_ij
        ]

        U = np.array([self.node_value(p) for p in nodes])

        # Barycentric coordinates relative to p_i, p_j, p_k
        L1, L2, L3 = barycentric_coords(point, p_i, p_j, p_k)

        phi = p2_basis(L1, L2, L3)

        return float(np.dot(U, phi))

    def __call__(self, point):
        return self.get_value(point)

def q3_lagrange_basis_1d(t):

    return np.array([
        -9/16  * (t + 1/3) * (t - 1/3) * (t - 1),
         27/16 * (t + 1)   * (t - 1/3) * (t - 1),
        -27/16 * (t + 1)   * (t + 1/3) * (t - 1),
          9/16 * (t + 1)   * (t + 1/3) * (t - 1/3)
    ], dtype=float)


def q3_basis(xi, eta):

    Lxi = q3_lagrange_basis_1d(xi)
    Leta = q3_lagrange_basis_1d(eta)

    phi = np.zeros(16)

    for j in range(4):
        for i in range(4):
            A = i + 4*j
            phi[A] = Lxi[i] * Leta[j]

    return phi

def find_quad_cell(point, division):

    point = np.asarray(point, dtype=float)
    x, y = point

    grid_size = int(round(2 / division))

    x_index = int(np.floor((x + 1) / division))
    y_index = int(np.floor((y + 1) / division))

    # Clamp for points exactly on x=1 or y=1
    x_index = max(0, min(x_index, grid_size - 1))
    y_index = max(0, min(y_index, grid_size - 1))

    x_min = -1 + x_index * division
    y_min = -1 + y_index * division

    x_max = x_min + division
    y_max = y_min + division

    return x_min, x_max, y_min, y_max


def bicubic_cell_nodes(x_min, x_max, y_min, y_max):
  
    xs = np.linspace(x_min, x_max, 4)
    ys = np.linspace(y_min, y_max, 4)

    nodes = []

    for j in range(4):
        for i in range(4):
            nodes.append(np.array([xs[i], ys[j]], dtype=float))

    return nodes

def physical_to_reference(point, x_min, x_max, y_min, y_max):
    point = np.asarray(point, dtype=float)
    x, y = point

    xi = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    eta = 2.0 * (y - y_min) / (y_max - y_min) - 1.0

    return xi, eta


class FEM_Q3:
    def __init__(self, domain_path, division):

        self.domain = self.get_domain(domain_path)
        self.division = division

    def get_domain(self, domain_path):
        domain = {}

        with open(domain_path, 'r') as f:
            lines = f.readlines()

            if len(lines) == 0:
                return domain

            # Remove header if present
            first = lines[0].strip()
            if len(first) > 0:
                if not first[0].isdigit() and not first.startswith("-"):
                    lines.pop(0)

            for line in lines:
                parts = line.strip().split(',')

                if len(parts) < 3:
                    continue

                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    u = float(parts[2])
                except ValueError:
                    continue

                domain[round_key((x, y))] = u

        return domain

    def node_value(self, p):
        key = round_key(p)

        if key not in self.domain:
            raise KeyError(
                f"Node {key} not found in CSV. "
                "For Q3 bicubic interpolation, the CSV must contain "
                "the 16-node bicubic mesh values, including the two "
                "edge nodes per edge and four interior nodes per quad."
            )

        return self.domain[key]

    def get_value(self, point):
        point = np.asarray(point, dtype=float)

        # If point is exactly a stored FEM node, return it directly
        key = round_key(point)
        if key in self.domain:
            return self.domain[key]

        # Find containing structured quad
        x_min, x_max, y_min, y_max = find_quad_cell(point, self.division)

        # Get 16 Q3 nodes in local order A = i + 4*j
        nodes = bicubic_cell_nodes(x_min, x_max, y_min, y_max)

        # Collect nodal solution values
        U = np.array([self.node_value(p) for p in nodes], dtype=float)

        # Map point to reference square [-1, 1]^2
        xi, eta = physical_to_reference(point, x_min, x_max, y_min, y_max)

        # Evaluate bicubic basis
        phi = q3_basis(xi, eta)

        return float(np.dot(U, phi))

    def __call__(self, point):
        return self.get_value(point)