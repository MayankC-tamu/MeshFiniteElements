import numpy as np
import meshio
import pyvista

COLORS = ['r', 'y', 'g', 'b']

import mains


class Face:
    def __init__(self, vertices, vertices_val, vals, idx, he):
        self.idx = idx
        self.vertices = vertices,
        self.vertices_val = vertices_val,
        self.vals = vals,
        self.is_extraordinary = False
        self.extraordinary_vertex = False
        self.he = he

    def check_extraordinary(self, extraordinary_vertices):
        for vertex in self.vertices[0]:
            if vertex in extraordinary_vertices:
                self.is_extraordinary = True
                self.extraordinary_vertex = vertex
                return

    def print_info(self):
        for v in self.vertices[0]:
            print(f'Vertex: {v} , {self.vertices_val[0][v]}, = {self.vals[0][v]}')

    def __eq__(self, other):
        return np.array_equal(self.vertices, other.vertices)

    def __hash__(self):
        return hash(self.idx)

    def __str__(self):
        verts = "".join(f"{v}, " for v in self.vertices[0])
        return f"Face {self.idx}, Vertices {verts}Ex: {self.is_extraordinary}"


class Patch:
    """Shared base for regular / irregular patches (equality, hashing, str)."""

    def __init__(self, center, faces, nodes, idx, is_extraordinary):
        self.center = center
        self.faces = faces
        self.nodes = nodes
        self.idx = idx
        self.is_extraordinary = is_extraordinary

    def __eq__(self, other):
        return np.array_equal(self.faces, other.faces)

    def __hash__(self):
        return hash(self.idx)

    def __str__(self):
        faces = "".join(f"{f.idx}, " for f in self.faces)
        return f"Patch {self.idx}, Faces {faces}Ex: {self.is_extraordinary}"


class RegularPatch(Patch):
    def __init__(self, center, faces, nodes, idx):
        super().__init__(center, faces, nodes, idx, is_extraordinary=False)

    @staticmethod
    def cubic_bspline_1d(t):
        return np.array([
            (1 - t) ** 3,
            3 * t ** 3 - 6 * t ** 2 + 4,
            -3 * t ** 3 + 3 * t ** 2 + 3 * t + 1,
            t ** 3,
        ]) / 6.0

    def _basis(self, u, v):
        """16-element tensor-product cubic B-spline basis at (u, v)."""
        Bu = self.cubic_bspline_1d(u)
        Bv = self.cubic_bspline_1d(v)
        N = np.zeros(16)
        for j in range(4):
            for i in range(4):
                N[4 * j + i] = Bu[i] * Bv[j]
        return N

    def evaluate_v(self, u, v, X):
        # Evaluate the B-spline surface (3D point) at parameters (u, v)
        return self._basis(u, v) @ X.reshape((16, 3))

    def evaluate_s(self, u, v, X):
        # Evaluate the B-spline surface (scalar) at parameters (u, v)
        return self._basis(u, v) @ X


class IrregularPatch(Patch):
    def __init__(self, center, faces, nodes, idx):
        super().__init__(center, faces, nodes, idx, is_extraordinary=True)


class Surface:
    def __init__(self, path):
        vertices, quads = self.load_mesh(path)
        self.vertices = vertices
        self.quads = quads
        self.edge_list, self.edges, self.edge_faces = self.load_edges()
        self.he, self.he_of = self.generate_half_edges()

        self.U = {}
        self.random_populate()

        self.mesh = pyvista.PolyData(
            vertices, np.hstack([np.full((quads.shape[0], 1), 4), quads])
        )
        self.plotter = pyvista.Plotter()

        self.valences = self.load_valences(vertices, quads)
        self.extraordinary_vertices = self.get_extraordinary_vertices(self.valences)

        self.faces = self.load_faces()
        self.extraordinary_faces = self.generate_extraordinary_faces()
        self.regular_faces = self.generate_regular_faces()

        self.regular_patches = self.generate_regular_patches()
        self.irregular_patches = self.generate_irregular_patches()

    # ---------------------------------------------------------------- 
    # loading
    # ---------------------------------------------------------------- 
    def load_mesh(self, path):
        mesh = meshio.read(path)
        vertices = np.asarray(mesh.points, dtype=float)
        quads = np.asarray(mesh.cells_dict["quad"], dtype=int)
        return vertices, quads

    def load_edges(self):
        edges = {}          # canonical (min,max) -> edge_id
        edge_faces = {}     # edge_id -> list of (face_id, local_edge_index)
        edge_list = []      # edge_id -> (v0, v1)

        for fi, quad in enumerate(self.quads):
            for e in range(4):
                a, b = quad[e], quad[(e + 1) % 4]
                key = (min(a, b), max(a, b))
                if key not in edges:
                    edges[key] = len(edge_list)
                    edge_list.append(key)
                    edge_faces[edges[key]] = []
                edge_faces[edges[key]].append((fi, e))

        return edge_list, edges, edge_faces

    def generate_half_edges(self):
        half_edges = []     # list of dicts
        he_of = {}          # (from,to) -> half_edge_id

        for fi, quad in enumerate(self.quads):
            base = len(half_edges)
            for e in range(4):
                a, b = quad[e], quad[(e + 1) % 4]
                half_edges.append({
                    'from': a, 'to': b, 'face': fi,
                    'next': base + (e + 1) % 4,
                    'twin': None,
                })
                he_of[(a, b)] = base + e

        # link twins
        for he in half_edges:
            he['twin'] = he_of.get((he['to'], he['from']))

        return half_edges, he_of

    def load_faces(self):
        faces = []
        for i, quad in enumerate(self.quads):
            half_edges = [he for he in self.he if he['face'] == i]
            vertices = {v: self.vertices[v] for v in quad}
            vals = {v: self.U[v] for v in quad}
            f = Face(quad, vertices, vals, i, half_edges)
            f.check_extraordinary(self.extraordinary_vertices)
            faces.append(f)
        return faces

    def load_valences(self, vertices, quads):
        valences = np.zeros(vertices.shape[0], dtype=int)
        for quad in quads:
            for v in quad:
                valences[v] += 1
        return valences

    def get_extraordinary_vertices(self, valences):
        return np.where(valences != 4)[0]

    def random_populate(self):
        for i in range(len(self.vertices)):
            self.U[i] = np.random.rand()

    # -------------------------------------------------------------- 
    # topology
    # -------------------------------------------------------------- 
    def get_half_edges(self, point):
        return [he for he in self.he if he['from'] == point]

    def face_to_idx(self, face):
        for i in range(len(self.quads)):
            if np.array_equal(self.quads[i], face):
                return i
        return -1

    def generate_one_ring(self, u_face):
        one_ring = set()
        for face in self.faces:
            for v in u_face.vertices[0]:
                if v in face.vertices[0]:
                    one_ring.add(face)
        return list(one_ring)

    def generate_extraordinary_faces(self):
        return [f for f in self.faces if f.is_extraordinary]

    def generate_regular_faces(self):
        return [f for f in self.faces if f not in self.extraordinary_faces]

    def generate_regular_patches(self):
        return [
            RegularPatch(face, self.generate_one_ring(face),
                         self.traverse_across(face), i)
            for i, face in enumerate(self.regular_faces)
        ]

    def generate_irregular_patches(self):
        return [
            IrregularPatch(face, self.generate_one_ring(face), 
                           self.traverse_across_e(face),i)
            for i, face in enumerate(self.extraordinary_faces)
        ]

    # -------------------------------------------------------------- 
    # traversal
    # -------------------------------------------------------------- 

    def traverse(self, he, idx, plot=True, twin=False):

        if not twin:
            he = self.he[he['next']]
            p = he['from']
            if plot:
                self.add_point(p, c=COLORS[idx])
                idx = (idx + 1) % 4
            return he, idx

        he = self.he[he['twin']]
        he = self.he[he['next']]
        return he, idx


    def traverse_face(self, face):
        he = min(face.he, key=lambda h: h['from'])
        for _ in range(4):
            point = he['from']
            print(f'point: {point} = {self.vertices[point]}')
            he = self.he[he['next']]

    def _traverse_ring(self, face, steps, on_visit):
 
        he = min(face.he, key=lambda h: h['from'])
        visited = set()

        while len(visited) < steps:
            point1 = he['from']
            point2 = he['to']
            twin = self.he[he['twin']]

            if point2 not in visited:
                visited.add(point1)
                on_visit(len(visited) - 1, point1, point2)
                he = self.he[he['next']]  # advance forward in current face
            else:
                # do twin swap, then advance in the new face
                he = twin
                he = self.he[he['next']]

    def traverse_across(self, face, steps=16):
        """Return the 4x4 control-node grid for a regular patch."""
        nodes = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        path = [5, 9, 10, 6, 2, 1, 0, 4, 8, 12, 13, 14, 15, 11, 7, 3]

        def record(order, point, _to):
            nodes[path[order] // 4][path[order] % 4] = point

        self._traverse_ring(face, steps, record)
        return nodes

    def traverse_across_p(self, face, steps=16):
        """Visualize the ring traversal (plots each visited point)."""
        print('Traversing Across Patch')
        print('-----------------------')

        def record(order, point, to):
            self.add_point(point, c=COLORS[order % 4])
            print(f'[{order + 1}/{steps}] point: {point} -> {to}')

        self._traverse_ring(face, steps, record)


    def traverse_across_e(self,face):
        nodes = []
        p = face.extraordinary_vertex
        p0 = p
        he = self.get_half_edges(p)[0]
        he0 = he
    
        p = he['from']
        idx = 1
    
        #Locate Diagonal
        #------------------------
        he,idx = self.traverse(he,idx,False)
        he,idx = self.traverse(he,idx,False)
        he,idx = self.traverse(he,idx,False,True)
        he,idx = self.traverse(he,idx,False,True)
        he,idx = self.traverse(he,idx,False)
        hef_t = self.he[he['twin']]
        he,idx = self.traverse(he,idx,False)
        hef = he
        #------------------------
    
        #Ring Traversal
        #------------------------
        he = he0
        visited = set()
        visited.add(he['from'])

        steps = 2 * self.valences[p] + 1
            
        while len(visited) < steps:
            point1 = he['from']
            point2 = he['to']
    
            twin = self.he[he['twin']]
    
            if point2 not in visited:
                    visited.add(point1)
                    nodes.append(point1)
                    if (len(visited) < steps):
                            (_, idx) = self.traverse(he,idx,False,True)
    
            else:
                    he = twin
                    point1 = he['from']
                    point2 = he['to']
                    twin = self.he[he['twin']]

            # advance forward in the current face
            he = self.he[he['next']]


                    
        swap = [0,3,2,1,6,5,4]
        node_temp = []

        for n in nodes:
            node_temp.append(n)

        for i in range(len(swap)):
            nodes[i] = node_temp[swap[i]]
        

        #Jump
        #------------------------
        visited.add(hef['from'])
        nodes.append(hef['from'])
        steps += 7   
    
       
        # Left Traversal
        #------------------------

        he = hef_t
    
        visited.add(he['to'])
        nodes.append(he['to'])
        (he,idx) = self.traverse(he,idx,False)
    
        (he, idx) = self.traverse(he,idx,False,True)
    
        visited.add(he['to'])
        nodes.append(he['to'])
        (he, idx) = self.traverse(he,idx,False)
    
        (he, idx) = self.traverse(he,idx,False,True)
    
        visited.add(he['to'])
        nodes.append(he['to'])
        (he,idx) = self.traverse(he,idx,False)

    
        # Right Traversal
        #------------------------
        he = hef # Jump Again
    
        visited.add(he['to'])
        nodes.append(he['to'])
        (he,idx) = self.traverse(he,idx,False)

        (he, idx) = self.traverse(he,idx,False,True)

        visited.add(he['to'])
        nodes.append(he['to'])
        (he, idx) = self.traverse(he,idx,False)

        (he, idx) = self.traverse(he,idx,False,True)

        visited.add(he['to'])
        nodes.append(he['to'])
        (he, idx) = self.traverse(he,idx,False)

        return nodes

    # -------------------------------------------------------------- 
    # evaluation
    # -------------------------------------------------------------- 
    def _patch_node_data(self, patch, source):
        """Flatten a patch's node grid, pulling values from `source` dict."""
        return np.array([source[point] for row in patch.nodes for point in row])

    def evaluate_patch_s(self, patch, u, v):
        return patch.evaluate_s(u, v, self._patch_node_data(patch, self.U))

    def evaluate_patch_v(self, patch, u, v):
        nodes = []
        for n in patch.nodes:
            nodes.append(self.vertices[n])
        return patch.evaluate_v(u, v, np.array(nodes))

    # ---------------------------------------------------------------- 
    # plotting
    # ----------------------------------------------------------------
    def add_surface(self, color='lightblue', opacity=0.5):
        self.plotter.add_mesh(self.mesh, color=color, opacity=opacity, show_edges=True)

    def add_face(self, face, color='orange', opacity=1):
        cell = self.mesh.extract_cells(face.idx)
        self.plotter.add_mesh(cell, color=color, opacity=opacity, show_edges=True)

    def add_face_he(self, face, color='orange', opacity=1):
        he = face.he[0]
        self.add_face(face, color=color, opacity=opacity)
        # color each corner vertex in traversal order
        self.add_point(he['from'], c=COLORS[0])
        for i in range(1, 4):
            he = self.he[he['next']]
            self.add_point(he['from'], c=COLORS[i])

    def add_point(self, point, c='o', o=1):
        self.plotter.add_points(self.vertices[point], color=c, opacity=o)

    def add_point_v(self, point, c='o', o=1):
        self.plotter.add_points(point, color=c, opacity=o)

    def add_points_all(self, opacity=1):
        for v in self.vertices:
            self.plotter.add_points(v, color=self.color_map(self.U[tuple(v)]))

    def add_patch(self, patch, c='orange', o=1):
        for face in patch.faces:
            self.add_face(face, color=c, opacity=o)

    def add_extraordinary_faces(self):
        for patch in self.extraordinary_faces:
            face_indices = [self.face_to_idx(face) for face in patch]
            for idx in face_indices:
                self.add_face(idx)

    def add_b_surface(self, patch, c="map"):
        X = np.linspace(0, 1, 101)
        Y = np.linspace(0, 1, 101)

        # Generate points array (shape: 101*101, 3)
        points = np.array([self.evaluate_patch_v(patch, x, y) for x in X for y in Y])
        vals = np.array([self.evaluate_patch_s(patch, x, y) for x in X for y in Y])

        # Reshape into a 2D structured mesh grid (101 x 101 x 3)
        grid_points = points.reshape((101, 101, 3))
        x_coords = grid_points[:, :, 0]
        y_coords = grid_points[:, :, 1]
        z_coords = grid_points[:, :, 2]

        # Create StructuredGrid with exact 10,201 points
        surface = pyvista.StructuredGrid(x_coords, y_coords, z_coords)

        if c == "map":
            colors = np.array([self.color_map(v) for v in vals], dtype=np.uint8)
            surface.point_data["RGB_Colors"] = colors
            self.plotter.add_mesh(surface, scalars="RGB_Colors", rgb=True)
        else:
            self.plotter.add_mesh(surface, color=c)

    def show(self):
        self.plotter.show()

    def clear(self):
        self.plotter.clear()

    def color_map(self, x):
        # Maps 0 -> 1 to an RGB color Blue -> Red
        return [int(255 * x), 0, int(255 * (1 - x))]

if __name__ == "__main__":
    mains.main8()
