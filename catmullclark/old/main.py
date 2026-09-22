import numpy as np
import meshio

import pyvista

from mains import *

class face:

    def __init__(self,vertices,vertices_val,vals,idx,he):
        self.idx = idx
        self.vertices=vertices,
        self.vertices_val=vertices_val,
        self.vals=vals,
        self.is_extraordinary = False
        self.extroardinary_vertex = False
        self.he = he



    def check_extraordinary(self,extraordinary_vertices):
        for vertex in self.vertices[0]:
            if vertex in extraordinary_vertices:
                self.is_extraordinary = True
                self.extroardinary_vertex = vertex
                return

    def __eq__(self,other):
        return np.array_equal(self.vertices,other.vertices)

    def __hash__(self):
        return hash(self.idx)

    def __str__(self):
       strng = f"Face {self.idx}, Vertices "
       for v in self.vertices[0]:
           strng += f"{v}, "
       strng += f"Ex: {self.is_extraordinary}"
       return strng

    def print_info(self):
        for v in self.vertices[0]:
            print(f'Vertex: {v} , {self.vertices_val[0][v]}, = {self.vals[0][v]}')


class regular_patch:
    def __init__(self,center,faces,nodes,idx):
        self.center = center
        self.faces = faces
        self.nodes = nodes
        self.idx = idx
        self.is_extraordinary = False

    def __eq__(self,other):
        return np.array_equal(self.faces,other.faces)

    def __hash__(self):
        return hash(self.idx)

    def __str__(self):
       strng = f"Patch {self.idx}, Faces "
       for f in self.faces:
           strng += f"{f.idx}, "
       strng += f"Ex: {self.is_extraordinary}"
       return strng

    def cubic_bspline_1d(self, t):
            return np.array([
                (1 - t) ** 3,
                3 * t**3 - 6 * t**2 + 4,
                -3 * t**3 + 3 * t**2 + 3 * t + 1,
                t**3,
            ]) / 6.0
    
    def evaluate(self, u, v, X):
        # Evaluate the B-spline surface at parameters (u, v)
        Bu = self.cubic_bspline_1d(u)
        Bv = self.cubic_bspline_1d(v)
        N = np.zeros(16)
        for j in range(4):
            for i in range(4):
                N[4 * j + i] = Bu[i] * Bv[j]
        return N @ X.reshape((16, 3))  # Return the evaluated point in 3D space

    def evaluate_s(self, u, v, X):
            # Evaluate the B-spline surface at parameters (u, v)
            Bu = self.cubic_bspline_1d(u)
            Bv = self.cubic_bspline_1d(v)
            N = np.zeros(16)
            for j in range(4):
                for i in range(4):
                    N[4 * j + i] = Bu[i] * Bv[j]
            return N @ X  # Return the evaluated scalar

class irregular_patch:
        def __init__(self,center,faces,idx):
            self.center = center
            self.faces = faces
            #self.nodes = nodes
            self.idx = idx
            self.is_extraordinary = True

        def __eq__(self,other):
            return np.array_equal(self.faces,other.faces)

        def __hash__(self):
            return hash(self.idx)

        def __str__(self):
            strng = f"Patch {self.idx}, Faces "
            for f in self.faces:
                strng += f"{f.idx}, "
            strng += f"Ex: {self.is_extraordinary}"
            return strng


class Surface:
    def __init__(self, path):
        vertices, quads = self.load_mesh(path)
        self.vertices = vertices
        self.quads = quads
        self.edge_list,self.edges,self.edge_faces = self.load_edges()
        self.he, self.he_of = self.generate_half_edges()

        self.U = {}
        self.random_populate()

        self.mesh = pyvista.PolyData(vertices, np.hstack([np.full((quads.shape[0], 1), 4), quads]))
        self.plotter = pyvista.Plotter()

        self.valences = self.load_valences(vertices, quads)

        self.extraordinary_vertices = self.get_extraordinary_vertices(self.valences)

        self.faces = self.load_faces()
        self.extraordinary_faces = self.generate_extraordinary_faces()
        self.regular_faces = self.generate_regular_faces()

        self.regular_patches = self.generate_regular_patches()
        self.irregular_patches = self.generate_irregular_patches()
        #self.second_ring_faces = self.generate_2nd_ring(self.extraordinary_faces, quads)

    def load_mesh(self,path):
        mesh = meshio.read(path)
        Vertices = np.asarray(mesh.points, dtype=float)
        Quads = np.asarray(mesh.cells_dict["quad"], dtype=int)
        return Vertices, Quads

    def load_faces(self):
        faces = []
        for i, quad in enumerate(self.quads):
            half_edges = []
            for he in self.he:
                if he['face'] == i:
                    half_edges.append(he)
            vertices = {}
            vals = {}
            for v in quad:
                vertices[v] = self.vertices[v]
                vals[v] = self.U[v]
            faces.append(face(quad,vertices,vals,i,half_edges))
            faces[-1].check_extraordinary(self.extraordinary_vertices)
        return faces


    def face_to_idx(self, face):
        for i in range(len(self.quads)):
            if np.array_equal(self.quads[i],face):
                return i
        return -1

    def load_valences(self, vertices, quads):
        n_verts = vertices.shape[0]
        valences = np.zeros(n_verts, dtype=int)
        for quad in quads:
            for v in quad:
                valences[v] += 1
        return valences

    def get_extraordinary_vertices(self, valences):
        return np.where(valences != 4)[0]

    def generate_one_ring(self,u_face):
        one_ring = set()
        for face in self.faces:
            for v in u_face.vertices[0]:
                if v in face.vertices[0]:
                    one_ring.add(face)
        
        return list(one_ring)

    def generate_extraordinary_faces(self):
        extraordinary_faces = []
        for f in self.faces:
            if f.is_extraordinary:
                extraordinary_faces.append(f)
        return extraordinary_faces

    def generate_regular_faces(self):
        faces = []
        for face in self.faces:
            if face in self.extraordinary_faces:
                continue
            faces.append(face)
        return faces

    def add_extraordinary_faces(self):

        for patch in self.extraordinary_faces:
            face_indices = [self.face_to_idx(face) for face in patch]
            for idx in face_indices:
                self.add_face(idx)

    def generate_2nd_ring(self, extraordinary_faces, quads):
        second_ring_faces = []
        for faces in extraordinary_faces:
            # Get all vertices in the first ring
            first_ring_vertices = set()
            for face in faces:
                first_ring_vertices.update(face)

            # Find all quads that include any of the first ring vertices
            second_ring = [quad for quad in quads if any(v in first_ring_vertices for v in quad)]
            second_ring_faces.append(second_ring)
        return second_ring_faces

    def load_edges(self):
        edges = {}                 # canonical (min,max) -> edge_id
        edge_faces = {}            # edge_id -> list of (face_id, local_edge_index)
        edge_list = []             # edge_id -> (v0, v1)

        for fi, quad in enumerate(self.quads):
            for e in range(4):
                a, b = quad[e], quad[(e+1) % 4]
                key = (min(a, b), max(a, b))
                if key not in edges:
                    edges[key] = len(edge_list)
                    edge_list.append(key)
                    edge_faces[edges[key]] = []
                edge_faces[edges[key]].append((fi, e))

        return edge_list, edges, edge_faces

    def generate_half_edges(self):
        half_edges = []            # list of dicts
        he_of = {}                 # (from,to) -> half_edge_id

        for fi, quad in enumerate(self.quads):
            base = len(half_edges)
            for e in range(4):
                a, b = quad[e], quad[(e+1) % 4]
                half_edges.append({
                    'from': a, 'to': b, 'face': fi,
                    'next': base + (e+1) % 4,
                    'twin': None
                })
                he_of[(a, b)] = base + e

        # link twins
        for he in half_edges:
            t = he_of.get((he['to'], he['from']))
            he['twin'] = t

        return half_edges, he_of

    def get_half_edges(self,point):
        he_list = []
        for he in self.he:
            if he['from'] == point:
                he_list.append(he)
        return he_list


    def generate_regular_patches(self):
        patches = []
        for i,face in enumerate(self.regular_faces):
            patches.append(regular_patch(face,self.generate_one_ring(face),self.traverse_across(face),i))

        return patches

    def generate_irregular_patches(self):
        patches = []
        for i, face in enumerate(self.extraordinary_faces):
            patches.append(irregular_patch(face,self.generate_one_ring(face),i))

        return patches
    

    def add_surface(self, color='lightblue', opacity=0.5):
        self.plotter.add_mesh(self.mesh, color=color, opacity=opacity, show_edges=True)

    def add_face(self, face, color='orange', opacity=1):
        face = self.mesh.extract_cells(face.idx)
        self.plotter.add_mesh(face, color=color, opacity=opacity, show_edges=True)

    def traverse_face(self,face):
        he = face.he[0]
        point = he['from']
        coordinate = self.vertices[point]
        for _ in range(4):
            print(f'point: {point} = {coordinate}')
            he = self.he[he['next']]
            point = he['from']
            coordinate = self.vertices[point]

    def traverse_across(self, face, steps=16):
    
            he = face.he[0]
            visited = set()
            nodes = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
            path = [5,9,10,6,2,1,0,4,8,12,13,14,15,11,7,3]
    
            idx = 0
    
            while len(visited) < steps:
                point1 = he['from']
                point2 = he['to']
    
                twin = self.he[he['twin']]
    
                if point2 not in visited:
                    visited.add(point1)
                    nodes[path[idx]//4][path[idx]%4] = point1
                    idx += 1

                else:
                    #do twin swap
                    he = twin
    
                # advance forward in the current face
                he = self.he[he['next']]

            return nodes
    
    def traverse_across_p(self, face, steps=16):

        colors = ['r','y','g','b']

        print('Traversing Across Patch')
        print('-----------------------')

        he = face.he[0]
        visited = set()

        idx = 0

        while len(visited) < steps:
            point1 = he['from']
            point2 = he['to']

            twin = self.he[he['twin']]

            t_point1 = twin['from']
            t_point2 = twin['to']

            if point2 not in visited:
                visited.add(point1)
                self.plotter.add_points(self.vertices[point1],color=colors[idx])
                idx = (idx + 1) % 4
                print(f'[{len(visited)}/{steps}] point: {point1} -> {point2}')

            else:
                he = twin
                point1 = he['from']
                point2 = he['to']
                twin = self.he[he['twin']]

                t_point1 = twin['from']
                t_point2 = twin['to']
                print(f'[{len(visited)}/{steps}] point: {point2} -> {point1}')
                print(f'[TAKING TWIN] point: {t_point2} -> {t_point1}')

            # advance forward in the current face
            he = self.he[he['next']]

    def evaluate_patch_s(self,patch,u,v):
        nodes = []
        for row in patch.nodes:
            for point in row:
                nodes.append(self.U[point])
        return patch.evaluate_s(u,v,np.array(nodes))
    
    def evaluate_patch_v(self,patch,u,v):
        nodes = []
        for row in patch.nodes:
            for point in row:
                nodes.append(self.vertices[point])
        return patch.evaluate(u,v,np.array(nodes))

    def add_face_he(self,face,color='orange',opacity=1):
        he = face.he[0]
        face = self.mesh.extract_cells(face.idx)
        self.plotter.add_mesh(face, color=color, opacity=opacity, show_edges=True)
        colors = ['r','y','g','b']
        
        point = he['from']
        coordinate = self.vertices[point]
        self.plotter.add_points(coordinate,color=colors[0])
        for i in range(1,4):
            he = self.he[he['next']]
            point = he['from']
            coordinate = self.vertices[point]
            self.plotter.add_points(coordinate,color=colors[i])

    def add_point(self,point,c='o',o=1):
        self.plotter.add_points(self.vertices[point],color=c,opacity=o)

    def add_point_v(self,point,c='o',o=1):
        self.plotter.add_points(point,color=c,opacity=o)

    def add_points_all(self,opacity=1):
        for v in self.vertices:
            self.plotter.add_points(v,color=self.color_map(self.U[tuple(v)]))

    def add_patch(self,patch,c='orange',o=1):
        for face in patch.faces:
            self.add_face(face,color=c,opacity=o)

    def add_b_surface(self, patch, c="map"):
        X = np.linspace(0, 1, 101)
        Y = np.linspace(0, 1, 101)

        # Generate points array (shape: 101*101, 3)
        points = [self.evaluate_patch_v(patch, x, y) for x in X for y in Y]
        vals = [self.evaluate_patch_s(patch,x,y) for x in X for y in Y]
        points = np.array(points)
        vals = np.array(vals)

        # Reshape into a 2D structured mesh grid (101 x 101 x 3)
        grid_points = points.reshape((101, 101, 3))
        x_coords = grid_points[:, :, 0]
        y_coords = grid_points[:, :, 1]
        z_coords = grid_points[:, :, 2]

        # Create StructuredGrid with exact 10,201 points
        surface = pyvista.StructuredGrid(x_coords, y_coords, z_coords)

        if c == "map":
            colors = np.array([self.color_map(v) for v in vals], dtype=np.uint8)  # shape: (10201, 3)
            surface.point_data["RGB_Colors"] = colors
            self.plotter.add_mesh(surface, scalars="RGB_Colors", rgb=True)
        else:
            self.plotter.add_mesh(surface, color=c)


    def show(self):
            self.plotter.show()

    def clear(self):
            self.plotter.clear()

    def random_populate(self):
        for i in range(len(self.vertices)):
            self.U[i] = np.random.rand()

    def color_map(self,x):
        #Maps 0 -> 1 to RGB Color Blue -> Red
       return [int(255*x),0,int(255*(1-x))]
    

if __name__ == "__main__":
    main4()
