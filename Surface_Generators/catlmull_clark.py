import numpy as np
import matplotlib.pyplot as plt

# Initial_points = [(-1,-1,0),(-1,1,0),(1,1,0),(1,-1,0),(-1,-1,1),(-1,1,1),(1,1,1),(1,-1,1)]
# Initial_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

# Initial_points = [(-1,-1,0),(-1,1,0),(1,1,0),(1,-1,0),(0,0,1)]
# Initial_faces = [[0,1,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]



def plot_mesh(vertices,faces):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for face in faces:
    x = [vertices[face[i],0] for i in range(len(face))]
    y = [vertices[face[i],1] for i in range(len(face))]
    z = [vertices[face[i],2] for i in range(len(face))]
    x.append(vertices[face[0],0])
    y.append(vertices[face[0],1])
    z.append(vertices[face[0],2])
    ax.plot(x,y,z)
  plt.show()


import numpy as np

def catmull_clark(vertices, faces):
    # Ensure inputs are numpy arrays for easy math
    vertices = np.array(vertices, dtype=float)
    
    # -------------------------------------------------------------------------
    # STEP 1: Compute Face Points
    # -------------------------------------------------------------------------

    face_points = []
    for face in faces:
        face_verts = vertices[face]
        face_points.append(np.mean(face_verts, axis=0))
    face_points = np.array(face_points)

    # -------------------------------------------------------------------------
    # STEP 2: Compute Edge Points
    # -------------------------------------------------------------------------


    edges = {}
    
    for face_idx, face in enumerate(faces):
        num_verts = len(face)
        for i in range(num_verts):
            v1 = face[i]
            v2 = face[(i + 1) % num_verts]
            edge_key = tuple(sorted((v1, v2)))
            
            if edge_key not in edges:
                edges[edge_key] = {'faces': [face_idx]}
            else:
                edges[edge_key]['faces'].append(face_idx)

    # Now calculate the actual smooth edge points
    edge_points = []
    edge_key_to_index = {} # Maps (v1, v2) to its index in the new vertex list
    
    # Base offset for indexing: edge points will sit after face points
    edge_start_idx = len(face_points)
    
    for idx, (edge_key, data) in enumerate(edges.items()):
        v1, v2 = edge_key
        shared_faces = data['faces']
        

        if len(shared_faces) == 2:
            fp1 = face_points[shared_faces[0]]
            fp2 = face_points[shared_faces[1]]
            edge_pt = (vertices[v1] + vertices[v2] + fp1 + fp2) / 4.0
        else:
            edge_pt = (vertices[v1] + vertices[v2]) / 2.0
            
        edge_points.append(edge_pt)
        edge_key_to_index[edge_key] = edge_start_idx + idx
        
    edge_points = np.array(edge_points)

    # -------------------------------------------------------------------------
    # STEP 3: Update Original Vertices
    # -------------------------------------------------------------------------


    v_valences = [0] * len(vertices)
    v_face_sums = np.zeros_like(vertices)
    v_edge_sums = np.zeros_like(vertices) # Sum of original edge midpoints (not smooth edge points)

    for face_idx, face in enumerate(faces):
        fp = face_points[face_idx]
        for v in face:
            v_valences[v] += 1
            v_face_sums[v] += fp

    for edge_key in edges.keys():
        v1, v2 = edge_key
        midpoint = (vertices[v1] + vertices[v2]) / 2.0
        v_edge_sums[v1] += midpoint
        v_edge_sums[v2] += midpoint

    updated_vertices = np.zeros_like(vertices)
    vertex_start_idx = len(face_points) + len(edge_points)

    for v in range(len(vertices)):
        n = v_valences[v]
        if n > 0:
            F = v_face_sums[v] / n
            R = v_edge_sums[v] / n
            V = vertices[v]
            # Catmull-Clark vertex update formula: (F + 2R + (n-3)V) / n
            updated_vertices[v] = (F + 2 * R + (n - 3) * V) / n
        else:
            updated_vertices[v] = vertices[v]


    new_vertices = np.vstack((face_points, edge_points, updated_vertices))

    # -------------------------------------------------------------------------
    # STEP 4: Build the New Topology (Faces)
    # -------------------------------------------------------------------------

    new_faces = []
    for face_idx, face in enumerate(faces):
        num_verts = len(face)
        face_pt_idx = face_idx # Face points are at the very beginning
        
        for i in range(num_verts):
            v_current = face[i]
            v_prev = face[(i - 1) % num_verts]
            v_next = face[(i + 1) % num_verts]
            
            # Get the global index of the updated original vertex
            orig_pt_idx = vertex_start_idx + v_current
            
            # Get global indices of the two edge points connected to this vertex
            edge_prev_idx = edge_key_to_index[tuple(sorted((v_prev, v_current)))]
            edge_next_idx = edge_key_to_index[tuple(sorted((v_current, v_next)))]
            
            # Form a new quad face (ordered counter-clockwise)
            new_faces.append([orig_pt_idx, edge_next_idx, face_pt_idx, edge_prev_idx])

    return new_vertices, new_faces

class Surface:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices)
        self.faces = faces

    def subdivide(self):
        self.vertices, self.faces = catmull_clark(self.vertices, self.faces)

    def plot(self):
        plot_mesh(self.vertices, self.faces)


import numpy as np

def make_base_torus(R=1.0, r=0.4, n_around=4, n_tube=4):
    """
    Creates a coarse base quad mesh for a Genus 1 Torus.
    R: Major radius (distance from center of hole to tube center)
    r: Minor radius (radius of the tube)
    n_around: divisions around the major ring (min 3 or 4)
    n_tube: divisions around the tube cross-section (min 3 or 4)
    """
    u = np.linspace(0, 2 * np.pi, n_around, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_tube, endpoint=False)
    
    vertices = []
    for i in range(n_around):
        for j in range(n_tube):
            # Parametric Torus equation
            x = (R + r * np.cos(v[j])) * np.cos(u[i])
            y = (R + r * np.cos(v[j])) * np.sin(u[i])
            z = r * np.sin(v[j])
            vertices.append([x, y, z])
            
    vertices = np.array(vertices, dtype=float)
    
    quads = []
    for i in range(n_around):
        i_next = (i + 1) % n_around
        for j in range(n_tube):
            j_next = (j + 1) % n_tube
            
            v0 = i * n_tube + j
            v1 = i_next * n_tube + j
            v2 = i_next * n_tube + j_next
            v3 = i * n_tube + j_next
            
            quads.append([v0, v1, v2, v3])
            
    return vertices, np.array(quads, dtype=int)

import numpy as np

def generate_multi_genus_base_mesh(g=1, hole_width=2, wall_thickness=1, height=2):
    """
    Generates a coarse base quad mesh of genus 'g' using voxel boundary extraction.
    
    Parameters:
      g : int
          Target genus (number of holes).
      hole_width : int
          Grid size of each rectangular hole cross-section.
      wall_thickness : int
          Grid thickness between holes and outer walls.
      height : int
          Grid height (Z dimension) of the structure.
          
    Returns:
      vertices : ndarray (N, 3)
      quads : ndarray (M, 4)
    """
    if g < 1:
        raise ValueError("Genus must be >= 1 for this generator.")

    # Calculate grid dimensions to accommodate 'g' holes side-by-side
    nx = wall_thickness * (g + 1) + hole_width * g
    ny = wall_thickness * 2 + hole_width
    nz = height

    grid = np.zeros((nx, ny, nz), dtype=bool)

    # 1. Fill solid bounding box
    grid[:, :, :] = True

    # 2. Carve out 'g' parallel through-holes along the Z axis
    for k in range(g):
        x_start = wall_thickness + k * (hole_width + wall_thickness)
        x_end = x_start + hole_width
        y_start = wall_thickness
        y_end = y_start + hole_width

        # Set hole voxels to empty
        grid[x_start:x_end, y_start:y_end, :] = False

    # 3. Extract boundary quads from occupied voxels
    # Define face orientations (direction, axis, fixed offset relative to voxel base)
    face_directions = [
        (-1, 0, 0), (1, 0, 0),  # -X, +X
        (0, -1, 0), (0, 1, 0),  # -Y, +Y
        (0, 0, -1), (0, 0, 1)   # -Z, +Z
    ]

    # Template corner offsets for a unit cube [0, 1]^3 for each direction (counter-clockwise orientation)
    cube_face_corners = {
        (-1, 0, 0): [(0,0,0), (0,0,1), (0,1,1), (0,1,0)],
        (1, 0, 0):  [(1,0,0), (1,1,0), (1,1,1), (1,0,1)],
        (0, -1, 0): [(0,0,0), (1,0,0), (1,0,1), (0,0,1)],
        (0, 1, 0):  [(0,1,0), (0,1,1), (1,1,1), (1,1,0)],
        (0, 0, -1): [(0,0,0), (0,1,0), (1,1,0), (1,0,0)],
        (0, 0, 1):  [(0,0,1), (1,0,1), (1,1,1), (0,0,1)]
    }

    raw_quads_pts = []

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not grid[x, y, z]:
                    continue  # Skip empty voxels

                # Check all 6 neighbors
                for dx, dy, dz in face_directions:
                    nx_i, ny_i, nz_i = x + dx, y + dy, z + dz

                    # Face is exposed if neighbor is out of bounds or empty
                    is_exposed = False
                    if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                        is_exposed = True
                    elif not grid[nx_i, ny_i, nz_i]:
                        is_exposed = True

                    if is_exposed:
                        corners = cube_face_corners[(dx, dy, dz)]
                        quad_pts = [(x + cx, y + cy, z + cz) for cx, cy, cz in corners]
                        raw_quads_pts.append(quad_pts)

    # 4. Remove duplicate vertices (welding)
    vertex_map = {}
    unique_vertices = []
    quads = []

    for quad in raw_quads_pts:
        face_indices = []
        for pt in quad:
            # Map point coordinate to unique index
            if pt not in vertex_map:
                vertex_map[pt] = len(unique_vertices)
                unique_vertices.append(pt)
            face_indices.append(vertex_map[pt])
        quads.append(face_indices)

    vertices = np.array(unique_vertices, dtype=float)
    quads = np.array(quads, dtype=int)

    # Optional: Center geometry around origin (0, 0, 0)
    vertices -= np.mean(vertices, axis=0)

    return vertices, quads

Initial_points, Initial_faces = generate_mesh(g=1, hole_width=2, wall_thickness=1, height=2)

surface = Surface(Initial_points, Initial_faces)
surface.plot()

for i in range(3):
    surface.subdivide()
    surface.plot()