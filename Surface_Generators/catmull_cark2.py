import numpy as np

def generate_mesh(g=1, hole_width=1, wall_thickness=1, height=1):
    """
    Generates a coarse base quad mesh of genus 'g' using voxel boundary extraction.
    Supports g = 0 (solid box / cube topology).
    """
    if g < 0:
        raise ValueError("Genus must be >= 0.")

    if g == 0:
        # Genus 0: Single solid block (no holes)
        nx = wall_thickness
        ny = wall_thickness
        nz = height
        grid = np.ones((nx, ny, nz), dtype=bool)
    else:
        # Genus g >= 1: Solid block with 'g' parallel through-holes along Z
        nx = wall_thickness * (g + 1) + hole_width * g
        ny = wall_thickness * 2 + hole_width
        nz = height

        grid = np.zeros((nx, ny, nz), dtype=bool)
        grid[:, :, :] = True

        # Carve out 'g' parallel through-holes along the Z axis
        for k in range(g):
            x_start = wall_thickness + k * (hole_width + wall_thickness)
            x_end = x_start + hole_width
            y_start = wall_thickness
            y_end = y_start + hole_width
            grid[x_start:x_end, y_start:y_end, :] = False

    # Extract boundary quads from occupied voxels
    face_directions = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1)
    ]

    cube_face_corners = {
        (-1, 0, 0): [(0,0,0), (0,0,1), (0,1,1), (0,1,0)],
        (1, 0, 0):  [(1,0,0), (1,0,1), (1,1,1), (1,1,0)],
        (0, -1, 0): [(0,0,0), (1,0,0), (1,0,1), (0,0,1)],
        (0, 1, 0):  [(0,1,0), (1,1,0), (1,1,1), (0,1,1)],
        (0, 0, -1): [(0,0,0), (0,1,0), (1,1,0), (1,0,0)],
        (0, 0, 1):  [(0,0,1), (1,0,1), (1,1,1), (0,1,1)]
    }

    raw_quads_pts = []
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if not grid[x, y, z]:
                    continue

                for dx, dy, dz in face_directions:
                    nx_i, ny_i, nz_i = x + dx, y + dy, z + dz
                    is_exposed = False
                    if not (0 <= nx_i < nx and 0 <= ny_i < ny and 0 <= nz_i < nz):
                        is_exposed = True
                    elif not grid[nx_i, ny_i, nz_i]:
                        is_exposed = True

                    if is_exposed:
                        corners = cube_face_corners[(dx, dy, dz)]
                        quad_pts = [(x + cx, y + cy, z + cz) for cx, cy, cz in corners]
                        raw_quads_pts.append(quad_pts)

    # Weld duplicate vertices
    vertex_map = {}
    unique_vertices = []
    quads = []

    for quad in raw_quads_pts:
        face_indices = []
        for pt in quad:
            if pt not in vertex_map:
                vertex_map[pt] = len(unique_vertices)
                unique_vertices.append(pt)
            face_indices.append(vertex_map[pt])
        quads.append(face_indices)

    vertices = np.array(unique_vertices, dtype=float)
    quads = np.array(quads, dtype=int)
    
    # Center geometry around origin (0, 0, 0)
    vertices -= np.mean(vertices, axis=0)

    return vertices, quads

def catmull_clark(vertices, faces):
    vertices = np.array(vertices, dtype=float)
    
    face_points = [np.mean(vertices[f], axis=0) for f in faces]
    face_points = np.array(face_points)

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

    edge_points = []
    edge_key_to_index = {}
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

    v_valences = [0] * len(vertices)
    v_face_sums = np.zeros_like(vertices)
    v_edge_sums = np.zeros_like(vertices)

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
            updated_vertices[v] = (F + 2 * R + (n - 3) * V) / n
        else:
            updated_vertices[v] = vertices[v]

    new_vertices = np.vstack((face_points, edge_points, updated_vertices))

    new_faces = []
    for face_idx, face in enumerate(faces):
        num_verts = len(face)
        face_pt_idx = face_idx
        
        for i in range(num_verts):
            v_current = face[i]
            v_prev = face[(i - 1) % num_verts]
            v_next = face[(i + 1) % num_verts]
            
            orig_pt_idx = vertex_start_idx + v_current
            edge_prev_idx = edge_key_to_index[tuple(sorted((v_prev, v_current)))]
            edge_next_idx = edge_key_to_index[tuple(sorted((v_current, v_next)))]
            
            new_faces.append([orig_pt_idx, edge_prev_idx, face_pt_idx, edge_next_idx])

    return new_vertices, np.array(new_faces)



class Surface:
    def __init__(self, vertices, faces):
        self.vertices = np.array(vertices, dtype=float)
        self.faces = np.array(faces, dtype=int)
        self.face_v = self.faces

    def plot(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        poly = Poly3DCollection([self.vertices[f] for f in self.face_v], alpha=0.6)
        poly.set_edgecolor('k')
        poly.set_facecolor('cyan')
        ax.add_collection3d(poly)

        # Explicitly set 3D bounding box
        min_xyz = self.vertices.min(axis=0)
        max_xyz = self.vertices.max(axis=0)
        max_range = (max_xyz - min_xyz).max() / 2.0
        mid = (max_xyz + min_xyz) / 2.0

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        plt.close()

    def subdivide(self):
        new_vertices, new_faces = catmull_clark(self.vertices, self.faces)
        self.vertices = new_vertices
        self.faces = new_faces
        self.face_v = self.faces

    def normalize_mesh_to_bounds(self, vertices, target_radius=1.2):
        """Rescales mesh vertices to fit within a sphere of target_radius."""
        max_norm = np.max(np.linalg.norm(vertices, axis=1))
        if max_norm > 0:
            vertices = vertices * (target_radius / max_norm)
        return vertices

    def project_to_surface(self, p, surface_fn, max_iters=15, tol=1e-5, max_step=0.2):
        """
        Damped Newton-Raphson level-set projection with step clamping.
        """
        p_proj = np.array(p, dtype=float)
        
        for _ in range(max_iters):
            val, grad = surface_fn(p_proj)
            
            if abs(val) < tol:
                break
                
            grad_norm_sq = np.dot(grad, grad)
            
            # Avoid division by zero at saddle points
            if grad_norm_sq < 1e-8:
                break
                
            # Compute Newton step
            step = (val / (grad_norm_sq + 1e-6)) * grad
            
            # Clamp step size to prevent massive overshoots
            step_mag = np.linalg.norm(step)
            if step_mag > max_step:
                step = step * (max_step / step_mag)
                
            p_proj -= step
            
        return p_proj

    


#==============
#Test Surfaces
#==============

def g_0(p):
    x, y, z = p[0], p[1], p[2]
    val = (x**2+y**2+z**2)**2 - x - z**2
    grad = np.array([2*(x**2+y**2+z**2)*2*x - 1, 2*(x**2+y**2+z**2)*2*y, 2*(x**2+y**2+z**2)*2*z - 2*z])
    return val, grad

def g_1(p,R=1.0, r0=0.3):
    x, y, z = p[0], p[1], p[2]
    theta = np.arctan2(y, x)
    
    # Radius fluctuates along the ring
    r_wave = r0 * (1.0 + 0.4 * np.cos(5 * theta))
    
    # Distance from major ring (R)
    r_xy = np.sqrt(x**2 + y**2)
    if r_xy < 1e-12:
        r_xy = 1e-12
        
    val = (r_xy - R)**2 + z**2 - r_wave**2
    
    # Numerical gradient fallback or analytical gradient
    eps = 1e-6
    dx = ((np.sqrt((x+eps)**2 + y**2) - R)**2 + z**2 - (r0*(1+0.4*np.cos(5*np.arctan2(y, x+eps))))**2 - val) / eps
    dy = ((np.sqrt(x**2 + (y+eps)**2) - R)**2 + z**2 - (r0*(1+0.4*np.cos(5*np.arctan2(y+eps, x))))**2 - val) / eps
    dz = 2 * z
    
    return val, np.array([dx, dy, dz])

def g_2(p):
    x, y, z = p[0], p[1], p[2]
    
    f1 = x**2 + y**2 - 1.0
    f2 = z**2 - 0.25
    f3 = y**2 + z**2 - 1.0
    
    val = f1**2 + f2**2 + f3**2 - 0.6
    
    df_dx = 2 * f1 * (2 * x)
    df_dy = 2 * f1 * (2 * y) + 2 * f3 * (2 * y)
    df_dz = 2 * f2 * (2 * z) + 2 * f3 * (2 * z)
    
    return val, np.array([df_dx, df_dy, df_dz])

def g_3(p):
    x, y, z = p[0], p[1], p[2]
    
    r2 = x**2 + y**2 + z**2
    val = x**4 + y**4 + z**4 - r2 + 0.45
    
    grad = np.array([
        4 * x**3 - 2 * x,
        4 * y**3 - 2 * y,
        4 * z**3 - 2 * z
    ])
    
    return val, grad

if __name__ == "__main__":       

    # 1. Generate  mesh
    vertices, faces = generate_mesh(g=1, hole_width=1, wall_thickness=1, height=1)
    surf = Surface(vertices, faces)

    # 2. Rescale initial vertices to match the mathematical surface scale
    vertices = surf.normalize_mesh_to_bounds(vertices, target_radius=1.2)
    #surf.plot()

    

    # 3. Subdivide and project safely
    for _ in range(1):
        #surf.subdivide()
        for i in range(len(surf.vertices)):
            surf.vertices[i] = surf.project_to_surface(
                surf.vertices[i], 
                g_1, 
                max_step=0.15
            )
        surf.plot()
