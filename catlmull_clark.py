import numpy as np
import matplotlib.pyplot as plt

# Initial_points = [(-1,-1,0),(-1,1,0),(1,1,0),(1,-1,0),(-1,-1,1),(-1,1,1),(1,1,1),(1,-1,1)]
# Initial_faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]

Initial_points = [(-1,-1,0),(-1,1,0),(1,1,0),(1,-1,0),(0,0,1)]
Initial_faces = [[0,1,2,3],[0,1,4],[1,2,4],[2,3,4],[3,0,4]]



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



plot_mesh(np.array(Initial_points),Initial_faces)

for i in range(5):
    Initial_points, Initial_faces = catmull_clark(Initial_points, Initial_faces)
    plot_mesh(np.array(Initial_points),Initial_faces)
