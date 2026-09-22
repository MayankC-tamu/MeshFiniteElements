import numpy as np
import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import json
import os
import trimesh

#=================================================================
# Bezier surface computation
#=================================================================

def bernstein_poly(i, n, t):
    return math.comb(n, i) * (t**i) * ((1 - t) ** (n - i))

def compute_bezier_surface(control_points, resolution=24):
    u_vals = np.linspace(0, 1, resolution)
    v_vals = np.linspace(0, 1, resolution)

    B_u = np.array([[bernstein_poly(i, 3, u) for i in range(4)] for u in u_vals])
    B_v = np.array([[bernstein_poly(j, 3, v) for j in range(4)] for v in v_vals])

    surface_points = np.zeros((resolution, resolution, 3))
    for i in range(3): 
        surface_points[:, :, i] = B_u @ control_points[:, :, i] @ B_v.T

    return surface_points
#=================================================================

#=================================================================
# GUI Helpers
#=================================================================

class Camera:
    def __init__(self, position, target, up, fov=45.0, min_fov=10.0, max_fov=120.0):
        self.position = np.array(position, dtype=np.float32)
        self.target = np.array(target, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)

        # Field of view (degrees)
        self.fov = float(fov)
        self.min_fov = float(min_fov)
        self.max_fov = float(max_fov)

        # Spherical coordinates for orbiting
        offset = self.position - self.target
        self.radius = float(np.linalg.norm(offset)) if np.linalg.norm(offset) != 0 else 1.0
        # azimuth (degrees) and elevation (degrees)
        self.angle_x = float(np.degrees(np.arctan2(offset[0], offset[2]))) if self.radius != 0 else 0.0
        self.angle_y = float(np.degrees(np.arcsin(np.clip(offset[1] / self.radius, -1.0, 1.0)))) if self.radius != 0 else 0.0

    def apply(self):
        # Recompute position from spherical coordinates before applying view
        x = self.target[0] + self.radius * np.cos(np.radians(self.angle_y)) * np.sin(np.radians(self.angle_x))
        y = self.target[1] + self.radius * np.sin(np.radians(self.angle_y))
        z = self.target[2] + self.radius * np.cos(np.radians(self.angle_y)) * np.cos(np.radians(self.angle_x))
        self.position = np.array([x, y, z], dtype=np.float32)

        gluLookAt(
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
            float(self.up[0]), float(self.up[1]), float(self.up[2])
        )

    def update_angles(self, delta_x, delta_y, sensitivity=0.3):
        # delta_x/delta_y are pixel differences from mouse motion
        self.angle_x += delta_x * sensitivity
        self.angle_y += delta_y * sensitivity
        # Clamp elevation to avoid gimbal flip
        self.angle_y = float(np.clip(self.angle_y, -89.9, 89.9))

    # Backwards compatible alias
    def update_camera(self, delta_x, delta_y):
        return self.update_angles(delta_x, delta_y)

    def change_fov(self, delta, sensitivity=2.5):
        # Positive delta -> scroll up -> zoom in (decrease fov)
        self.fov -= float(delta) * sensitivity
        self.fov = float(np.clip(self.fov, self.min_fov, self.max_fov))
        
    
def draw_point(point):
    glPointSize(1.0)
    glColor3f(1.0, 1.0, 1.0) 
    glBegin(GL_POINTS)
    glVertex3fv(point)
    glEnd()

def draw_node(node):
    glPointSize(10.0)
    glColor3f(0.0, 1.0, 0.0) 
    glBegin(GL_POINTS)
    glVertex3fv(node)
    glEnd()

def draw_selected_node(node):
    glPointSize(14.0)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_POINTS)
    glVertex3fv(node)
    glEnd()

def draw_control_polygon(control_points, selected_idx=None):
    # control_points expected shape (4,4,3) or flat iterable (N,3)
    cp = np.asarray(control_points)
    if cp.ndim == 2 and cp.shape[1] == 3:
        for p in cp:
            draw_node(p)
        return

    for i in range(cp.shape[0]):
        for j in range(cp.shape[1]):
            p = cp[i, j]
            if selected_idx is not None and (i, j) == selected_idx:
                draw_selected_node(p)
            else:
                draw_node(p)

def draw_surface(surface_points):
    for point in surface_points.reshape(-1, 3):
        draw_point(point)

def pick_control_point(control_points, mouse_pos, width, height, threshold=12):
    # returns (i, j, winz, screen_x, screen_y) or None
    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
    projection = glGetDoublev(GL_PROJECTION_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    best = None
    best_dist = threshold
    cp = np.asarray(control_points)
    for i in range(cp.shape[0]):
        for j in range(cp.shape[1]):
            x, y, z = float(cp[i, j, 0]), float(cp[i, j, 1]), float(cp[i, j, 2])
            win = gluProject(x, y, z, modelview, projection, viewport)
            winx, winy, winz = win[0], win[1], win[2]
            screen_x = winx
            screen_y = height - winy
            dx = screen_x - mouse_pos[0]
            dy = screen_y - mouse_pos[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = (i, j, winz, screen_x, screen_y)
    return best


def save_points_and_obj(control_points, surface_resolution=100, cp_filename='./meshes/bezier_config.json', obj_filename='./meshes/bezier_surface.obj'):
    cp_path = os.path.abspath(cp_filename)
    obj_path = os.path.abspath(obj_filename)

    json.dump(np.asarray(control_points).tolist(), open(cp_path, 'w'), indent=2)
    print(f"Saved control points to {cp_path}")

    surf = compute_bezier_surface(np.asarray(control_points), resolution=100)
    verts = surf.reshape(-1, 3)
    res = surface_resolution
    faces = []
    for i in range(res - 1):
        for j in range(res - 1):
            idx = i * res + j
            idx_right = idx + 1
            idx_down = (i + 1) * res + j
            idx_down_right = idx_down + 1
            faces.append([idx, idx_down, idx_right])
            faces.append([idx_right, idx_down, idx_down_right])

    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(obj_path)
    print(f"Exported OBJ via trimesh to {obj_path}")

#=================================================================

def main():

    #---- Setup ----

    Camera_position = [0.0, 0.0, 5.0]
    camera = Camera(position=Camera_position, target=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0], fov=45.0)

    pygame.init()
    pygame.display.set_caption('Bezier Surface Viewer')
    display = (800, 600)
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    width, height = display

    points = np.array([
        [[-1, -1, 0], [-0.5, -1, 1], [0.5, -1, 1], [1, -1, 0]],
        [[-1, -0.5, 1], [-0.5, -0.5, 2], [0.5, -0.5, 2], [1, -0.5, 1]],
        [[-1, 0.5, 1], [-0.5, 0.5, 2], [0.5, 0.5, 2], [1, 0.5, 1]],
        [[-1, 1, 0], [-0.5, 1, 1], [0.5, 1, 1], [1, 1, 0]],
    ], dtype=np.float32)

    # Enable depth testing so hidden surfaces are correctly occluded
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)

    clock = pygame.time.Clock()

    # Interaction state
    dragging_orbit = False
    dragging_node = False
    last_mouse_pos = (0, 0)
    selected_idx = None
    selected_winz = None
    selected_offset = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    show_nodes = True

    # ---- Main loop ---

    while True:

        #Initializations
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = width / float(height) if height != 0 else 1.0
        gluPerspective(camera.fov, aspect, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera.apply()

        #mouse handler
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Left mouse: get control node
                if event.button == 1:
                        pick = pick_control_point(points, event.pos, width, height, threshold=12)
                        if pick is not None:
                            i, j, winz, sx, sy = pick
                            selected_idx = (i, j)
                            selected_winz = winz
                            
                            # Compute world position under the mouse at the node's depth
                            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                            projection = glGetDoublev(GL_PROJECTION_MATRIX)
                            viewport = glGetIntegerv(GL_VIEWPORT)
                            winX = float(event.pos[0])
                            winY = float(viewport[3] - event.pos[1])
                            world = gluUnProject(winX, winY, selected_winz, modelview, projection, viewport)
                            world = np.array(world, dtype=np.float32)
                            selected_offset = points[i, j].copy() - world
                            dragging_node = True
                        else:
                            selected_idx = None

                # Right mouse button starts orbit drag
                elif event.button == 3:
                    dragging_orbit = True
                    last_mouse_pos = event.pos
                # Older pygame mouse wheel events (button 4/5)
                elif event.button == 4:
                    camera.change_fov(1)
                elif event.button == 5:
                    camera.change_fov(-1)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_node = False
                elif event.button == 3:
                    dragging_orbit = False

            elif event.type == pygame.MOUSEMOTION:
                if dragging_node and selected_idx is not None:
                    # Move selected control point along the view ray at the stored depth
                    modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
                    projection = glGetDoublev(GL_PROJECTION_MATRIX)
                    viewport = glGetIntegerv(GL_VIEWPORT)
                    winX = float(event.pos[0])
                    winY = float(viewport[3] - event.pos[1])
                    world = gluUnProject(winX, winY, selected_winz, modelview, projection, viewport)
                    world = np.array(world, dtype=np.float32)
                    i, j = selected_idx
                    points[i, j] = (world + selected_offset).astype(np.float32)
                elif dragging_orbit:
                    x, y = event.pos
                    lx, ly = last_mouse_pos
                    dx = x - lx
                    dy = y - ly
                    # Invert Y so dragging up rotates the camera up
                    camera.update_angles(dx, -dy)
                    last_mouse_pos = (x, y)

            elif event.type == pygame.MOUSEWHEEL:
                # pygame 2 wheel event: event.y is 1 (up) or -1 (down)
                camera.change_fov(event.y)
                
            elif event.type == pygame.KEYDOWN:
                # Quick save with 's' key
                if event.key == pygame.K_s:
                    ok = save_points_and_obj(points, surface_resolution=100, cp_filename='./meshes/bezier_config.json', obj_filename='./meshes/bezier_surface.obj')
                    print('Save completed' if ok else 'Save failed')

                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                
                if event.key == pygame.K_n:
                    show_nodes = not show_nodes

        # Clear color and depth buffers before drawing
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Drawing
        surface_points = compute_bezier_surface(points, resolution=100)
        if show_nodes:
            draw_control_polygon(points, selected_idx)
        draw_surface(surface_points)

        # Updates
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()
