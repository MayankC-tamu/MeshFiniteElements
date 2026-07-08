import numpy as np
import trimesh
from scipy.spatial import Delaunay
import os

def func(x, y):
    return (x**2-1)*(y**2-1)

def load_csv(filename):
  with open(filename, 'r') as f:
      f.readline()  # skip header
      lines = f.readlines()
      X = []
      Y = []
      Z = []
      U = []

      for line in lines:
         x, y, z, u = map(float, line.strip().split(',')[:4])
         X.append(x)
         Y.append(y)
         Z.append(z)
         U.append(np.abs(u-func(x, y)))
  return X, Y, Z, U

X, Y, Z, U = load_csv('./solutions/parab-quad/5.csv')


domain = np.vstack((X,Y)).T # pairs of X,Y
tri = Delaunay(domain)
faces = tri.simplices
vertices = np.vstack((X,Y,Z)).T # triplets of X,Y,Z

# create mesh without post-processing to keep vertex ordering
mesh  = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

U_max = np.max(U)
U_min = np.min(U)

# guard against constant field
if U_max == U_min:
   norm = np.zeros_like(U, dtype=float)
else:
   norm = (U - U_min) / (U_max - U_min)

# red = high, blue = low
Red = (norm * 255).astype(np.uint8)
Green = np.zeros_like(Red, dtype=np.uint8)
Blue = ((1.0 - norm) * 255).astype(np.uint8)
Alpha = 255 * np.ones_like(Red, dtype=np.uint8)

vertex_colors = np.vstack((Red, Green, Blue, Alpha)).T

mesh.visual.vertex_colors = vertex_colors

mesh.export(f"./meshes/parab-quad-5.obj")

print('Finished Exporting')
