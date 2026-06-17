import numpy as np
import trimesh
from scipy.spatial import Delaunay

class Surface:
    def __init__(self, name, U, V, X, Y, Z):
      
      self.name = name
      
      domain = np.vstack((U.flatten(),V.flatten())).T #pairs of U,V
      tri = Delaunay(domain)

      self.faces = tri.simplices
      self.vertices = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T #triplets of X,Y,Z
      self.mesh  = trimesh.Trimesh(vertices=self.vertices,faces=self.faces)
      
      self.mesh.visual.face_colors = [100, 150, 255, 255]

    def export(self):
       print(f"Exporting....")
       print(f"Mesh is closed: {self.mesh.is_watertight}")
       print(f"Generated {len(self.mesh.faces)} triangles")
       self.mesh.export(f"./meshes/{self.name}.obj")
       print(f"Finished exporting to {self.name}.obj")
    
      

class saddle(Surface):
    def __init__(self,precision=100):
       u = np.linspace(-2,2,precision)
       v = np.linspace(-2,2,precision)

       U,V = np.meshgrid(u,v)

       X = U
       Y = V
       Z = U**2/2 - V**2/9

       super().__init__("saddle",U,V,X,Y,Z)


class ripple(Surface):
    def __init__(self,precision=100):
       u = np.linspace(-np.pi,np.pi,precision)
       v = np.linspace(-np.pi,np.pi,precision)

       U,V = np.meshgrid(u,v)

       X = U
       Y = V
       Z = np.sin(np.sqrt(9*U**2 +9*V**2))

       super().__init__("ripple",U,V,X,Y,Z)

class mobius(Surface):
    def __init__(self,precision=100):
       u = np.linspace(-np.pi,np.pi,precision)
       v = np.linspace(-1,1,precision)

       U,V = np.meshgrid(u,v)

       X = (1 + V/2 * np.cos(U/2)) * np.cos(U)
       Y = (1 + V/2 * np.cos(U/2)) * np.sin(U)
       Z = V/2 * np.sin(U/2)

       super().__init__("mobius",U,V,X,Y,Z)


def main():
   surface = mobius(precision=100)
   surface.export()

if __name__ == "__main__":
   main()