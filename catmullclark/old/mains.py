from main import np, Surface
import tqdm as tqdm

COLORS = ['r','y','g','b']

#--------------------------------
# Display Information About
# - vertices, extraordinary vertices
# - faces, extraordinary faces,
# - regular patches
# Shows Surface faces (highlights extraordinary faces)
#--------------------------------
def main1():
        
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()


        print(f"{len(surf.extraordinary_vertices)} Extraordinary Vertices")

        for vertex in surf.extraordinary_vertices:
            print(f"Vertex: {vertex}", surf.vertices[vertex])

        print(f"{len(surf.extraordinary_faces)} Extraordinary Faces")

        for face in surf.extraordinary_faces:
            print(face)
            surf.add_face(face,color="orange")

        print(f"{len(surf.regular_faces)} Regular Faces")

        for face in surf.regular_faces:
            print(face)
            surf.add_face(face,color="blue")


        print(f"{len(surf.regular_patches)} Regular Patches")
        for patch in surf.regular_patches:
              print(patch)

        surf.show()
    
#------------------------------
# Traverses Across Regular Patch
#------------------------------
def main2():
        
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()
     
        p1 = surf.regular_patches[3]
        f1 = p1.center

        #traversal path
        # surf.add_patch(p1,o=0.5)
        # surf.traverse_across_p(f1)

        #reordering
        colors = ['r','y','g','b',]
        surf.add_patch(p1,o=0.5)
        for row in p1.nodes:
               for i in range(4):
                       surf.add_point(row[i],c=colors[i]) 

        surf.show()
        


#------------------------------
# Limit Surface of Regular Patch
#------------------------------
def main3():
        
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()
     
        p1 = surf.regular_patches[3]
        f1 = p1.center

        surf.add_patch(p1,o=0.5)
        surf.add_b_surface(p1)
    
        surf.show()

#------------------------------
# Limit Surface of Modified Patch
#------------------------------

def main4():
        
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()

        s = 1

        patch = surf.regular_patches[3]
        for face in patch.faces:
              for vertex in face.vertices:
                    surf.vertices[vertex] *= s

        surf.load_faces()

        patches = surf.regular_patches[:12]
        for p in tqdm.tqdm(patches):
                      surf.add_patch(p,o=0.05)
                      surf.add_b_surface(p)
        

        surf.show()

#------------------------------
# Limit Surface of All Regular Patch
#------------------------------

def main5():
        
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()


        patches = surf.regular_patches
        for p in tqdm.tqdm(patches):
                      surf.add_patch(p,o=0.05)
                      surf.add_b_surface(p)
        

        surf.show()


def traverse(he,surf,idx,plot = True, twin = False):
        if twin == False:
                he = surf.he[he['next']]
                p = he['from']
                if plot:
                        surf.add_point(p,c=COLORS[idx])
                        idx = (idx+1)%4
                return (he,idx)
        else:
                he = surf.he[he['twin']]
                he = surf.he[he['next']]
                return (he,idx)



def main6():
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()

     
        p1 = surf.irregular_patches[3]
        f1 = p1.center
        f1.check_extraordinary(surf.extraordinary_vertices)

        p = f1.extroardinary_vertex

        #traversal path
        surf.add_patch(p1,o=0.5)

        he = surf.get_half_edges(p)[0]
        he0 = he

        p = he['from']
        surf.add_point(p,c='r')
        idx = 1

        #Locate Diagonal
        #------------------------
        he,idx = traverse(he,surf,idx,False)
        he,idx = traverse(he,surf,idx,False)
        he,idx = traverse(he,surf,idx,False,True)
        he,idx = traverse(he,surf,idx,False,True)
        he,idx = traverse(he,surf,idx,False)
        hef_t = surf.he[he['twin']]
        he,idx = traverse(he,surf,idx,False)
        hef = he
        print(f'Located Diagonal [{hef['from']}]')
        #------------------------

        he = he0

        print(f'Starting Ring Traversal')
        #------------------------
        visited = set()
        visited.add(he['from'])

        steps = 7
        
        while len(visited) < steps:
                point1 = he['from']
                point2 = he['to']

                twin = surf.he[he['twin']]

                t_point1 = twin['from']
                t_point2 = twin['to']

                if point2 not in visited:
                        visited.add(point1)
                        if (len(visited) < steps):
                                (_, idx) = traverse(he,surf,idx,False)
                        print(f'[{len(visited)}/{steps}] point: {point1} -> {point2}')

                else:
                        he = twin
                        point1 = he['from']
                        point2 = he['to']
                        twin = surf.he[he['twin']]

                        t_point1 = twin['from']
                        t_point2 = twin['to']
                        print(f'[{len(visited)}/{steps}] point: {point2} -> {point1}')
                        print(f'[TAKING TWIN] point: {t_point2} -> {t_point1}')

                # advance forward in the current face
                he = surf.he[he['next']]


        print(f'[JUMP] point {he['from']} -> {hef['from']}')        

        #------------------------
        # Left Traversal
        #------------------------

        he = hef_t
        idx = 0
        surf.add_point(hef['from'],c = COLORS[idx])
        idx = 1
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he,idx) = traverse(he,surf,idx)
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he, idx) = traverse(he,surf,idx,False,True)
        (he, idx) = traverse(he,surf,idx)
        (he, idx) = traverse(he,surf,idx,False,True)
        (he,idx) = traverse(he,surf,idx)

        #------------------------
        # Right Traversal
        #------------------------

        # he = hef
        # idx = 0
        # surf.add_point(hef['from'],c = COLORS[idx])
        # idx = 1
        # (he,idx) = traverse(he,surf,idx)
        # (he, idx) = traverse(he,surf,idx,False,True)
        # (he, idx) = traverse(he,surf,idx)
        # (he, idx) = traverse(he,surf,idx,False,True)
        # (he, idx) = traverse(he,surf,idx)


        surf.show()
