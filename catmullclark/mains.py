from main import np, Surface
import tqdm as tqdm

COLORS = ['r', 'y', 'g', 'b']

MESH_PATH = 'meshes/ex_patches/3.obj'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def load_surface():
    """Load the test surface and (re)build its faces."""
    surf = Surface(MESH_PATH)
    surf.load_faces()
    return surf


def traverse(he, surf, idx, plot=True, twin=False):
    """
    Advance one half-edge. If `twin` is False, step to next['from'] and
    optionally plot it (cycling through COLORS). If `twin` is True, hop across
    the twin then advance, returning (he, idx) unchanged in color index.
    """
    if not twin:
        he = surf.he[he['next']]
        p = he['from']
        if plot:
            surf.add_point(p, c=COLORS[idx])
            idx = (idx + 1) % 4
        return he, idx

    he = surf.he[he['twin']]
    he = surf.he[he['next']]
    return he, idx


def render_patch_surfaces(surf, patches):
    """Overlay a translucent control mesh and B-spline surface for each patch."""
    for p in tqdm.tqdm(patches):
        surf.add_patch(p, o=0.05)
        surf.add_b_surface(p)


# ---------------------------------------------------------------------------
# main1 - Display info about the surface (highlights extraordinary faces)
# ---------------------------------------------------------------------------
def main1():
    surf = load_surface()

    print(f"{len(surf.extraordinary_vertices)} Extraordinary Vertices")
    for vertex in surf.extraordinary_vertices:
        print(f"Vertex: {vertex}", surf.vertices[vertex])

    print(f"{len(surf.extraordinary_faces)} Extraordinary Faces")
    for face in surf.extraordinary_faces:
        print(face)
        surf.add_face(face, color="orange")

    print(f"{len(surf.regular_faces)} Regular Faces")
    for face in surf.regular_faces:
        print(face)
        surf.add_face(face, color="blue")

    print(f"{len(surf.regular_patches)} Regular Patches")
    for patch in surf.regular_patches:
        print(patch)

    surf.show()


# ---------------------------------------------------------------------------
# main2 - Traverse across a regular patch (plot control nodes in order)
# ---------------------------------------------------------------------------
def main2():
    surf = load_surface()

    p1 = surf.regular_patches[3]
    f1 = p1.center 

    # surf.traverse_across_p(f1)

    # reordering / plot the 4x4 node grid, coloring each column
    surf.add_patch(p1, o=0.5)
    for row in p1.nodes:
        for i in range(4):
            surf.add_point(row[i], c=COLORS[i])

    surf.show()


# ---------------------------------------------------------------------------
# main3 - Limit surface of a single regular patch
# ---------------------------------------------------------------------------
def main3():
    surf = load_surface()

    p1 = surf.regular_patches[3]
    f1 = p1.center 

    surf.add_patch(p1, o=0.5)
    surf.add_b_surface(p1)

    surf.show()


# ---------------------------------------------------------------------------
# main4 - Limit surface of a modified patch
# ---------------------------------------------------------------------------
def main4():
    surf = load_surface()

    s = 1
    patch = surf.regular_patches[3]
    for face in patch.faces:
        for vertex in face.vertices:
            surf.vertices[vertex] *= s

    surf.load_faces()

    render_patch_surfaces(surf, surf.regular_patches[:12])
    surf.show()


# ---------------------------------------------------------------------------
# main5 - Limit surface of all regular patches
# ---------------------------------------------------------------------------
def main5():
    surf = load_surface()

    render_patch_surfaces(surf, surf.regular_patches)
    surf.show()


# ---------------------------------------------------------------------------
# main6 - Extraordinary patch detailed traversal
# ---------------------------------------------------------------------------
def main6():
        surf = Surface('meshes/closed_surfaces/sphere_g0/sphere_386.obj')
        surf.load_faces()

     
        p1 = surf.irregular_patches[2]
        f1 = p1.center
        f1.check_extraordinary(surf.extraordinary_vertices)

        p = f1.extraordinary_vertex

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
        print(f'Located Diagonal [{hef['from']}]\n')
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
                                (_, idx) = traverse(he,surf,idx)
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


        print(f'\n[JUMP][8] point {he['from']} -> {hef['from']}\n') 
        visited.add(hef['from'])
            

        steps += 7   

        #------------------------
        # Left Traversal
        #------------------------

        he = hef_t
        idx = 0
        surf.add_point(hef['from'],c = COLORS[idx])
        idx = 1

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he,idx) = traverse(he,surf,idx)

        print(f'[Taking Twin]')
        (he, idx) = traverse(he,surf,idx,False,True)

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he, idx) = traverse(he,surf,idx)

        print(f'[Taking Twin]')
        (he, idx) = traverse(he,surf,idx,False,True)

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he,idx) = traverse(he,surf,idx)

        #------------------------
        # Right Traversal
        #------------------------

        print(f'\n[JUMP] point {he['from']} -> {hef['from']}\n') 

        he = hef
        idx = 0
        surf.add_point(hef['from'],c = COLORS[idx])
        idx = 1

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he,idx) = traverse(he,surf,idx)

        print(f'[Taking Twin]')
        (he, idx) = traverse(he,surf,idx,False,True)

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he, idx) = traverse(he,surf,idx)

        print(f'[Taking Twin]')
        (he, idx) = traverse(he,surf,idx,False,True)

        visited.add(he['to'])
        print(f'[{len(visited)}/{steps}] point: {he['from']} -> {he['to']}')
        (he, idx) = traverse(he,surf,idx)
        


        surf.show()

# ---------------------------------------------------------------------------
# main6 - Extraordinary patch: reordered traversal
# ---------------------------------------------------------------------------
def main7():
    surf = load_surface()

    p1 = surf.irregular_patches[2]
    f1 = p1.center 

    nodes = p1.nodes

    print(f'{len(nodes)} Nodes')
    surf.add_patch(p1, o=0.5)
    for i in range(len(nodes)):
        #print(nodes[i])
        surf.add_point(nodes[i],COLORS[i%4])

    surf.show()

# ---------------------------------------------------------------------------
# main6 - Extraordinary patch: reordered traversal
# ---------------------------------------------------------------------------
def main8():
    surf = load_surface()

    surf.show()