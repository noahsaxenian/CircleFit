import numpy as np
from stl import mesh


def subdivide_triangle(triangle):
    v1 = np.array(triangle[0])
    v2 = np.array(triangle[1])
    v3 = np.array(triangle[2])

    v12 = (v1 + v2) / 2
    v23 = (v2 + v3) / 2
    v13 = (v3 + v1) / 2

    triangles = []
    tri1 = [v1, v12, v13]
    tri2 = [v2, v12, v23]
    tri3 = [v3, v23, v13]
    tri4 = [v12, v23, v13]

    return [tri1, tri2, tri3, tri4]


def subdivide_mesh(your_mesh):
    vertices = your_mesh.vectors

    shape = vertices.shape

    subdivided_vertices = np.zeros((shape[0]*4, 3, 3))

    for i, triangle in enumerate(vertices):
        new_triangles = subdivide_triangle(triangle)
        for j in range(4):
            subdivided_vertices[i*4+j] = new_triangles[j]

    modified_mesh = mesh.Mesh(np.zeros(subdivided_vertices.shape[0], dtype=mesh.Mesh.dtype))
    modified_mesh.vectors = subdivided_vertices

    return modified_mesh

file_path = 'C:/Users/noahs/Downloads/wood_plate.stl'
your_mesh = mesh.Mesh.from_file(file_path)

for i in range(4):
    your_mesh = subdivide_mesh(your_mesh)

modified_file_path = 'C:/Users/noahs/Downloads/wood_plate1.stl'
your_mesh.save(modified_file_path)
