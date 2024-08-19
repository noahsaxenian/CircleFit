import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
from collections import defaultdict, deque
import heapq
import pyvista as pv
import time
from matplotlib.colors import ListedColormap
import networkx as nx
from scipy.interpolate import RBFInterpolator, griddata
from sklearn.manifold import MDS



def select_points(your_mesh):
    # Extract vertices
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Determine the range for each axis
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    max_range = max(x_range, y_range)
    x_center = (vertices[:, 0].max() + vertices[:, 0].min()) / 2
    y_center = (vertices[:, 1].max() + vertices[:, 1].min()) / 2

    # Create a 2D plot (top-down view) for point selection
    fig, ax = plt.subplots()
    sc = ax.scatter(vertices[:, 0], vertices[:, 1], c='blue', s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Click to select points')

    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    selected_points = []

    def on_click(event):
        if event.inaxes == ax:
            # Find the closest vertex to the click
            x_click = event.xdata
            y_click = event.ydata
            distances = np.sqrt((vertices[:, 0] - x_click)**2 + (vertices[:, 1] - y_click)**2)
            min_idx = np.argmin(distances)
            selected_points.append(vertices[min_idx])
            # Highlight the selected point
            ax.scatter(vertices[min_idx, 0], vertices[min_idx, 1], c='red', s=50, edgecolors='black')
            plt.draw()
            print("Selected point:", vertices[min_idx])

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    return selected_points


def get_internal_boundaries(your_mesh, xmin, xmax, ymin, ymax, plot=False):
    # Create a dictionary to keep track of edges
    edges_dict = {}
    vertices_on_boundary_edges = set()
    # Iterate through faces and extract edges
    for face in your_mesh.vectors:
        for i in range(3):
            edge = tuple(sorted([tuple(face[i]), tuple(face[(i + 1) % 3])]))
            if edge not in edges_dict:
                edges_dict[edge] = 0
            edges_dict[edge] += 1
    # Collect vertices on boundary edges
    for edge, count in edges_dict.items():
        if count == 1:  # Boundary edge
            vertices_on_boundary_edges.update(edge)
    boundary_vertices = np.array(list(vertices_on_boundary_edges))

    internal_boundary_vertices = []
    for vertex in boundary_vertices:
        if xmin < vertex[0] and vertex[0] < xmax:
            if ymin < vertex[1] and vertex[1] < ymax:
                internal_boundary_vertices.append(vertex)
    internal_boundary_vertices = np.array(internal_boundary_vertices)

    if plot:
        plotter = pv.Plotter()

        # Add the mesh to the plotter
        mesh_pv = pv.PolyData(your_mesh.vectors.reshape(-1, 3))
        plotter.add_mesh(mesh_pv, opacity=0.5)

        # Add boundary vertices to the plot
        plotter.add_points(boundary_vertices, color='red', point_size=10, render_points_as_spheres=True)

        # Add internal boundary vertices to the plot
        plotter.add_points(internal_boundary_vertices, color='blue', point_size=10, render_points_as_spheres=True)

        # Display the plot
        plotter.show()

    return internal_boundary_vertices


def multi_source_dijkstra(graph, sources):
    # Initialize distances dictionary
    distances = {node: [float('inf')] * len(sources) for node in graph}

    max_dist = 0

    # Initialize priority queue for Dijkstra's algorithm
    pq = []

    # Add all source nodes to the priority queue and set their distances to 0
    for i, source in enumerate(sources):
        distances[source][i] = 0
        heapq.heappush(pq, (0, source, i))

    # Calculate total number of nodes for progress tracking
    total_nodes = len(graph) * len(sources)
    count = 0
    print('calculating paths', end='')

    # Perform Dijkstra's algorithm
    while pq:
        current_distance, current_node, source_index = heapq.heappop(pq)

        # If we've found a longer path, skip
        if current_distance > distances[current_node][source_index]:
            continue

        for neighbor in graph[current_node]:
            edge_length = np.linalg.norm(np.array(current_node) - np.array(neighbor))
            distance = current_distance + edge_length

            if distance < distances[neighbor][source_index]:
                distances[neighbor][source_index] = distance
                if distance > max_dist:
                    max_dist = distance
                heapq.heappush(pq, (distance, neighbor, source_index))

        # Update progress
        count += 1
        if count % (total_nodes // 20) == 0:
            print('.', end='')
    print("done")

    return distances, max_dist

def landmark_dijkstra(graph, landmark_vertices):
    n = len(landmark_vertices)
    distance_matrix = np.full((n, n), float('inf'))

    # Create a mapping from landmark vertices to their indices
    landmark_indices = {vertex: i for i, vertex in enumerate(landmark_vertices)}

    for i, source in enumerate(landmark_vertices):
        # Initialize distances for this source
        distances = {node: float('inf') for node in graph}
        distances[source] = 0
        pq = [(0, source)]
        landmarks_found = 0

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            # If we've found a longer path, skip
            if current_distance > distances[current_node]:
                continue

            # If current_node is a landmark, update the distance matrix
            if current_node in landmark_indices:
                j = landmark_indices[current_node]
                distance_matrix[i, j] = current_distance
                landmarks_found += 1

            # If we've found all landmarks, we can stop
            if landmarks_found == n:
                break

            for neighbor in graph[current_node]:
                edge_length = np.linalg.norm(np.array(current_node) - np.array(neighbor))
                distance = current_distance + edge_length

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))

        print(f'\rProcessed landmark {i + 1}/{n}', end='   ')
    print('')

    return distance_matrix


def create_voronoi_diagram(your_mesh, locations):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Create a graph representation of the mesh
    graph = defaultdict(list)
    for triangle in your_mesh.vectors:
        for i in range(3):
            v1 = tuple(triangle[i])
            for j in range(i + 1, 3):
                v2 = tuple(triangle[j])
                graph[v1].append(v2)
                graph[v2].append(v1)

    for node in graph:
        coords = np.array(graph[node])
        unique = np.unique(coords, axis=0)
        unique_list = [tuple(xyz) for xyz in unique.tolist()]
        graph[node] = unique_list


    # Convert vertices and locations to tuples
    vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in locations]

    # Run multi-source Dijkstra from all sources
    distances, _ = multi_source_dijkstra(graph, sources)

    # Determine Voronoi regions
    voronoi_regions = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        distance_array = np.array(distances[tuple(vertex)])
        closest_source = np.argmin(distance_array)
        voronoi_regions[i] = closest_source

    return np.array(voronoi_regions)


def plot_voronoi_diagram(your_mesh, locations, voronoi_regions):
    vertices = your_mesh.vectors.reshape(-1, 3)
    faces = np.column_stack((np.full(len(your_mesh.vectors), 3), np.arange(len(vertices)).reshape(-1, 3)))
    #faces = np.array([3,0,0,0])
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.point_data["voronoi_regions"] = voronoi_regions

    num_regions = len(np.unique(voronoi_regions))
    color_map = 'rainbow'
    plotter = pv.Plotter()
    plotter.add_mesh(pv_mesh, scalars="voronoi_regions", cmap=color_map, show_scalar_bar=False, interpolate_before_map=False)

    # Add points for the source location
    points = pv.PolyData(locations)
    plotter.add_mesh(points, color="black", point_size=10)

    # Add text labels for each region
    for i, location in enumerate(locations):
        plotter.add_point_labels(location, [f"Region {i}"], point_size=1, font_size=10)

    plotter.show()



def interpolate_mode_shapes(your_mesh, locations, mode_shapes):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Create a graph representation of the mesh
    graph = defaultdict(list)
    for triangle in your_mesh.vectors:
        for i in range(3):
            v1 = tuple(triangle[i])
            for j in range(i + 1, 3):
                v2 = tuple(triangle[j])
                graph[v1].append(v2)
                graph[v2].append(v1)

    # Distance Calculation
    # Convert vertices and sources to tuples
    vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in locations]

    # Run multi-source BFS from all sources
    distances, max_dist = multi_source_dijkstra(graph, sources)

    # Mode Shape Calculation
    shapes = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        distance_array = np.array(distances.get(vertex, float('inf')))
        shapes[i] = np.sum(mode_shapes * np.exp(-(0.04 * distance_array)**2))
    shapes = shapes / np.max(np.abs(shapes))

    return shapes


def interpolate_mode_shapes_scipy(your_mesh, locations, mode_shapes, smooth=0):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Check if the mesh is 2D or 3D by examining the z-coordinates
    if np.all(vertices[:, 2] == vertices[0, 2]):
        # 2D mesh
        vertices_2d = vertices[:, :2]
        locations_2d = np.array([loc[:2] for loc in locations])
        interpolator = RBFInterpolator(locations_2d, mode_shapes, kernel='thin_plate_spline', smoothing=smooth)
        shapes = interpolator(vertices_2d)
    else:
        # 3D mesh
        locations_3d = np.array([loc for loc in locations])
        interpolator = RBFInterpolator(locations_3d, mode_shapes, kernel='thin_plate_spline', smoothing=smooth)
        shapes = interpolator(vertices)

    return shapes


def interpolate_mode_shapes_voronoi(your_mesh, locations, mode_shapes):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Create a graph representation of the mesh
    graph = defaultdict(list)
    for triangle in your_mesh.vectors:
        for i in range(3):
            v1 = tuple(triangle[i])
            for j in range(i + 1, 3):
                v2 = tuple(triangle[j])
                graph[v1].append(v2)
                graph[v2].append(v1)

    sources = []
    for loc in locations:
        dist = np.linalg.norm(vertices - loc, axis=1)
        closest_vertex_index = np.argmin(dist)
        sources.append(vertices[closest_vertex_index])

    # Distance Calculation
    # Convert vertices and sources to tuples
    vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in sources]

    # Run multi-source BFS from all sources
    distances, max_dist = multi_source_dijkstra(graph, sources)

    voronoi_regions = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        distance_array = np.array(distances[tuple(vertex)])
        closest_source = np.argmin(distance_array)
        voronoi_regions[i] = closest_source

    # Mode Shape Calculation
    shapes = np.zeros(len(vertices))
    for i in range(len(vertices)):
        index = int(voronoi_regions[i])
        shapes[i] = mode_shapes[index]

    return shapes


def interpolate_voronoi_smooth(your_mesh, locations, mode_shapes):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Create a graph representation of the mesh
    graph = defaultdict(list)
    for triangle in your_mesh.vectors:
        for i in range(3):
            v1 = tuple(triangle[i])
            for j in range(i + 1, 3):
                v2 = tuple(triangle[j])
                graph[v1].append(v2)
                graph[v2].append(v1)

    sources = []
    for loc in locations:
        dist = np.linalg.norm(vertices - loc, axis=1)
        closest_vertex_index = np.argmin(dist)
        sources.append(vertices[closest_vertex_index])

    # Distance Calculation
    # Convert vertices and sources to tuples
    vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in sources]

    # Run multi-source BFS from all sources
    distances, max_dist = multi_source_dijkstra(graph, sources)

    voronoi_regions = np.zeros(len(vertices))
    for i, vertex in enumerate(vertices):
        distance_array = np.array(distances[tuple(vertex)])
        closest_source = np.argmin(distance_array)
        voronoi_regions[i] = closest_source

    # Mode Shape Calculation
    shapes = np.zeros(len(vertices))
    for i in range(len(vertices)):
        index = int(voronoi_regions[i])
        shapes[i] = mode_shapes[index]

    unique_vertices, indices = np.unique(vertices, axis=0, return_index=True)
    unique_shapes = shapes[indices]

    shapes = interpolate_mode_shapes_scipy(your_mesh, unique_vertices, unique_shapes, smooth=5000)

    return shapes


def interpolate_boundaries(your_mesh, locations, mode_shapes):
    vertices = your_mesh.vectors.reshape(-1, 3)

    # Create a graph representation of the mesh
    graph = defaultdict(list)
    for triangle in your_mesh.vectors:
        for i in range(3):
            v1 = tuple(triangle[i])
            for j in range(i + 1, 3):
                v2 = tuple(triangle[j])
                graph[v1].append(v2)
                graph[v2].append(v1)

    sources = []
    for loc in locations:
        dist = np.linalg.norm(vertices - loc, axis=1)
        closest_vertex_index = np.argmin(dist)
        sources.append(vertices[closest_vertex_index])

    # Distance Calculation
    # Convert vertices and sources to tuples
    # vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in sources]

    # Run multi-source BFS from all sources
    distances, max_dist = multi_source_dijkstra(graph, sources)

    boundaries = get_internal_boundaries(your_mesh, 0, 70, -70, 70)
    #boundaries = [boundaries[i] for i in range(0, len(boundaries), 5)]

    landmark_shapes = [shape for shape in mode_shapes]
    landmark_vertices = [source for source in sources]

    for i, vertex in enumerate(boundaries):
        distance_array = np.array(distances[tuple(vertex)])
        # Get the indices of the 3 closest sources
        closest_sources = np.argsort(distance_array)[:3]
        closest_distances = distance_array[closest_sources]
        closest_shapes = mode_shapes[closest_sources]

        # Calculate weights as the inverse of the distances
        weights = 1 / closest_distances
        # Normalize the weights so they sum to 1
        weights /= weights.sum()

        # Compute the weighted average of the shapes
        weighted_shape = np.average(closest_shapes, axis=0, weights=weights)

        landmark_shapes.append(weighted_shape)
        landmark_vertices.append(vertex)

    shapes = interpolate_mode_shapes_scipy(your_mesh, landmark_vertices, landmark_shapes, smooth=100)

    return shapes


def interpolate_multidimensional(your_mesh, landmark_vertices, distance_matrix, measurement_locations, measured_shapes):
    vertices = your_mesh.vectors.reshape(-1, 3)

    closest_vertices = []
    for loc in measurement_locations:
        dist = np.linalg.norm(vertices - loc, axis=1)
        closest_vertex_index = np.argmin(dist)
        closest_vertices.append(vertices[closest_vertex_index])

    # Distance Calculation
    # Convert vertices and sources to tuples
    vertices = [tuple(v) for v in vertices]
    sources = [tuple(loc) for loc in closest_vertices]

    # Apply MDS
    print('Applying multidimensional scaling...')
    mds = MDS(n_components=8, dissimilarity='precomputed', max_iter=10000, n_init=3, eps=1e-12)

    transformed_landmarks = mds.fit_transform(distance_matrix)

    source_indices = [landmark_vertices.index(source) for source in sources]
    transformed_locations = []
    for index in source_indices:
        transformed_locations.append(transformed_landmarks[index])

    # check mds quality
    n = len(landmark_vertices)
    differences = []
    for i in range(n):
        for j in range(i+1, n):
            original_distance = distance_matrix[i, j]

            t1 = np.array(transformed_landmarks[i])
            t2 = np.array(transformed_landmarks[j])

            new_distance = np.linalg.norm(t1 - t2)

            difference = np.abs(original_distance - new_distance)
            differences.append(difference)
    differences = np.sort(differences)[::-1]
    print('differences between distances along mesh and in multidimensional space:')
    print(differences)

    print('Interpolating...')
    interpolator = RBFInterpolator(transformed_locations, measured_shapes, kernel='thin_plate_spline')

    landmark_shapes = interpolator(transformed_landmarks)
    shapes = interpolate_mode_shapes_scipy(your_mesh, landmark_vertices, landmark_shapes)

    return shapes


def animate_3D(your_mesh, mode_shapes, measurement_locations, measured_shapes):
    max_shape = np.max(np.abs(measured_shapes))
    measured_shapes = measured_shapes/max_shape
    mode_shapes = mode_shapes/max_shape

    vertices = your_mesh.vectors.reshape(-1, 3)
    faces = np.column_stack((np.full(len(your_mesh.vectors), 3), np.arange(len(vertices)).reshape(-1, 3))).ravel()


    print('plotting')
    pv_mesh = pv.PolyData(vertices, faces)
    pv_mesh.point_data["mode_shape"] = mode_shapes

    plotter = pv.Plotter()
    color_map = 'rainbow'
    mesh_actor = plotter.add_mesh(pv_mesh, scalars="mode_shape", cmap=color_map, show_scalar_bar=False, interpolate_before_map=False)

    points = pv.PolyData(measurement_locations)
    points_actor = plotter.add_mesh(points, color="black", point_size=10)

    # Add a close button
    stop_animation = False
    def close_animation(state):
        nonlocal stop_animation
        stop_animation = state

    plotter.add_checkbox_button_widget(
        callback=close_animation,
        value=False,
        position=(10.0, 10.0),
        size=50,
        border_size=5,
        color_off='red',
        background_color='white',
    )

    # Start the plotter's interactive window
    plotter.show(interactive_update=True, auto_close=False)
    t = 0
    while not stop_animation:
        # Calculate the displacement factor
        displacement = np.sin(2 * np.pi * t) * 10

        updated_vertices = vertices.copy()
        updated_vertices[:, 2] += displacement * mode_shapes
        pv_mesh.points = updated_vertices

        updated_locations = np.array(measurement_locations.copy())
        updated_locations[:, 2] += displacement * measured_shapes
        points.points = updated_locations

        mesh_actor.GetMapper().Update()
        points_actor.GetMapper().Update()

        plotter.update()

        t += 0.005  # Adjust this to control the speed of oscillation

    plotter.clear()
    plotter.close()