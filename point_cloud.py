import numpy as np
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial import KDTree, ConvexHull, Delaunay

from functions import *

def uniform_downsample(pcd_raw, point_cloud_density):

    def compute_grid_size(min_bound, max_bound, point_cloud_density):
        # compute the diagonal length of the bounding box
        diagonal_length = np.linalg.norm(max_bound - min_bound)
        
        # compute the number of grids in each dimension based on the desired point cloud density
        grid_size = diagonal_length / np.sqrt(point_cloud_density)
        
        return grid_size
    

    '''downsample point cloud, similar to voxel downsampling'''
    threshold_upper = 804 #depth roi
    threshold_lower = 700 #depth roi
    point_cloud = np.asarray(pcd_raw.points)[~np.isnan(np.asarray(pcd_raw.points)).any(axis = 1)]
    min_bound = np.min(point_cloud, axis=0)
    max_bound = np.max(point_cloud, axis=0)

    grid_size = compute_grid_size(min_bound, max_bound, point_cloud_density)

    # compute grid dimensions
    grid_dims = np.ceil((max_bound - min_bound) / grid_size).astype(int)
    
    #initialize grid
    grid = [[] for _ in range(np.prod(grid_dims))]
    
    # assign points to grid cells
    
    # for pc wrt robot base (eye to hand calib)
    for i, point in enumerate(point_cloud):
        cell_indices = np.floor((point - min_bound) / grid_size).astype(int)
        if np.any(cell_indices < 0) or np.any(cell_indices > grid_dims):
            continue
        if not (np.isnan(point[0]) and np.isnan(point[1]) and np.isnan(point[2])):
            cell_index = np.ravel_multi_index(cell_indices, grid_dims)
            grid[cell_index].append(i)
    
    ''' 
    # for pc wrt camera
    for i, point in enumerate(point_cloud):
        cell_indices = np.floor((point - min_bound) / grid_size).astype(int)
        if np.any(cell_indices < 0) or np.any(cell_indices > grid_dims) or point[2] > threshold_upper or point[2] < threshold_lower:
            continue
        if not (np.isnan(point[0]) and np.isnan(point[1]) and np.isnan(point[2])):
            cell_index = np.ravel_multi_index(cell_indices, grid_dims)
            grid[cell_index].append(i)
    '''
    
    # select points from each cell
    downsampled_indices = []
    for cell_points in grid:
        if cell_points:
            downsampled_indices.append(np.random.choice(cell_points))
    
    # generate downsampled point cloud
    downsampled_point_cloud = []
    for index in downsampled_indices:
        downsampled_point_cloud.append(point_cloud[index])
    

    return downsampled_point_cloud #returns type(list)


def estimate_normal(point_cloud): #pcd (list) to pcd (o3d)
    '''pcd.normals may not be calculated for raw pcd, this function returns pcd with xyz and normals'''
    # convert list arr output to o3d point cloud format
    if isinstance(point_cloud, list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    else:
        pcd = point_cloud
    
    # center = pcd.get_center()

    # estimate normal
    pcd.estimate_normals(search_param = o3d.geometry.KDTreeSearchParamHybrid(radius = 10, max_nn = 30), fast_normal_computation = True)
    # pcd.orient_normals_consistent_tangent_plane(5) #orient wrt tangent plane
    pcd.orient_normals_to_align_with_direction(orientation_reference = np.array([0, 0, -1]))
    # pcd.orient_normals_towards_camera_location(camera_location = np.array([0.35430414997587173, 0.1957132814893931, 1.0872309432039695]))
    # o3d.visualization.draw_geometries([pcd], point_show_normal = True)
    # print(1 if pcd.has_normals() else 0)

    return pcd

def calculate_normal(query_point, pcd):
    '''returns normal of centroid given the known position xyz of the centroid of the point cloud'''
    pc_xyz = np.asarray(pcd.points)
    pc_normals = np.asarray(pcd.normals)
    kdtree = KDTree(pc_xyz)

    # query nearest point in point cloud
    _, nearest_point_index = kdtree.query(query_point)
    normal = pc_normals[nearest_point_index]

    return normal

def pose_aligned_bounding_box(pcd, pos, normal, show = True):
    '''computes the OBB derived from AABB of point cloud where the faces of the bounding box is orthogonal to the input 3D vector'''
    import copy

    # normalize
    normal = normal / np.linalg.norm(normal)

    # rotation matrix to align the input vector with one of the principal axes (z-axis)
    z_axis = np.array([0, 0, -1])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)
    
    # skew-symmetric cross-product matrix
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    # compute rotation matrix
    if s == 0:
        R = np.eye(3)  # if vectors are parallel, no rotation is needed
    else:
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    
    # rotate the point cloud 
    pcd_rotated = copy.deepcopy(pcd)
    pcd_rotated.rotate(R, center = pos)
    
    # obtain AABB and convert to OBB
    aabb = pcd_rotated.get_axis_aligned_bounding_box()
    obb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(aabb)
    
    # rotate the oBB back to original orientation
    obb = obb.rotate(R.T, center = pos)

    #visualization
    if show:    
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()

        pcd_rotated.paint_uniform_color(color = [1, 0, 0])
        pcd.paint_uniform_color(color = [0, 1, 0])
        aabb.color = [1, 0, 0]
        obb.color = [0, 1, 0]

        visualizer.add_geometry(pcd)
        visualizer.add_geometry(obb)
        visualizer.add_geometry(pcd_rotated)
        visualizer.add_geometry(aabb)
        view_ctl = visualizer.get_view_control()
        view_ctl.set_up((1, 0, 0))  # set the positive direction of the x-axis as the up direction
        view_ctl.set_up((0, -1, 0))  # set the negative direction of the y-axis as the up direction
        view_ctl.set_front((1, 0, 0))  # set the positive direction of the x-axis toward you
        view_ctl.set_lookat((0, 0, 0))  # set the original point as the center point of the window
        visualizer.run()

    return obb, R


# def align_bounding_box_with_point_cloud(ax, ordered_pos_lst, point_cloud, show = True):
#     '''oriented bounding box generated from function bounding_box_3d() is a planar obb (2D in quality) lying in 3D space, 
#     this function superimposes the obb onto the actual pcd to obtain the actual z-value, accounts for curvatures'''
#     kdtree = o3d.geometry.KDTreeFlann(point_cloud)  # consider only x and y coordinates for KD-tree

#     aligned_points = []
#     for point_pos in ordered_pos_lst:
#         # find nearest neighbor in KD-tree
#         [_, nearest_index, _] = kdtree.search_knn_vector_3d(point_pos[:3], 1)
#         nearest_point = np.asarray(point_cloud.points)[nearest_index]

#         # Update z coordinate of the current point position
#         point = np.array([point_pos[0], point_pos[1], nearest_point[0, 2]])
        
#         if show: 
#             # ax.scatter(*nearest_point[0], color = "black")
#             ax.scatter(*point, color = "cyan")

#         aligned_points.append(point) #if nearest_point[0, 0] - point_pos[0] < threshold or nearest_point[0, 1] - point_pos[1] < threshold else None 

#     return aligned_points

def align_bounding_box_with_point_cloud(ax, ordered_pos_lst, point_cloud, show=True):
    """
    Aligns a 2D bounding box onto a 3D point cloud by adjusting the z-values of the bounding box points
    based on the nearest neighbor z-values from the point cloud.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object for plotting.
    ordered_pos_lst (list): List of 2D points representing the bounding box.
    point_cloud (open3d.geometry.PointCloud): The 3D point cloud data.
    show (bool): Whether to plot the aligned points.

    Returns:
    aligned_points (list): List of points with updated z-values.
    """
    
    # Extract xy coordinates from the point cloud and zero the z-coordinates
    xy_coords = np.asarray(point_cloud.points)[:, :2]
    zero_z = np.zeros((xy_coords.shape[0], 1))
    xy_coords_3d = np.hstack((xy_coords, zero_z))
    
    # Create a new PointCloud object for KD-Tree construction
    xy_point_cloud = o3d.geometry.PointCloud()
    xy_point_cloud.points = o3d.utility.Vector3dVector(xy_coords_3d)
    kdtree = o3d.geometry.KDTreeFlann(xy_point_cloud)
    
    aligned_points = []

    for point_pos in ordered_pos_lst:
        # Find nearest neighbor based on xy coordinates
        query_point = np.array([point_pos[0], point_pos[1], 0])
        [_, nearest_index, _] = kdtree.search_knn_vector_3d(query_point, 1)
        nearest_point = np.asarray(point_cloud.points)[nearest_index[0]]

        # Update z coordinate of the current point position
        aligned_point = np.array([point_pos[0], point_pos[1], nearest_point[2]])

        if show: 
            ax.scatter(*aligned_point, color="cyan")

        aligned_points.append(aligned_point)

    return aligned_points


#----------------------- alpha shape -----------------------------#
def alpha_shape(pcd, alpha = 0.05):
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    o3d.visualization.draw_geometries([pcd, mesh])

#----------------------- concave hull -----------------------------#
'''
1. compute Delaunay triangulation of pcd
2. filter triangles -> for each triangle: calculate circumradius -> if circuradius < threshold (1/alpha): keep the triangle; else: discard the triangle
3. from remaining triangles, extract boundary edges to form concave hull
'''
import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay
from collections import defaultdict

def alpha_shape(points, alpha):
    tri = Delaunay(points)
    triangles = []
    
    def add_triangle(ia, ib, ic):
        # add a triangle to the list if the triangle is within the alpha threshold
        pa, pb, pc = points[ia], points[ib], points[ic]
        a = np.linalg.norm(pa - pb)
        b = np.linalg.norm(pb - pc)
        c = np.linalg.norm(pc - pa)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        
        if circum_r < 1.0 / alpha:
            triangles.append([ia, ib, ic])
    
    # process each triangle
    for simplex in tri.simplices:
        ia, ib, ic = simplex
        add_triangle(ia, ib, ic)
    
    # extract unique vertices
    vertices = np.array(list(set([tuple(points[i]) for t in triangles for i in t])))
    
    # map old vertex indices to new indices
    vertex_map = {tuple(point): idx for idx, point in enumerate(vertices)}
    triangles = np.array([[vertex_map[tuple(points[ia])],
                           vertex_map[tuple(points[ib])],
                           vertex_map[tuple(points[ic])]] for ia, ib, ic in triangles])
    
    return vertices, triangles

def extract_outer_edges(triangles): #extract outer edges of concave hull
    edge_count = defaultdict(int)
    
    for triangle in triangles:
        for i in range(3):
            ia = triangle[i]
            ib = triangle[(i + 1) % 3]
            edge = tuple(sorted((ia, ib)))
            edge_count[edge] += 1
    
    # Only keep edges that appear exactly once
    outer_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    return outer_edges

#----------------------- plot pcd -----------------------------#
def plot_pcd(ax, path, cloud_density):
    pcd = o3d.io.read_point_cloud(path)
    pcd_downsampled = uniform_downsample(pcd, cloud_density)
    
    out_arr = np.asarray(pcd_downsampled)
    # print("output array:", out_arr)
    x, y, z = out_arr[:,0], out_arr[:,1], out_arr[:,2]

    pcd_with_normals = estimate_normal(pcd_downsampled)

    "-----------triangular mesh------------"
    # ax.plot_trisurf(x, y, z, color='white', edgecolors='lightgrey', alpha=0.25)
    # surf = ax.plot_trisurf(x, y, z, cmap = "viridis", edgecolor = "none", antialiased = True)
    # cbar = fig.colorbar(surf, ax = ax, label = "depth", orientation = "vertical")
    
    "-------------depth map--------------"
    # normalized_depth = (z - np.min(z)) / (np.max(z) - np.min(z))
    # sc = ax.scatter(x, y, z, c = normalized_depth, cmap = "viridis", alpha = 0.8, s = 4)
    # cbar = plt.colorbar(sc, ax=ax, label = 'Depth')
     
    "-------------point cloud--------------"
    ax.scatter(x, y, z, c = 'grey', alpha = 1)

    "------------.XYZ file gen-------------"
    # script_dir = r"C:\Users\leong\Documents\Chie Weng\Mech Mind\pratt"
    # open("point_cloud.xyz", "w").close()
    # with open(os.path.join(script_dir, "point_cloud.xyz"), "a") as f:
    #     for line in out_arr:
    #         coor = np.array2string(line, separator = " ").strip()
    #         f.write(coor[1:-1] + "\n")
    
    return pcd_with_normals

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_box_aspect(aspect = (2,2,0.5))
# plot_pcd(ax, r"C:\Users\User\Documents\Augmentus\MechMind\pratt\result_files\defect_pc_00024.pcd", 100)
# plt.show()
