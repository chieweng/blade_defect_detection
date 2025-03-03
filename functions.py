import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def xyz_euler_to_4x4(pos, rot):
    '''returns 4x4 pose matrix given an array [x,y,z,rx,ry,rz]'''
    pos = np.array(pos)
    pos_reshaped = pos.reshape(3,1)
    rot_3x3 = euler_to_rotation_matrix(rot[0], rot[1], rot[2])
    pose_4x4 = np.vstack([np.hstack([rot_3x3, pos_reshaped]), [0, 0, 0, 1]])
    return pose_4x4

def extract_pos(pose_lst): 
    '''extract pos xyz from 4x4 pose matrix'''
    return [pose[:3, 3] for pose in pose_lst]

def euler_to_rotation_matrix(yaw, pitch, roll): #ZYX
    r_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])
    r_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    r_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    
    return np.dot(r_yaw, np.dot(r_pitch, r_roll))


def quaternion_to_euler(x, y, z, w):
     t0 = +2.0 * (x * x + y * z)
     t1 = +1.0 - 2.0 * (x * x + y * y)
     roll_x = np.arctan2(t0, t1)

     t2 = +2.0 * (w * y - z * x)
     t2 = +1.0 if t2 > +1.0 else t2
     t2 = -1.0 if t2 < -1.0 else t2
     pitch_y = np.arcsin(t2)

     t3 = +2.0 * (w * z + x * y)
     t4 = +1.0 - 2.0 * (y * y + z * z)
     yaw_z = np.arctan2(t3, t4)

     return roll_x, pitch_y, yaw_z


def euler_to_quaternion(rx, ry, rz):
    # Convert Euler angles to quaternion
    cy = np.cos(ry * 0.5)
    sy = np.sin(ry * 0.5)
    cp = np.cos(rx * 0.5)
    sp = np.sin(rx * 0.5)
    cr = np.cos(rz * 0.5)
    sr = np.sin(rz * 0.5)

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return qw, qx, qy, qz

def axis_angle_rotation(pose, axis, angle): 
    '''note: axis of rotation from [0,0,0]'''
    # normalize axis vector
    axis = axis / np.linalg.norm(axis)

    # calculate the rotation matrix using Rodrigues' rotation formula
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    cross_product_matrix = np.array([[0,-axis[2], axis[1]],
                                     [axis[2], 0, -axis[0]],
                                     [-axis[1], axis[0], 0]])
    
    rotation_matrix = cos_theta * np.eye(3) + sin_theta * cross_product_matrix + (1 - cos_theta) * np.outer(axis, axis)

    # create a 4x4 rotation matrix
    rotation_matrix_4x4 = np.eye(4)
    rotation_matrix_4x4[:3, :3] = rotation_matrix

    # perform rotation by left-multiplying the rotation matrix
    rotated_matrix = np.dot(rotation_matrix_4x4, pose)

    return rotated_matrix

def axis_angle_representation(pose1, pose2):
    '''returns the rotation axis and rotation angle required to rotate pose1 to pose2'''
    rot1 = pose1[:3, :3]
    rot2 = pose2[:3, :3]
    
    # calculate the rotation vector using axis-angle representation
    r = R.from_matrix(np.dot(rot2, np.linalg.inv(rot1)))
    rotation_vector = r.as_rotvec()
    
    angle = np.linalg.norm(rotation_vector)
    axis = rotation_vector / angle if angle != 0 else np.zeros_like(rotation_vector)

    return axis, angle

def reorient_pose(pose):
    '''reorient a given pose such that it has a non-negative z-component (z-axis points upwards)'''
    rotation_matrix = pose[:3, :3]

    z_axis = rotation_matrix[:, 2]

    if z_axis[2] >= 0: return pose
    else:
        rot = np.array([[1,  0,  0], #rotate about x-axis by 180 deg
                                               [0, -1,  0],
                                               [0,  0, -1]])
        rotation_matrix_new = np.dot(rotation_matrix, rot)
        pose[:3, :3] = rotation_matrix_new

        return pose

#------------------------- matplotlib ---------------------------#
def plot_orientation(ax, pose, label="", scale=0.05):
    origin = pose[:3, 3]
    x_axis = pose[:3, 0] * scale + origin
    y_axis = pose[:3, 1] * scale + origin
    z_axis = pose[:3, 2] * scale + origin

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='red')   # X-axis
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='green') # Y-axis
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='blue')   # Z-axis

    if label:
        ax.text(origin[0], origin[1], origin[2], label, color='black', fontsize=9)
        

def plot_line(ax, point1, point2, color='black', linestyle='--', label = ""):
    """Function to plot a line between two points."""
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=color, linestyle=linestyle, linewidth = 0.1)


def plot_line_with_arrow(ax, point1, point2, color='blue', linestyle='-', arrow_length_ratio=0.3):
    # Plotting the line
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], color=color, linestyle=linestyle)

    # Calculating the direction vector for the arrow
    u = point2[0] - point1[0]
    v = point2[1] - point1[1]
    w = point2[2] - point1[2]

    # Normalize the direction vector
    norm = (u**2 + v**2 + w**2)**0.5
    u, v, w = u / norm, v / norm, w / norm

    # Shorten the arrow length based on the arrow_length_ratio
    u, v, w = u * arrow_length_ratio, v * arrow_length_ratio, w * arrow_length_ratio

    # Plotting the arrow
    ax.quiver(point2[0], point2[1], point2[2], u, v, w, color=color, length=norm, arrow_length_ratio=arrow_length_ratio)
    

#----------------------- oriented bounding box -----------------------------# (refer to point_cloud.py for other bounding box calculation methods)
def bounding_box_2d(ax, pos, box_dim, show = True): # width_x, length_y
    width, length, height = box_dim
    x, y, z = pos

    # corners of the bounding box
    corners = np.array([
        [x - width/2, y + length/2, z],
        [x + width/2, y + length/2, z],
        [x + width/2, y - length/2, z],
        [x - width/2, y - length/2, z]
    ])
    
    width_step_dist =  0.005
    length_step_dist = box_dim[1] / 16

    points = []
    
    # determine if the bounding box should be subdivided
    subdivide_width = width > width_step_dist
    subdivide_length = length > length_step_dist

    if subdivide_width:
        x_steps = np.linspace(corners[0][0], corners[1][0], num = int(width / width_step_dist) + 1)
    else:
        x_steps = [x - width/2, x + width/2]
    
    if subdivide_length:
        y_steps = np.linspace(corners[3][1], corners[0][1], num = int(length / length_step_dist))
    else:
        y_steps = [y - length/2, y + length/2]
    
    for x_pos in x_steps:
        for y_pos in y_steps:
            point = [x_pos, y_pos, z]
            pose = xyz_euler_to_4x4(point, [0, 0, 0])
            points.append(pose)
            
            if show:
                ax.scatter(*pose[:3, 3], color="lime")
                # plot_orientation(ax, pose, scale=0.02)

    return points, [len(x_steps), len(y_steps), height]


def bounding_box_3d(ax, pose, box_dim, filename, show = True): #dim (LxWxH)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim([-5, 5])
    # ax.set_ylim([-5, 5])
    # ax.set_zlim([-5, 5])
    
    points, box_size = bounding_box_2d(ax, pose[:3, 3], box_dim, show = False)
    height = box_size[2]

    initial_pose = xyz_euler_to_4x4([0, 0, 0], [0, 0, 0])

    pose = reorient_pose(pose)
    rot_axis, angle = axis_angle_representation(initial_pose, pose)
    
    rotated_points_lst = []
    for i, point in enumerate(points):
        translated_point = point - [[0, 0, 0, pose[0][3]], # offset translation
                                    [0 ,0, 0, pose[1][3]],
                                    [0, 0, 0, pose[2][3]],
                                    [0, 0, 0, 0         ]]
        rotated_point_homogenous = axis_angle_rotation(translated_point, rot_axis, angle)
        rotated_point = rotated_point_homogenous + [[0, 0, 0, pose[0][3]], # translation to account for offset
                                                    [0 ,0, 0, pose[1][3]],
                                                    [0, 0, 0, pose[2][3] + height/2],
                                                    [0, 0, 0, 0         ]]
        # rotated_point = np.dot(rotated_point, [[0, 0, 0, 0       ], # translation wrt itself, post multiplication
        #                                        [0 ,0, 0, 0       ],
        #                                        [0, 0, 0, height/2],
        #                                        [0, 0, 0, 1       ]])
        rotated_points_lst.append(rotated_point)
        
        'plot rotated points for each point in bounding box'
        if show:
            ax.scatter(*rotated_point[:3, 3], color = "darkorange") 
            # ax.text(*rotated_point[:3, 3], s = i, color = 'black', fontsize = 10)
            # plot_orientation(ax, rotated_point, scale = 0.02)
    
    'plot defect pose and axis of rot'
    # if show:
        # ax.scatter(*pose[:3, 3], color = "red")
        # plot_orientation(ax, pose, scale = 0.02)
        # ax.text(*pose[:3, 3], s = filename[:-4], color = 'black', fontsize = 10)
        # ax.quiver(*pose[:3, 3], *rot_axis, length = 0.03, arrow_length_ratio=0.1)

    # plt.show()

    return extract_pos(rotated_points_lst), box_size


def raster(ax, bounding_box_points, box_size, show = True): #ax, pose_lst, lengthxwidth
    num_rows, num_cols, _ = box_size

    # Rearrange the points in a zigzag pattern
    ordered_lst = []
    for row in range(num_rows):
        start_idx = row * num_cols
        end_idx = (row + 1) * num_cols
        if row % 2 == 0:
            ordered_lst.extend(bounding_box_points[start_idx:end_idx])
        else:
            ordered_lst.extend(bounding_box_points[start_idx:end_idx][::-1])

    ordered_lst = smoothen(ax, ordered_lst, box_size)

    if show:
        x_coords = [point[0] for point in ordered_lst]
        y_coords = [point[1] for point in ordered_lst]
        z_coords = [point[2] for point in ordered_lst]

        # Calculate differences between consecutive points
        dx = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
        dy = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        dz = [z_coords[i+1] - z_coords[i] for i in range(len(z_coords)-1)]

        # Plot arrows using quiver
        ax.quiver(x_coords[:-1], y_coords[:-1], z_coords[:-1], dx, dy, dz, color = 'cyan', arrow_length_ratio = 0.1)

    return ordered_lst


def smoothen(ax, points, box_size):
    num_rows, num_cols, _ = box_size
    points = np.array(points)
    smoothed_points = points.copy()

    # Get the valid neighbors for a point at position (i, j) in the grid
    def get_neighbors(i, j):
        neighbors = []
        # Possible neighbor offsets in row and column
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in neighbor_offsets:
            ni, nj = i + dr, j + dc
            if 0 <= ni < num_rows and 0 <= nj < num_cols:
                neighbors.append((ni, nj))
        
        return neighbors

    # Convert 2D index to 1D index in boustrophedon order
    def index_2d_to_1d(i, j):
        if i % 2 == 0:
            return i * num_cols + j
        else:
            return i * num_cols + (num_cols - 1 - j)

    # Iterate over each point in the grid
    for i in range(num_rows):
        for j in range(num_cols):
            # Get the 1D index of the current point
            idx = index_2d_to_1d(i, j)
            # Get the neighbors' indices
            neighbors = get_neighbors(i, j)
            neighbor_indices = [index_2d_to_1d(ni, nj) for ni, nj in neighbors]
            
            # Calculate the average z-coordinates
            if neighbor_indices:
                avg_z = np.mean([points[idx][2]] + [points[n_idx][2] for n_idx in neighbor_indices])
                smoothed_points[idx][2] = avg_z

    # [ax.scatter(*i, color = "deeppink") for i in smoothed_points];

    return smoothed_points.tolist()


"----------- raster(legacy) ------------"
# def raster(ax, bounding_box_points, show = True): #pose_lst
#     point_lst = bounding_box_points

#     #check if len is even
#     if len(point_lst) % 2 != 0: raise ValueError("Length of list is odd")

#     mid_len = len(point_lst) // 2
#     n1 = point_lst[:mid_len]
#     n2 = point_lst[mid_len:]

#     ordered_lst = []

#     for i in range(mid_len):
#         if i%2 == 0:
#             ordered_lst.append(n1[i])
#             ordered_lst.append(n2[i])
#         else:
#             ordered_lst.append(n2[i])
#             ordered_lst.append(n1[i])
    
#     """
#     plotting raster
#     """
#     ordered_pos_lst = extract_pos(ordered_lst)

#     if show:
#         x_coords = [point[0] for point in ordered_pos_lst]
#         y_coords = [point[1] for point in ordered_pos_lst]
#         z_coords = [point[2] for point in ordered_pos_lst]

#         # Calculate differences between consecutive points
#         dx = [x_coords[i+1] - x_coords[i] for i in range(len(x_coords)-1)]
#         dy = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
#         dz = [z_coords[i+1] - z_coords[i] for i in range(len(z_coords)-1)]

#         # Plot arrows using quiver
#         ax.quiver(x_coords[:-1], y_coords[:-1], z_coords[:-1], dx, dy, dz, color='cyan', arrow_length_ratio = 0.1)
        
#     return ordered_pos_lst

# def check_angle_xy_plane(point_normal):
#     norm = point_normal/np.linalg.norm(point_normal) #normalize
#     z_axis = np.array([0, 0, 1])

#     dot_prod = np.dot(norm, z_axis)
#     angle_rad = np.arccos(dot_prod)
#     angle_deg = np.degrees(angle_rad)

#     print(angle_deg)

"-------------------2d bounding box(legacy)----------------------"
# def bounding_box_2d(ax, pos, box_dim, show = True): #width_x, length_y
#     width = box_dim[0]
#     length = box_dim[1]
#     x, y, z = pos[0], pos[1], pos[2]
#     rot = np.array([0, 0, 0])
    
#     pos_2d = []     
#     pos_2d.append(np.array([x - width/2, y + length/2, z]))
#     pos_2d.append(np.array([x + width/2, y + length/2, z]))
#     pos_2d.append(np.array([x - width/2, y - length/2, z]))
#     pos_2d.append(np.array([x + width/2, y - length/2, z]))

#     step_dist = 0.005   #5mm step distance
#     edge_density = math.ceil(width/ step_dist) + 1
#     points = []
#     for i in range(2):
#         for j in range(0, edge_density + 1):
#             posi = pos_2d[i] + (pos_2d[i + 2] - pos_2d[i]) * (j / (edge_density))
#             pose = xyz_euler_to_4x4(posi, rot)
#             points.append(pose)
        
#             'plot pose and orientation of unrotated points of bounding box'
#             if show == True: 
#                 ax.scatter(*pose[:3, 3], color = "lime")
#                 plot_orientation(ax, pose, scale = 0.02)

#                 # if j == 0 or j == edge_density: ax.text(*pose[:3, 3], s = pose[:3, 3], color='black', fontsize=5)
    
#     return points # list of points poses