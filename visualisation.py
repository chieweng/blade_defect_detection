import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from functions import *
from file_processing import *
from point_cloud import *

script_dir = os.path.dirname(os.path.abspath(__file__))
directory = os.path.join(script_dir, "result_files")

def plot():
    filenames_json, file_contents, dim_list, dimensions = file_processing_json()
    defect_filenames, defect_file_contents, object_file_content = file_processing_pcd()
    box_dimension_list = dimensions[dim_list[0]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    #robot base
    origin = np.eye(4)
    plot_orientation(ax, origin, label="robot base")

    defect_pos_lst = []
    waypoints = []

    for filename in filenames_json:
        # extract position and rotation data
        pos, rot = file_contents[filename][:3], file_contents[filename][3:7] #do not use rot from here, rot in quaternion
        defect_pos_lst.append(np.array(pos))   
    
    object_pcd = o3d.io.read_point_cloud(directory + "\{}".format(next(iter(object_file_content))))

    for i, filename in enumerate(defect_filenames):
        pcd = o3d.io.read_point_cloud(directory + "\{}".format(filename))
        pcd_with_normal = plot_pcd(ax, directory + "\{}".format(filename), 100)

        defect_pos = defect_pos_lst[i]
        defect_normal = calculate_normal(defect_pos, pcd_with_normal)
        print(f"Calculated normal for defect position: {defect_pos}")

        aabb, R = pose_aligned_bounding_box(pcd, defect_pos, defect_normal, show = False)
        
        pos = aabb.center
        rot = R.T
        dimension = aabb.extent

        defect_pose = np.vstack([np.hstack([rot, pos.reshape(3,1)]), [0, 0, 0, 1]])
        pos_lst, box_size = bounding_box_3d(ax, defect_pose, dimension, filenames_json[i], show = False)
        
        aligned_pos_lst = align_bounding_box_with_point_cloud(ax, pos_lst, object_pcd, show = False)
        waypoint = raster(ax, aligned_pos_lst, box_size, show = True)
        waypoints.append(waypoint)
        print(f"Generated waypoints for {filename}")

    # axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plot limits
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 0.5])

    plt.show()
    print("Plot displayed.")
    
    return(waypoints)

print("Script execution started.")
plot()
print("Script execution finished.")

'''
current bugs:
1) odd blob shapes/ defects on a very curved surface causes a very convexed shape, o3d.get_minimal_bounding_box()
might return bounding box aligned with XZ or YZ plane instead of XY plane. (fixed)
2) blob with elongated corner will not result in optimal bounding box too.

improvements:
1) generate non-planar raster movement: project bounding box onto highest surface layer of blob point cloud for more robust surface tracking and handling of curvatures. [done]
2) current raster: left1 -> right1 -> right2 -> left2 -> left 3....... ; 
   improved raster: left1 -> center1.0 -> center 1.1 -> center 1.2 -> right1 -> right2 -> center2.2 -> center2.1 -> center2.0 -> left2 [done]
3) add orientation of TCP (improved collision avoidance) with the surrounding surfaces of the object, TCP should reorientate between each waypoints in the improved raster
for better surface tracking
4) more tight fitting obb to account for a more optimal obb size. current o3d.obb is calculated using PCA -> convex hull; other methods to calculate bounding box:
    a. alpha shapes [ ]
    b. concave hull [ ]
    c. axis aligned bounding box (edges of box aligned with x, y and z axis) [not optimal]
    d. oriented bounding box using PCA [currently implemented]
'''