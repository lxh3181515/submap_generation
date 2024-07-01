import numpy as np
import open3d as o3d
from vis_pointcloud import visualize_clouds


def remove_ground(pc, ceiling=False, show=False):
    angle_threshold = np.deg2rad(10)
    num_iter = 10
    
    # Search ground plane
    pc_copy = o3d.geometry.PointCloud(pc)
    for i in range(num_iter):
        plane_model, inliers = pc_copy.segment_plane(distance_threshold=0.1, ransac_n=5, num_iterations=100)
        # Get plane angle
        a, b, c, d = plane_model
        angle_with_z = np.arccos(c / np.sqrt(a*a + b*b + c*c))
        # Get plane z-axis
        z = -d / c
        # Check ground plane
        if angle_with_z < angle_threshold and z < 0:
            plane = pc_copy.select_by_index(inliers)
            break
        else:
            pc_copy = pc_copy.select_by_index(inliers, invert=True)
    pc_points = np.asarray(pc.points)
    ground_points = np.asarray(plane.points)
    under_ground_points = np.vstack([p for p in pc_points if a*p[0]+b*p[1]+c*p[2]+d < 0])
    remove_points = np.concatenate((ground_points, under_ground_points), axis=0)
    
    # Search ceiling plane
    if ceiling:
        pc_copy = o3d.geometry.PointCloud(pc)
        for i in range(num_iter):
            plane_model, inliers = pc_copy.segment_plane(distance_threshold=0.2, ransac_n=10, num_iterations=100)
            # Get plane angle
            a, b, c, d = plane_model
            angle_with_z = np.arccos(c / np.sqrt(a*a + b*b + c*c))
            # Get plane z-axis
            z = -d / c
            # Check ceiling plane
            if angle_with_z < angle_threshold and z > 2.5:
                plane = pc_copy.select_by_index(inliers)
                break
            else:
                pc_copy = pc_copy.select_by_index(inliers, invert=True)
        pc_points = np.asarray(pc.points)
        ceiling_points = np.asarray(plane.points)
        above_ceiling_points = np.vstack([p for p in pc_points if a*p[0]+b*p[1]+c*p[2]+d > 0])
        remove_points = np.concatenate((remove_points, ceiling_points, above_ceiling_points), axis=0)
    
    # Get inlier and outlier pointcloud
    remove_points = np.unique(remove_points, axis=0)
    indices_to_remove = np.array([np.where((pc_points == point).all(axis=1))[0][0] for point in remove_points])
    inlier_cloud = pc.select_by_index(indices_to_remove)
    outlier_cloud = pc.select_by_index(indices_to_remove, invert=True)

    if show:
        inlier_cloud.paint_uniform_color([1.0, 0, 0])
        outlier_cloud.paint_uniform_color([0, 1.0, 0])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    
    return outlier_cloud


def remove_ground_ceiling(pc, ground_z_th, ceiling_z_th):
    pc_nonground = []
    for pt in pc:
        if pt[2] < ground_z_th or pt[2] > ceiling_z_th:
            continue
        pc_nonground.append(pt)
    return np.vstack(pc_nonground)


if __name__ == '__main__':
    pcd_file = 'cmp/submap_0000_raw.pcd'
    pc_o3d = o3d.io.read_point_cloud(pcd_file)
    pc_np = np.asarray(pc_o3d.points)

    visualize_clouds(pc_o3d, 'before segmentation')

    ground_z_th = -0.05
    ceiling_z_th = 2.0
    pc_np = remove_ground_ceiling(pc_np, ground_z_th, ceiling_z_th)

    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc_np)
    visualize_clouds(pc_o3d, 'after segmentation')