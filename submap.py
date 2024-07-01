import numpy as np
from tqdm import tqdm
import open3d as o3d
import os
import pandas as pd
import math

from scipy.spatial import KDTree
import matplotlib.pyplot as plt

from vis_pointcloud import visualize_clouds

def load_clouds_poses(base_path):
    # Load clouds
    cloud_path = os.path.join(base_path, 'PCD')
    pcd_files = os.listdir(cloud_path)
    pcd_files = sorted(pcd_files, key=lambda x: int(x[6:-4]))
    print(pcd_files[0:3])
    clouds_o3d = [o3d.io.read_point_cloud(os.path.join(cloud_path, file)) for file in pcd_files]
    clouds_np = [np.asarray(cl.points) for cl in clouds_o3d]

    # Load poses
    trajectory_path = os.path.join(base_path, 'refine_pose.txt')
    with open(trajectory_path, 'r') as file:
        poses = [np.vstack(list(map(float, line.strip().split()))).reshape(3, 4) for line in file]

    assert len(clouds_np) == len(poses)
    print('Total frames:', len(clouds_np))
    return clouds_np, poses


def transform_to_world(clouds, poses):
    assert len(clouds) == len(poses)
    full_map = []
    for cl, po in zip(clouds, poses):
        R = po[:, 0:3]
        t = po[:, -1]
        full_map.append(np.dot(R, cl.T).T + np.tile(t, (cl.shape[0], 1)))

    return np.vstack(full_map)

    
def get_distance(pose1, pose2):
    assert pose1.shape == pose2.shape
    assert np.size(pose1) == 12
    return np.linalg.norm(pose1[:2, -1] - pose2[:2, -1])


def farthest_point_sampling(points, num_samples):
    """
    实现最远点采样算法。
    
    参数:
        points (np.ndarray): 原始点云数据，形状为 (N, 3)。
        num_samples (int): 需要采样的点数。
    
    返回:
        np.ndarray: 采样后的点云数据。
    """
    N = points.shape[0]
    centroids = np.zeros((num_samples,), dtype=int)
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid)**2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance)
    
    return centroids, points[centroids]


def export_to_pcd(points, filename="output.pcd"):
    """
    将点云数据导出到PCD文件。
    
    参数:
        points (np.ndarray): 点云数据，形状为 (N, 3)，其中N是点的数量。
        filename (str): 输出文件的名称。
    """
    # 创建一个PointCloud对象
    pcd = o3d.geometry.PointCloud()
    
    # 将NumPy数组中的点赋值给PointCloud对象
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 写入PCD文件
    o3d.io.write_point_cloud(filename, pcd)


def get_submaps_centroid(poses, centroid_dis):
    centroid_idxs = [0]
    
    for idx, pose in enumerate(poses):
        if idx == 0:
            continue
        # Check if far away from all centroid
        valid_pose = True
        for c in reversed(centroid_idxs):
            if get_distance(poses[c], pose) < centroid_dis:
                valid_pose = False
                break
        if not valid_pose:
            continue
        # Add to centroids
        centroid_idxs.append(idx)
    print('Total submaps:', len(centroid_idxs))
    return centroid_idxs


def output_to_csv(submaps_poses, base_path):
    df = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_path = os.path.join(base_path, 'pointcloud_centroids.csv')
    for idx, pose in enumerate(submaps_poses):
        row = ['submap_'+'%04d'%idx, pose[0, 3], pose[1, 3]]
        df.loc[len(df)] = row
    print('Export csv:', df_path)
    df.to_csv(df_path, sep=',', index=False, header=True)


def output_ot_pcd(submaps, base_path):
    root = os.path.join(base_path, 'submaps')
    if not os.path.exists(root):
        os.makedirs(root)

    rm_files = os.listdir(root)
    for f in rm_files:
        os.remove(os.path.join(root, f))
    for i, m in enumerate(submaps):
        file_path = os.path.join(root, 'submap_' + '%04d'%i + '.pcd')
        export_to_pcd(m, file_path)
    print('Export pcd:', root)


def remove_ground(pc, ceiling=False, show=False):
    angle_threshold = np.deg2rad(10)
    num_iter = 10
    
    # Search ground plane
    pc_copy = o3d.geometry.PointCloud(pc)
    for i in range(num_iter):
        plane_model, inliers = pc_copy.segment_plane(distance_threshold=0.2, ransac_n=5, num_iterations=1000)
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
            plane_model, inliers = pc_copy.segment_plane(distance_threshold=0.2, ransac_n=10, num_iterations=1000)
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


def crop_point_cloud(pc, radius):
    pc_crop = []
    for p in pc:
        if np.linalg.norm(p[:2]) <= radius:
            pc_crop.append(p)
    return np.vstack(pc_crop)


def get_submaps(centroid_idxs, centroid_dis, width, clouds, poses):
    # Build a frame KDTree
    positions = [np.array(poses[i][:2, -1]) for i in range(len(poses))]
    positions = np.vstack(positions)
    frame_tree = KDTree(positions)

    # Get pointclouds in range
    submaps = []
    trajectorys = []
    for c_idx in centroid_idxs:
        query_pos = np.array(poses[c_idx][:2, -1])
        frames_idx = frame_tree.query_ball_point(query_pos, r=width*2, p=2)
        cl = [np.array(clouds[i]) for i in frames_idx]
        po = [np.array(poses[i]) for i in frames_idx]
        # world frame -> centroid frame
        ref = np.array(poses[c_idx])
        # R_ref_inv = ref[:, :3].T
        R_ref_inv = np.eye(3)
        t_ref_inv = np.dot(-R_ref_inv, ref[:, -1])
        for i in range(len(po)):
            po[i][:, :3] = np.dot(R_ref_inv, po[i][:, :3])
            po[i][:, -1] = np.dot(R_ref_inv, po[i][:, -1]) + t_ref_inv
        sm = transform_to_world(cl, po)
        submaps.append(sm)
        trajectorys.append(po)

    # Downsample
    voxel_size = 0.1
    points_num = 4096
    out_submaps = []
    for m in tqdm(submaps, desc='Processing'):
        # numpy -> o3d
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(m)
        # Voxel grid filter
        pc = pc.voxel_down_sample(voxel_size=voxel_size)
        # Remove outlier
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Remove ground plane
        # pc = remove_ground(pc, ceiling=True, show=True)
        # o3d -> numpy
        pc = np.asarray(pc.points)
        # Crop
        pc = crop_point_cloud(pc, width)
        # FPS
        _, pc = farthest_point_sampling(pc, points_num)
        # Scale to [-1, 1]
        pc = pc / width
        out_submaps.append(pc)
    return out_submaps


if __name__ == '__main__':
    
    base_path = 'dataset/corridor/south_5f_run2'

    # Load clouds and trajectory
    clouds, poses = load_clouds_poses(base_path)

    # Get centroids
    centroid_dis = 5.0  # meter
    centroid_idxs = get_submaps_centroid(poses, centroid_dis)
    output_to_csv([poses[i] for i in centroid_idxs], base_path)
    
    # Get submaps
    submap_width = 15.0
    print('Generating submaps...')
    submaps = get_submaps(centroid_idxs, centroid_dis, submap_width, clouds, poses)

    output_ot_pcd(submaps, base_path)
    print('Done.')
 