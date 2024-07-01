import open3d as o3d
import numpy as np
import os
import copy

from submap import transform_to_world


def load_clouds_poses(base_path):
    # Load clouds
    cloud_path = os.path.join(base_path, 'PCD')
    pcd_files = os.listdir(cloud_path)
    pcd_files = sorted(pcd_files, key=lambda x: int(x[6:-4]))
    print(pcd_files[0:3])
    clouds_o3d = [o3d.io.read_point_cloud(os.path.join(cloud_path, file)) for file in pcd_files]
    clouds_np = [np.asarray(cl.points) for cl in clouds_o3d]

    # Load poses
    trajectory_path = os.path.join(base_path, 'pose.txt')
    with open(trajectory_path, 'r') as file:
        poses = [np.vstack(list(map(float, line.strip().split()))).reshape(3, 4) for line in file]

    assert len(clouds_np) == len(poses)
    print('Total frames:', len(clouds_np))
    return clouds_np, poses


def voxel_downsample(cloud, voxel_size=0.05):
    """
    使用体素下采样方法减少点的数量
    :param cloud: 输入的点云
    :param voxel_size: 体素的大小
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)

    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return downsampled_pcd


base_path = 'dataset/corridor'

run_paths = ['south_4f_run1', 'south_4f_run2']

# Load clouds and trajectory
clouds1, poses1 = load_clouds_poses(os.path.join(base_path, run_paths[0]))
clouds2, poses2 = load_clouds_poses(os.path.join(base_path, run_paths[1]))

source = transform_to_world(clouds1, poses1)
target = transform_to_world(clouds2, poses2)

voxel_size = 1.0
source = voxel_downsample(source, voxel_size)
target = voxel_downsample(target, voxel_size)

initial_transformation = np.array([
    [ 9.97229622e-01, -7.13512485e-02, -2.10257093e-02,  1.17772961e+02],
    [ 7.08538391e-02,  9.97209313e-01, -2.35227379e-02,  1.99729454e+01],
    [ 2.26454098e-02,  2.19678188e-02,  9.99502176e-01, -6.03174534e-01],
    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
])
# source.transform(initial_transformation)

# 可视化原始点云
source.paint_uniform_color([1.0, 0, 0])
target.paint_uniform_color([0, 0, 1.0])
o3d.visualization.draw_geometries([source, target])

# 应用 ICP 配准
threshold = 10  # 设置一个阈值，这个值依赖于场景
icp_result = o3d.pipelines.registration.registration_generalized_icp(
    source, target, threshold, initial_transformation
)

T = icp_result.transformation
source.transform(T)

o3d.visualization.draw_geometries([source, target])

print("变换矩阵:")
print(T)

poses1_t = []
for p in poses1:
    R = np.dot(T[:3, :3], p[:, :3])
    t = np.dot(T[:3, :3], p[:, -1]) + T[:3, -1]
    poses1_t.append(np.insert(R, 3, t, axis=1).reshape(-1, 12))
poses1_t = np.vstack(poses1_t)
poses2 = np.vstack(poses2).reshape(-1, 12)


# 轨迹
traj1 = poses1_t[:, (3, 7, 11)]
traj2 = poses2[:, (3, 7, 11)]

o3d_traj1 = o3d.geometry.PointCloud()
o3d_traj2 = o3d.geometry.PointCloud()
o3d_traj1.points = o3d.utility.Vector3dVector(traj1)
o3d_traj2.points = o3d.utility.Vector3dVector(traj2)
o3d_traj1.paint_uniform_color([1.0, 0, 0])
o3d_traj2.paint_uniform_color([0, 0, 1.0])

o3d.visualization.draw_geometries([o3d_traj1, o3d_traj2])

# 输出
out_file_name = 'refine_pose.txt'
np.savetxt(os.path.join(base_path, run_paths[0], out_file_name), poses1_t, delimiter=' ')
np.savetxt(os.path.join(base_path, run_paths[1], out_file_name), poses2, delimiter=' ')
