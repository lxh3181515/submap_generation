import numpy as np
import open3d as o3d
import os
from tqdm import tqdm


class DataLoader:

    def __init__(self, base_path):
        self.base_path = base_path

    def get_poses(self, pose_file='refine_pose.txt'):
        trajectory_path = os.path.join(self.base_path, pose_file)
        with open(trajectory_path, 'r') as file:
            poses = [np.vstack(list(map(float, line.strip().split()))).reshape(3, 4) for line in file]
        print('Total poses:', len(poses))

        return poses

    def get_clouds(self):
        cloud_path = os.path.join(self.base_path, 'PCD')
        pcd_files = os.listdir(cloud_path)
        pcd_files = sorted(pcd_files, key=lambda x: int(x[6:-4]))
        print('PCD files name:', pcd_files[0:3], '...')
        # Load point clouds
        clouds_np = []
        for file in tqdm(pcd_files, desc='Loading PCD'):
            clouds_o3d = o3d.io.read_point_cloud(os.path.join(cloud_path, file))
            clouds_np.append(np.asarray(clouds_o3d.points))

        return clouds_np
    

    def get_data(self):
        return self.get_poses(), self.get_clouds()
