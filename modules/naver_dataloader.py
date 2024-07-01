import kapture.io.csv as csv
from tqdm import tqdm
import open3d as o3d
import numpy as np
import os
import quaternion


class DataLoader:

    def __init__(self, load_path):
        real_load_path = os.path.join(load_path, 'release', 'mapping_lidar_only')
        self.tar_handlers = csv.get_all_tar_handlers(real_load_path)
        self.kapture_data = csv.kapture_from_dir(real_load_path, tar_handlers=self.tar_handlers)
        self.data_path = os.path.join(real_load_path, 'sensors', 'records_data')
    

    def get_data(self):
        tfs = []
        pcs = []

        # loop over the nested trajectories [timestamp][device_id] = pose -> position
        for timestamps, poses in tqdm(self.kapture_data.trajectories.items(), desc='Loading data'):
            # point cloud
            for lidar in range(2):
                if lidar == 0:
                    pcd_file = os.path.join(self.data_path, 'lidar' + str(lidar), str(timestamps) + '.pcd')
                    pc_o3d = o3d.io.read_point_cloud(pcd_file)
                    pc_o3d = pc_o3d.voxel_down_sample(voxel_size=0.1)
                    pc_np = np.asarray(pc_o3d.points)
                    pcs.append(pc_np)
            # transform
            for sensor_id, pose in poses.items():
                if sensor_id == 'lidar0':
                    R = quaternion.as_rotation_matrix(pose.inverse().r)
                    t = pose.inverse().t
                    tfs.append(np.hstack([R, t]))

        self.tar_handlers.close()

        return tfs, pcs
    