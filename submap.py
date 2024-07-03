import numpy as np
from tqdm import tqdm
import open3d as o3d
import os
import pandas as pd
from modules.corridor_dataloader import DataLoader
# from modules.naver_dataloader import DataLoader
from multiprocessing import Pool

from vis_pointcloud import visualize_clouds
from ground_segmentation import remove_ground_ceiling, remove_ground


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


def get_submaps_centroid(poses, centroid_dis):
    centroid_idxs = [0]
    
    for idx, pose in enumerate(poses):
        if idx == 0:
            continue
        # Skip the close frames
        c = centroid_idxs[-1]
        if get_distance(poses[c], pose) < centroid_dis:
            continue
        # Add to centroids
        centroid_idxs.append(idx)
    return centroid_idxs


def output_to_csv(submaps_poses, base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    df = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_path = os.path.join(base_path, 'pointcloud_centroids.csv')
    for idx, pose in enumerate(submaps_poses):
        row = ['submap_'+'%04d'%idx, pose[0, 3], pose[1, 3]]
        df.loc[len(df)] = row
    print('Export csv:', df_path)
    df.to_csv(df_path, sep=',', index=False, header=True)


def output_to_pcd(num, submap, base_path):
    root = os.path.join(base_path, 'submaps')
    if not os.path.exists(root):
        os.makedirs(root)
    
    file_path = os.path.join(root, 'submap_' + '%04d'%num + '.pcd')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(submap)
    o3d.io.write_point_cloud(file_path, pcd)


def crop_point_cloud(pc, radius):
    pc_crop = []
    for p in pc:
        if np.linalg.norm(p[:2]) <= radius:
            pc_crop.append(p)
    return np.vstack(pc_crop)


def get_submaps_task(cii, centroid_idxs, width, clouds, poses, output_path, align, view):
    voxel_size = 0.1
    points_num = 4096

    ## Get pointclouds indexs in range
    if cii == 0:
        si = centroid_idxs[cii]
        ei = centroid_idxs[cii+1]
    elif cii == len(centroid_idxs)-1:
        si = centroid_idxs[cii-1]
        ei = centroid_idxs[cii]
    else:
        si = centroid_idxs[cii-1]
        ei = centroid_idxs[cii+1]
    cl = [np.array(clouds[i]) for i in range(si, ei)]
    po = [np.array(poses[i]) for i in range(si, ei)]
    ref = np.array(poses[centroid_idxs[cii]])
    # world frame -> centroid frame
    if align:
        R_ref_inv = np.eye(3)
        ref[-1, -1] = 0
    else:
        R_ref_inv = ref[:, :3].T
    t_ref_inv = np.dot(-R_ref_inv, ref[:, -1])
    for i in range(len(po)):
        po[i][:, :3] = np.dot(R_ref_inv, po[i][:, :3])
        po[i][:, -1] = np.dot(R_ref_inv, po[i][:, -1]) + t_ref_inv
    submap = transform_to_world(cl, po)

    ## Down sample
    pc_view = []
    # Voxel grid filter
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(submap)
    pc_view.append(o3d.geometry.PointCloud(pc))
    pc = pc.voxel_down_sample(voxel_size=voxel_size)
    pc = np.asarray(pc.points)
    # Remove ground plane
    # Dept.1F/B1:-0.35, 2.95;
    # Stat.B1/B2:-0.35, 2.00;
    # corridor:-0.15, 2.45
    pc = remove_ground_ceiling(pc, -0.15, 2.45) 
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pc)
    pc_view.append(pc_o3d)
    # Crop
    pc = crop_point_cloud(pc, width)
    # FPS
    _, pc = farthest_point_sampling(pc, points_num)
    # Scale to [-1, 1]
    pc = pc / width

    output_to_pcd(cii, pc, output_path)
    if view:
        visualize_clouds(pc_view[0], 'before')
        visualize_clouds(pc_view[1], 'after')


def get_submaps(centroid_idxs, width, clouds, poses, output_path, align=False, view=False):
    # tqdm setting
    pbar = tqdm(total=len(centroid_idxs))
    pbar.set_description('Generating submaps')
    update = lambda *args: pbar.update()

    # Multi processing
    n_proc = 12
    if view:
        n_proc = 1
    pool = Pool(n_proc)
    for cii in range(len(centroid_idxs)):
        pool.apply_async(get_submaps_task, args=(cii, centroid_idxs, width, clouds, poses, output_path, align, view), callback=update)
    pool.close()
    pool.join()


if __name__ == '__main__':

    run_path = 'corridor/south_4f_run2'
    
    data_folder = 'dataset'
    output_folder = 'output'

    load_path = os.path.join(data_folder, run_path)
    output_path = os.path.join(output_folder, run_path)

    # Load clouds and trajectory
    dataloader = DataLoader(load_path)
    poses, clouds = dataloader.get_data()

    # Get centroids
    centroid_dis = 1.0  # meter
    centroid_idxs = get_submaps_centroid(poses, centroid_dis)
    output_to_csv([poses[i] for i in centroid_idxs], output_path)
    
    # Get submaps
    submap_width = 10.0
    get_submaps(centroid_idxs, submap_width, clouds, poses, output_path, align=True, view=False)
    print('Done.')
