import open3d as o3d
import numpy as np
import os


def load_pcds(folder_path):
    # 获取文件夹内所有PCD文件
    pcd_files = os.listdir(folder_path)
    # pcd_files.sort(key = lambda x: int(x[:-4]))
    pcd_files.sort()
    print(pcd_files[0:3])

    print(f"Found {len(pcd_files)} PCD files in {folder_path}")

    # 读取并存储所有点云
    pcds = [o3d.io.read_point_cloud(os.path.join(folder_path, file)) for file in pcd_files]

    return pcds, pcd_files


def visualize_clouds(cloud, pcd_name, colors=None):

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=pcd_name, width=1920, height=1080)

    vis.add_geometry(cloud)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))

    # view_ctrl = vis.get_view_control()
    # view_ctrl.set_front([-1, 0, 0])
    # view_ctrl.set_up([0, -1, 0])

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':

    folder_path = '/home/lxhong/work_space/submap/output/corridor/south_3f_run1/submaps'
    pcds, files = load_pcds(folder_path)

    # base_path = '/home/lxhong/work_space/submap/dataset'
    # files = ['corridor/south_4f_run2/submaps/submap_0055.pcd',
    #          'corridor/south_3f_run1/submaps/submap_0014.pcd',
    #          'corridor/south_4f_run1/submaps/submap_0014.pcd',
    #          'corridor/south_3f_run1/submaps/submap_0055.pcd',
    #          'corridor/south_3f_run1/submaps/submap_0015.pcd',
    #          'corridor/south_3f_run1/submaps/submap_0054.pcd']
    # pcds = [o3d.io.read_point_cloud(os.path.join(base_path, p)) for p in files]

    pcs_np = [np.asarray(pcd.points) for pcd in pcds]

    print("Showing next point cloud. Press 'Q' to close the window and show next.")
    
    for i in range(len(pcds)):
        print('x_max:', max(pcs_np[i][:, 0]), 'x_min:', min(pcs_np[i][:, 0]))
        print('y_max:', max(pcs_np[i][:, 1]), 'y_min:', min(pcs_np[i][:, 1]))
        print('z_max:', max(pcs_np[i][:, 2]), 'z_min:', min(pcs_np[i][:, 2]))
        print('Points num:', pcs_np[i].shape[0])
        visualize_clouds(pcds[i], files[i])
