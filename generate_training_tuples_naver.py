# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
import tqdm


P1 = [470, -430]
P2 = [470, -375]
P3 = [490, -450]
P4 = [490, -385]

# For training and test data splits
X_WIDTH = 10
Y_WIDTH = 10

# Test set boundaries
P = [P1, P2, P3, P4]

RUNS_FOLDER = "NAVER/HyundaiDepartmentStore/"
FILENAME = "pointcloud_centroids.csv"
POINTCLOUD_FOLS = "/submaps/"


class TrainingTuple:
    # Tuple describing an element for training/validation
    def __init__(self, id: int, timestamp: int, rel_scan_filepath: str, positives: np.ndarray,
                 non_negatives: np.ndarray, position: np.ndarray):
        # id: element id (ids start from 0 and are consecutive numbers)
        # ts: timestamp
        # rel_scan_filepath: relative path to the scan
        # positives: sorted ndarray of positive elements id
        # negatives: sorted ndarray of elements id
        # position: x, y position in meters (northing, easting)
        assert position.shape == (2,)

        self.id = id
        self.timestamp = timestamp
        self.rel_scan_filepath = rel_scan_filepath
        self.positives = positives
        self.non_negatives = non_negatives
        self.position = position


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set


def construct_query_dict(df_centroids, base_path, filename, ind_nn_r, ind_r_r=2):
    # ind_nn_r: threshold for positive examples
    # ind_r_r: threshold for negative examples

    tree = KDTree(df_centroids[['northing', 'easting']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_nn_r)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting']], r=ind_r_r)
    queries = {}
    for anchor_ndx in range(len(ind_nn)):
        anchor_pos = np.array(df_centroids.iloc[anchor_ndx][['northing', 'easting']])
        query = df_centroids.iloc[anchor_ndx]["file"]
        # Extract timestamp from the filename
        # scan_filename = os.path.split(query)[1]
        # assert os.path.splitext(scan_filename)[1] == '.bin', f"Expected .bin file: {scan_filename}"
        # timestamp = int(os.path.splitext(scan_filename)[0])
        timestamp = anchor_ndx

        positives = ind_nn[anchor_ndx]
        non_negatives = ind_r[anchor_ndx]

        positives = positives[positives != anchor_ndx]
        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(id=anchor_ndx, timestamp=timestamp, rel_scan_filepath=query,
                                            positives=positives, non_negatives=non_negatives, position=anchor_pos)

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root

    all_folders = sorted(os.listdir(os.path.join(base_path, RUNS_FOLDER)))
    folders = []

    # All runs are used for training (both full and partial)
    index_list = range(len(all_folders))
    print("Number of runs: " + str(len(index_list)))
    for index in index_list:
        folders.append(all_folders[index])
    print(folders)

    df_train = pd.DataFrame(columns=['file', 'northing', 'easting'])
    df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

    for folder in tqdm.tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, RUNS_FOLDER, folder, FILENAME), sep=',')
        # df_locations['timestamp'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['timestamp'].astype(str) + '.bin'
        # df_locations = df_locations.rename(columns={'timestamp': 'file'})
        df_locations['file'] = RUNS_FOLDER + folder + POINTCLOUD_FOLS + df_locations['file'].astype(str) + '.pcd'

        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                df_test = pd.concat([df_test, pd.DataFrame([row])], ignore_index=True)
            else:
                df_train = pd.concat([df_train, pd.DataFrame([row])], ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    print("Number of non-disjoint test submaps: " + str(len(df_test['file'])))
    # ind_nn_r is a threshold for positive elements - 10 is in original PointNetVLAD code for refined dataset
    construct_query_dict(df_train, base_path, "training_queries_naver_v2.pickle", ind_nn_r=2)
    construct_query_dict(df_test, base_path, "test_queries_naver_v2.pickle", ind_nn_r=2)
