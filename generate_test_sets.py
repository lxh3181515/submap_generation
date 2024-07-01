import pandas as pd
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import random
import argparse

from submap import farthest_point_sampling


def generate_test_set(df_locations, test_set_num):
	test_set_idx = []
	np_locations = df_locations.values
	test_set_idx, _ = farthest_point_sampling(np_locations, test_set_num)

	return test_set_idx


def output_to_file(output, filename):

	with open(filename, 'wb') as handle:
		pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
		
	print('Done', filename)


def construct_query_and_database_sets(base_path, runs_folder, clouds_folder, pose_file, output_file):
	
	runs_path = os.path.join(base_path, runs_folder)
	# folders = sorted(os.listdir(runs_path))
	folders = ['south_3f_run1', 'south_4f_run1', 'south_5f_run1', 'south_5f_run2']

	# Build KD-trees for every runs
	db_trees = []
	for folder in folders:
		print(folder)
		# Build KDTree for positive matches searching
		df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
		pose_path = os.path.join(runs_path, folder, pose_file)
		df_locations = pd.read_csv(pose_path, sep=',')
		for index, row in df_locations.iterrows():
			df_database.loc[len(df_database)] = row
		db_treee = KDTree(df_database[['northing','easting']])
		db_trees.append(db_treee)

	# Build database and test sets
	db_sets = []
	test_sets = []
	for folder in folders:
		# database: [0:{'query':path_to_pcd, 'northing':x, 'easting':y}, ...]
		# test:     [0:{'query':path_to_pcd, 'northing':x, 'easting':y}, ...]
		db = {}
		test = {}
		pose_path = os.path.join(runs_path, folder, pose_file)
		df_locations = pd.read_csv(pose_path, sep=',')
		df_locations['file'] = [os.path.join(runs_folder, folder, clouds_folder, df_locations['file'][i] + '.pcd') for i in range(df_locations.shape[0])]
		
		for index, row in df_locations.iterrows():
			if folder == 'south_4f_run2' or folder == 'south_5f_run2':
				test[len(test.keys())] = {'query':row['file'], 'northing':row['northing'], 'easting':row['easting']}
			db[len(db.keys())] = {'query':row['file'], 'northing':row['northing'], 'easting':row['easting']}
		db_sets.append(db)
		test_sets.append(test)	

	# Build positive matches
	for i in range(len(db_sets)):
		tree = db_trees[i]
		for j in range(len(test_sets)):
			if i == j or not bool(test_sets[j]):
				continue

			for key in range(len(test_sets[j].keys())):
				coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
				index = tree.query_radius(coor, r=10)
				# test: [0:{'query':path_to_pcd, 'northing':x, 'easting':y, i:[idx_of_P_matches]}, ...]
				test_sets[j][key][i] = index[0].tolist()
				### special case
				if (i == 2 and j == 3):
					pass
				else:
					test_sets[j][key][i] = []
				###
			print('Test:')
			print(list(test_sets[j].items())[:5])
 
	output_to_file(db_sets, output_file + '_evaluation_database.pickle')
	output_to_file(test_sets, output_file + '_evaluation_query.pickle')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--datasets', type=str, required=True, help='Path to datasets')

	args = parser.parse_args()
	base_path = args.datasets
	clouds_folder  = 'submaps'
	pose_file = 'pointcloud_centroids.csv'
	
	# corridor
	runs_folder = 'corridor'
	construct_query_and_database_sets(base_path, runs_folder, clouds_folder, pose_file, 'corridor')

