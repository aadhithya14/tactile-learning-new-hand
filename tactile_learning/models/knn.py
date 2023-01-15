import numpy as np
from copy import deepcopy as copy

# Custom nearest neighbor implementation
class KNearestNeighbors(object):
    def __init__(self, input_values, output_values):
        self.input_values = input_values
        self.output_values = output_values

    def get_sorted_idxs(self, datapoint):
        l1_distances = self.input_values - datapoint
        l2_distances = np.linalg.norm(l1_distances, axis = 1)

        sorted_idxs = np.argsort(l2_distances)
        return sorted_idxs

    def get_nearest_neighbor(self, datapoint):
        sorted_idxs = self.get_sorted_idxs(datapoint)
        nn_idx = sorted_idxs[0]
        return self.output_values[nn_idx], nn_idx

    def get_k_nearest_neighbors(self, datapoint, k):
        if k == 1:
            return self.get_nearest_neighbor(datapoint)

        assert datapoint.shape == self.input_values[0].shape

        sorted_idxs = self.get_sorted_idxs(datapoint)
        k_nn_idxs = sorted_idxs[:k]
        return self.output_values[k_nn_idxs], k_nn_idxs

# It uses each part separately to scale the distances 
class ScaledKNearestNeighbors(object):
    
    def __init__(
        self,
        input_values,
        output_values,
        repr_types,
        tactile_repr_size=64
    ):
        self.input_values = input_values 
        self.output_values = output_values 
        self.repr_types = repr_types
        self.tactile_repr_size = tactile_repr_size
        self._get_index_values() # Will set the beginning and ending indices for each repr type

    def _get_index_values(self):
        self.index_values = {}
        last_index = 0
        for repr_type in self.repr_types:
            if repr_type == 'image':
                self.index_values['image'] = [last_index, last_index+512]
                last_index += 512
            elif repr_type == 'tactile':
                self.index_values['tactile'] = [last_index, last_index+self.tactile_repr_size]
                last_index += self.tactile_repr_size
            elif repr_type == 'kinova':
                self.index_values['kinova'] = [last_index, last_index+7]
                last_index += 7
            elif repr_type == 'allegro':
                self.index_values['allegro'] = [last_index, last_index+12]
                last_index += 12

        print('SCALED KNN - self.index_values: {}'.format(self.index_values))

    def _get_type_based_dist(self, l1_distances, repr_type):
        type_based_idx = self.index_values[repr_type]
        type_l1_dist = l1_distances[:,type_based_idx[0]:type_based_idx[1]]
        type_l2_dist = np.linalg.norm(type_l1_dist, axis = 1)

    # print('type: {} - type_l2_dist.shape: {}'.format(repr_type, type_l2_dist.shape))
        type_l2_dist = type_l2_dist / (type_l2_dist.max() - type_l2_dist.min())
        return type_l2_dist

    def _get_l2_distances(self, datapoint):
        l1_distances = self.input_values - datapoint
        for i,repr_type in enumerate(self.repr_types):
            curr_l2_dist = self._get_type_based_dist(l1_distances, repr_type)
            # if i == 0: 
            #     final_l2_dist = curr_l2_dist
            # else:
            #     final_l2_dist += curr_l2_dist # Each distance will be scaled on its own
            if i == 0: 
                final_l2_dist = copy(curr_l2_dist)
                final_l2_dist_arr = np.expand_dims(curr_l2_dist, 1)
            else:
                final_l2_dist += curr_l2_dist
                # print('final_l2_dist_arr.shape: {}, curr_l2_dist.shape: {}'.format(final_l2_dist_arr.shape, curr_l2_dist.shape))
                final_l2_dist_arr = np.concatenate([final_l2_dist_arr, np.expand_dims(curr_l2_dist, 1)], axis=1)

        return final_l2_dist, final_l2_dist_arr

    def get_sorted_idxs(self, datapoint):
        l2_distances, separate_l2_distances = self._get_l2_distances(datapoint)
        
        sorted_idxs = np.argsort(l2_distances)
        sorted_separate_l2_dists = separate_l2_distances[sorted_idxs]
        return sorted_idxs, sorted_separate_l2_dists

    def get_nearest_neighbor(self, datapoint):
        sorted_idxs = self.get_sorted_idxs(datapoint)
        nn_idx = sorted_idxs[0]
        return self.output_values[nn_idx], nn_idx

    def get_k_nearest_neighbors(self, datapoint, k):
        if k == 1:
            return self.get_nearest_neighbor(datapoint)

        assert datapoint.shape == self.input_values[0].shape

        sorted_idxs, sorted_separate_l2_dists = self.get_sorted_idxs(datapoint)
        k_nn_idxs = sorted_idxs[:k]
        k_nn_separate_dists = sorted_separate_l2_dists[:k] # This will have separate dists for each representation
        return self.output_values[k_nn_idxs], k_nn_idxs, k_nn_separate_dists