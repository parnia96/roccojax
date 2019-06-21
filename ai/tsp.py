

import numpy as np
import itertools
import time


def get_random_distances_matrix(cities_number, max_distance):

    asymmetric_distances_matrix = np.random.random_integers(1, max_distance, size=(cities_number, cities_number))
    distances_matrix = (asymmetric_distances_matrix + asymmetric_distances_matrix.transpose()) / 2
    distances_matrix[np.diag_indices(cities_number)] = 0

    return distances_matrix


def get_trip_distance(distances_matrix, path):
    distance = 0

    for index in range(len(path))[1:]:

        distance += distances_matrix[path[index - 1], path[index]]

    return distance


class BruteForceTSPSolver:
    def __init__(self, distances_matrix):

        self.distances_matrix = distances_matrix

    def solve(self):

        cities_number = self.distances_matrix.shape[0]
        paths = itertools.permutations(range(cities_number))

        best_path = next(paths)
        best_path_distance = get_trip_distance(self.distances_matrix, best_path)

        for path in paths:

            path_distance = get_trip_distance(self.distances_matrix, path)

            if path_distance < best_path_distance:

                best_path = path
                best_path_distance = path_distance

        return best_path



if __name__ == "__main__":

    cities_number = 5
    max_distance = 100

    distances_matrix = get_random_distances_matrix(cities_number, max_distance)

    start = time.time()
    optimal_path = BruteForceTSPSolver(distances_matrix).solve()
    print("[+] Optimal path is " + str(optimal_path))
    print("[+] Distance is " + str(get_trip_distance(distances_matrix, optimal_path)))
    print("[+] Computational time is: {0:.2f} seconds".format(time.time() - start))