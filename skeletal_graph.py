import numpy as np


class SkeletalGraph:
    def __init__(self, structure):
        self.vertices = structure
        self.compute_edges()

    def compute_edges(self):
        pass

    def add_edge(self, edge):
        pass

    def remove_edges(self, edge):
        pass

    def get_intersecting_edges(self, x = None, y = None, z = None):
        pass

    def compute_slices_x(self):
        pass
    
    def compute_slices_y(self):
        pass
    
    def compute_slices_z(self):
        pass

    def get_closest_vertex(self, v, vertices):
        """
        Find a vertex in vertices closest to v.
        """
        v0 = None
        min_dist = 10**10
        for v1 in vertices:
            dist = np.linalg.norm(v1 - v):
            if dist < min_dist:
                min_dist = dist
                v0 = v1

        return v0

    def compute_slice_similarity(self, this_slice, that_slice):
        dist = 0
        for v in this_slice:
            v1 = self.get_closest_vertex(v, that_slice)
            dist += np.linalg.norm(v - v1)
        return dist