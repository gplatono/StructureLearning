import numpy as np
from queue import LifoQueue
import math

class SkeletalGraph:
    def __init__(self, structure):
        self.vertices = structure
        self.compute_edges()
        self.block_size = 1

    def compute_edges(self):
        # Add edge if blocks are within a certain radius of target block.
        # Depth first searches through the set of vertices, adding close 
        # vertices as edges.
        edges = []
        visited = {}
        stack = LifoQueue()
        stack.put(self.vertices[0])

        while not stack.empty():
            curr_vertex = stack.get()
            for v in self.vertices:
                # diistance between 2 verticies is < 1.3 * block_size and not visited
                if self.block_size * 1.3 > math.sqrt((curr_vertex[0]-v[0])**2 \
                + (curr_vertex[0]-v[0])**2 + (curr_vertex[0]-v[0])**2) \
                and visited.get(curr_vertex, False) is False:
                    edges.append([curr_vertex, v])
                    stack.put(v)
            visited[curr_vertex] = True
        
        return edges

    def add_edge(self, edge):
        # add edges + vertex on an existing edge
        pass

    def remove_edges(self, edge):
        # remove vertex + extra edges
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