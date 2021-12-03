import numpy as np
from queue import LifoQueue
import math

class SkeletalGraph:
    def __init__(self, structure):
        self.vertices = structure
        self.edges = self.compute_edges()
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
                # distance between 2 verticies is < 1.3 * block_size and not visited
                if self.block_size * 1.3 > math.sqrt((curr_vertex[0]-v[0])**2 \
                + (curr_vertex[1]-v[1])**2 + (curr_vertex[2]-v[2])**2) \
                and visited.get(curr_vertex, False) is False:
                    edges.append([curr_vertex, v])
                    stack.put(v)
            visited[curr_vertex] = True
        
        return edges

    def add_edge(self, edge):
        # add edges + vertex on an existing edge
        if edge not in self.edges:
            if edge[0] not in self.vertices:
                self.vertices.append(edge[0])
            if edge[1] not in self.vertices:
                self.vertices.append(edge[1])
            self.edges.append(edge)

    def add_edge(self, v1, v2):
        pass

    def split_edge(self, edge, v):
        # create new vertex and edges at coordinate of v
        pass

    def remove_edges(self, edge):
        # remove extra edges
        if edge in self.edges:
            self.edges.remove(edge)

    def get_intersecting_edges(self, x = None, y = None, z = None):
        # find edges that intersect with plane (x,y,z) that dont' have verticies
        
        pass

    def compute_slices_x(self, index):
        return self.compute_slices(0)

    def compute_slices_y(self, index):
        return self.compute_slices(1)

    def compute_slices_z(self, index):
        return self.compute_slices(2)

    # index from 0 to 2
    def compute_slices(self, index):
        verticies = self.vertices.copy()
        verticies.sort(key = lambda verticies: verticies[index])
        slices = []
        curr_level = verticies[0]
        curr_slice = [verticies[0]]
        for i in range(1, len(verticies)):
            if math.abs(verticies[i][index] - curr_level[index]) < self.block_size/5:
                curr_slice.append(verticies[i])
            else:
                slices.append(curr_slice)
                curr_level = verticies[i]
                curr_slice = [verticies[i]]
        return slices
        

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