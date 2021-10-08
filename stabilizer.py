import numpy as np
class Stabilizer:
    def __init__(self, block_locations, size):
        self.block_locations = block_locations
        self.block_locations.sort(reverse=True, key=lambda x: x[2])
        self.block_size = size
        self.distances = {}
        self.adjacent = {}
        self.distance_matrix = {}
        self.support_hierarchy = {}
        for block in self.block_locations:
            if block not in self.distance_matrix:
                self.distance_matrix[block] = {}
            for bl in self.block_locations:
                dist = np.linalg.norm(block - bl)
                if bl not in self.distance_matrix:
                    self.distance_matrix[bl] = {}
                
                self.distance_matrix[block][bl] = dist
                self.distance_matrix[bl][block] = dist

            self.distances[block] = [(idx, np.linalg.norm(block - self.block_locations[idx])) for idx in range(len(self.block_locations))]
            self.distances[block].sort(key=lambda x: x[1])
            self.neighbors[block] = [self.block_locations[item[0]] for item in self.distances[block] if item[1] <= 1.5 * size]
                        
        pass

    def check_shadow_overlap(self, bl1, bl2):
        if bl2[2] >= bl[1]:
            return False
        
        x_overlap = False
        y_overlap = False
        if bl2[0] + self.block_size >= bl1[0] - self.block_size and bl2[0] + self.block_size <= bl1[0] + self.block_size \
            or bl2[0] - self.block_size >= bl1[0] - self.block_size and bl2[0] - self.block_size <= bl1[0] + self.block_size:
            x_overlap = True

        if bl2[1] + self.block_size >= bl1[1] - self.block_size and bl2[1] + self.block_size <= bl1[1] + self.block_size \
            or bl2[1] - self.block_size >= bl1[1] - self.block_size and bl2[1] - self.block_size <= bl1[1] + self.block_size:
            y_overlap = True

        return x_overlap and y_overlap

    def compute_shadow_overlap(self, block, candidates):
        ret_val = []
        for bl in candidates:
            if self.check_shadow_overlap(block, bl):
                ret_val.append(bl)

        return ret_val

    def supporters(self, block):
        ret_val = []
        in_the_shadow = self.compute_shadow_overlap(block, self.block_locations)
        for bl in in_the_shadow:
            if bl in self.neighbors[block]:
                ret_val.append(bl)

        return ret_val

    def supportees(self, block):
        ret_val = []
        for bl in self.block_locations:
            if block in self.supporters(bl):
                ret_val.append(bl)

        return ret_val