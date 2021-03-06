import numpy as np
import os
import sys
import bpy
filepath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, filepath)
core_path = os.path.normpath("../SpatRel")
sys.path.insert(0, core_path)
from geometry_utils import is_in_polygon, find_closest_point_of_polygon


class Stabilizer:
	def __init__(self, block_locations, size):
		self.block_locations = block_locations
		#sort according to the z-coordinate
		self.block_locations.sort(reverse=True, key=lambda x: x[2])

		self.block_size = size
		self.distances = {}
		self.adjacent = {}
		self.distance_matrix = {}
		self.support_hierarchy = {}
		self.neighbors = {}


		for block in self.block_locations:
			#Fill-in the matrix of pairwise block distances
			if block not in self.distance_matrix:
				self.distance_matrix[block] = {}
			for bl in self.block_locations:

				dist = np.linalg.norm(np.array(block) - np.array(bl))
				if bl not in self.distance_matrix:
					self.distance_matrix[bl] = {}

				self.distance_matrix[block][bl] = dist
				self.distance_matrix[bl][block] = dist

			#Compute sorted distances for a given block
			self.distances[block] = [(idx, np.linalg.norm(np.array(block) - np.array(self.block_locations[idx]))) for idx in range(len(self.block_locations))]
			self.distances[block].sort(key=lambda x: x[1])

			#Fill-in neighbors (close blocks) for a given block
			self.neighbors[block] = [self.block_locations[item[0]] for item in self.distances[block] if item[1] <= 1.5 * size]

		#For testing/debugging
		#sa = self.support_area(self.block_locations[0])
		#print (sa)
		#print(self.direct_supporters(self.block_locations[0]), self.supporters(self.block_locations[0]), self.supportees(self.block_locations[-1]))

	def support_area(self, blocks):
		"""
		Computes rectangular support area for a given block.
		The support area is the minimal rectangle that encircles all direct supporters.
		"""
		direct_supporters = []
		if not isinstance(blocks[0], list) and not isinstance(blocks[0], tuple):			
			direct_supporters = self.direct_supporters(blocks)
		else:
			for bl in blocks:
				direct_supporters += self.direct_supporters(bl)
		#print (direct_supporters)
		x_min = 1e10
		x_max = -1e10
		y_min = 1e10
		y_max = -1e10
		for bl in direct_supporters:
			x_min = min(x_min, bl[0] - self.block_size/2)
			x_max = max(x_max, bl[0] + self.block_size/2)
			y_min = min(y_min, bl[1] - self.block_size/2)
			y_max = max(y_max, bl[1] + self.block_size/2)

		return [x_min, x_max, y_min, y_max]


	def support_poly(self, block):
		sup_area = self.support_area(block)
		#print (block, sup_area)
		return [[sup_area[0], sup_area[2], 0], [sup_area[0], sup_area[3], 0],
				[sup_area[1], sup_area[3], 0], [sup_area[1], sup_area[2], 0]]

	def check_shadow_overlap(self, bl1, bl2):
		"""
		Check whether block bl2 is in the shadow of the block bl1, i.e., overlaps it bl1's projection onto
		the xy-plane.
		Returns a boolean value reflecting whether the overlap occurs.
		"""

		#If bl2 is higher or at the same height as bl1, it cannot overlap with its shadow
		if bl2[2] >= bl1[2]:
			return False

		x_overlap = False
		y_overlap = False
		#Checks if there is an overlap along the x-axis
		if bl2[0] + self.block_size >= bl1[0] - self.block_size and bl2[0] + self.block_size <= bl1[0] + self.block_size \
			or bl2[0] - self.block_size >= bl1[0] - self.block_size and bl2[0] - self.block_size <= bl1[0] + self.block_size:
			x_overlap = True

		#Checks if there is an overlap along the y-axis
		if bl2[1] + self.block_size >= bl1[1] - self.block_size and bl2[1] + self.block_size <= bl1[1] + self.block_size \
			or bl2[1] - self.block_size >= bl1[1] - self.block_size and bl2[1] - self.block_size <= bl1[1] + self.block_size:
			y_overlap = True

		#bl2 overlaps the shadow of bl1 if and only if it overlaps it along both axes
		return x_overlap and y_overlap

	def compute_shadow_overlap(self, block, candidates):
		"""
		Given a list of candidate blocks, find all candidates that overlap the shadow of the given block.
		Returns the sublist of candidates that do overlap.
		"""
		#print ("SHADOW: ", block, candidates)
		ret_val = []
		for bl in candidates:
			#print (block, bl, self.check_shadow_overlap(block, bl))
			if self.check_shadow_overlap(block, bl):
				ret_val.append(bl)

		return ret_val

	def direct_supporters(self, block):
		"""
		Finds all the direct (vertical) supporters of the block
		"""

		#Supporters must overlap the shadow and be neighbors
		ret_val = []
		in_the_shadow = self.compute_shadow_overlap(block, self.block_locations)
		#print (block, in_the_shadow)
		for bl in in_the_shadow:
			if bl in self.neighbors[block]:
				ret_val.append(bl)

		return ret_val

	def supporters(self, block):
		supporters = []
		queue = self.direct_supporters(block)
		while len(queue) > 0:
			supp = queue.pop(0)
			supporters.append(supp)
			queue += self.direct_supporters(supp)

		return supporters

	def supportees(self, block):
		"""
		Finds all the supportees of the block.
		"""

		#If A is a supportee of B, then B is a supporter of A
		ret_val = []
		for bl in self.block_locations:
			if block in self.supporters(bl):
				ret_val.append(bl)

		return ret_val

	def stabilize(self, blocks=None):
		"""
		The main method for the stability checking/enforcing algorithm.
		"""
		if blocks is None:
			blocks = self.block_locations


		for bl in blocks:
			Z = {bl}
			supporters = self.supporters(bl)
			supportees = self.supportees(bl)
			X = {bl}.union(supporters, supportees)

			for s in supportees:
				sup_set = set(self.supporters(s))
				#print (s, sup_set, X, Z)
				if sup_set.issubset(X):
					Z.add(s)			
			vpmc = self.VPMC(Z)			
			supB = self.support_poly(bl)
			print ("bl: ", bl)
			print ("Z: ", Z)
			print ("VPMC: ", vpmc, supB, self.is_VPMC_in_sup(vpmc, bl))			
			if not self.is_VPMC_in_sup(vpmc, bl):
				#print (vpmc, supB)
				p = find_closest_point_of_polygon(vpmc, supB)
				#A = supB
				print ("p: ", p)
				print ("SUPP: ", supportees)
				print ()
				for N in supportees:
					if N not in Z and self.overlaps(N, supB):
						Z.add(N)
				supZ = self.support_area(list(Z))
				if not self.is_VPMC_in_sup(vpmc, supZ):
					vpmc1 = self.VMPC(Z)
					return False
		return True

	def overlaps(self, block, area):        
		overlap = is_in_polygon(area, (block[0] - self.size / 2, block[1] - self.size / 2, 0)) or \
		is_in_polygon(area, (block[0] - self.size / 2, block[1] + self.size / 2, 0)) or \
		is_in_polygon(area, (block[0] + self.size / 2, block[1] - self.size / 2, 0)) or \
		is_in_polygon(area, (block[0] + self.size / 2, block[1] + self.size / 2, 0))
		return overlap

	def VPMC(self, blocks):		
		vpmc = np.average(list(blocks), axis=0)
		vpmc[2] = 0
		return vpmc

	def is_VPMC_in_sup(self, VPMC, block):
		poly = self.support_poly(block)
		return is_in_polygon(poly, VPMC)

	def is_level_neighbour(self, block, neighbour):
		# if block and neighbour are on same axis
		on_same_height = np.absolute(block[2] - neighbour[2]) < 0.1 * self.block_size
		poly = [[blocks[0] + block_size/2, blocks[1] + block_size/2, blocks[2] - block_size/2],
		[blocks[0] - block_size/2, blocks[1] + block_size/2, blocks[2] - block_size/2],
		[blocks[0] + block_size/2, blocks[1] - block_size/2, blocks[2] - block_size/2],
		[blocks[0] - block_size/2, blocks[1] - block_size/2, blocks[2] - block_size/2]]
		supp = self.support_area(block)
		
		block_supported = is_in_polygon(poly, self.VPMC(block))

		if not block_supported and on_same_height:
			# if block and neighbour are within 90% to 110% of block_size of
			# each other in the x (or y axis), we consider it a neighbour
			if (np.absolute(block[0] - neighbour[0]) > 0.9 * self.block_size
			and np.absolute(block[0] - neighbour[0]) < 1.1 * self.block_size) or \
			(np.absolute(block[1] - neighbour[1]) > 0.9 * self.block_size
			and np.absolute(block[1] - neighbour[1]) < 1.1 * self.block_size):
				return True
			# elif (numpy.absolute(block[1] - neighbour[1]) > 0.9 * self.block_size
			# and numpy.absolute(block[1] - neighbour[1]) < 1.1 * self.block_size):
			#     return True
		return False

	# def supportees(self, block):
	# """
	# Finds all the supportees of the block.
	# """

	# #If A is a supportee of B, then B is a supporter of A
	# ret_val = []
	# for bl in self.block_locations:
	#     if block in self.supporters(bl):
	#         ret_val.append(bl)
	#     else if is_neighbour(block, bl):
	#         ret_val.append(bl)

	# return ret_val

objects = bpy.context.scene.objects
blocks = [tuple(obj.location) for obj in objects]
print (blocks)
#stab = Stabilizer([(0, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, 2)], 1)
stab = Stabilizer(blocks, 1)
print (stab.stabilize())