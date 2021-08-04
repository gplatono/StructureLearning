import bpy
import torch
import numpy as np

objects = bpy.context.scene.objects

def get_centroids():
	return [np.array(obj.location) for obj in objects]

#print (get_centroids())

def perturb(struct, size):
	perturbed = []
	amplitude = size / 5
	for point in struct:
		perturb = np.random.multivariate_normal([0,0,0], [[amplitude, 0, 0],[0, amplitude, 0], [0, 0, amplitude]], size=1)
		perturbed.append(point + perturb[0])
	# print (perturbed)
	return perturbed

def generate_stack(height):
	struct = []
	#max_height = 10
	size = 1
	z = size/2
	for i in range(height):
		struct.append((0,0,z))		
		z += size

	return perturb(struct, size)

def save_struct(struct):
	#print (struct)
	with open('struct_data', 'a') as data:
		for point in struct:
			#data.write([point for point in struct])
			data.write(str(point))
		data.write("\n")

def render(struct):
	for point in struct:
		bpy.ops.mesh.primitive_cube_add(location=point)

def generate_stacks(num):
	max_height = 10
	stacks = []
	for i in range(num):
		height = np.random.randint(2, max_height)
		stack = generate_stack(height)
		stacks.append(stack)
		save_struct(stack)
	
	render(stacks[-1])

generate_stacks(100)