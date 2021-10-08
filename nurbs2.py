import bpy
import bmesh
import numpy as np
import math
from mathutils import Vector

def get_neighbors(obj, objects):
	dists = [(o, np.linalg.norm(obj.location - o.location)) for o in objects]
	dists.sort(key = lambda x: x[1])
	neighbors = [dists[1][0]]
	for pair in dists[2:]:
		if pair[1] < 1.3 * dists[1][1]:
			neighbors.append(pair[0])

	return neighbors

def get_pivots(objects):
	return [Vector((obj.location[0], obj.location[1], obj.location[2], 1.0)) for obj in objects]

def return_closest(obj, objects, dist_matrix):
	min_dist = 1e10
	closest_obj = None
	for obj_i in objects:
		if dist_matrix[obj][obj_i] < min_dist:
			min_dist = dist_matrix[obj][obj_i]
			closest_obj = obj_i

	return closest_obj

def order(objects):
	dist_matrix = {}
	for ob1 in objects:
		if ob1 not in dist_matrix:
			dist_matrix[ob1] = {}

		for ob2 in objects:
			if ob2 not in dist_matrix:
				dist_matrix[ob2] = {}

			dist_matrix[ob1][ob2] = np.linalg.norm(ob1.location - ob2.location)
			dist_matrix[ob2][ob1] = dist_matrix[ob1][ob2]

	endpoint = None
	for obj in objects:
		neighbors = get_neighbors(obj, objects)
		print (obj, neighbors)
		if len(neighbors) == 1:
			endpoint = obj
			break

	ordered = [endpoint]
	objects = set(objects)
	while len(objects) > 1:
		current = ordered[-1]
		objects.remove(current)
		ordered.append(return_closest(current, objects, dist_matrix))

	return ordered

# Get vertices and edges of the NURBS curve by converting it to mesh and back.
def get_verts_edges(nurbs_object, use_modifiers=True, settings='PREVIEW'):
    scene = bpy.context.scene
    # create a temporary mesh
    obj_data = nurbs_object.to_mesh(preserve_all_data_layers=True,depsgraph=bpy.context.evaluated_depsgraph_get())

	# ------- increase the number of subdivisions on the mesh
    bm = bmesh.new()
    bm.from_mesh(obj_data)

	# subdivide
    bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=2, use_grid_fill=True,)

	# Write back to the mesh
    bm.to_mesh(obj_data)
    obj_data.update()
	# -------

    verts = [v.co for v in obj_data.vertices]
    edges = obj_data.edge_keys
    for v in verts:
        print(v)
    # discard temporary mesh
    #bpy.data.meshes.remove(obj_data)
    return verts, edges

ordered = order(bpy.context.scene.objects)
points = get_pivots(ordered)

voxel_size = 0.25
box_size = voxel_size

# methods for finding voxel intersections with NURBS curve.

def find_line_x_plane_intersection(p1, p2, p3, l1, l2):
    """
    Find a point of intersection of a given plane defined by p1, p2, p3 and a line
    defined by l1, l2.
    Return the coordinates of the intersection point if it exists or None otherwise.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    l1 = np.array(l1)
    l2 = np.array(l2)
    p1 -= l1
    p2 -= l1
    p3 -= l1
    l2 -= l1
    n = np.cross(p2-p1, p3-p1)
    k = np.dot(p1, n) / np.dot(l2, n) if np.dot(l2, n) != 0 else 1e10
    if k < 1e10:
        return l1 + k*l2
    else:
        return None
def is_in_polygon(poly, p):
    """
    Check if a given point p lies inside a convex polygon poly.
    Assumes that the point and the polygon lie in the same plane.
    """
    poly = np.array(poly)
    p = np.array(p)
    npoly = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    k = np.cross(poly[0] - poly[-1], p - poly[-1])
    if np.dot(k, npoly) == 0:
        return True
    else:
        sign = 1 if np.dot(k, npoly) > 0 else -1
    for i in range(1, len(poly)):
        k1 = np.cross(poly[i] - poly[i-1], p - poly[i-1])
        prod = np.dot(npoly, k1)
        if prod == 0:
            return True
        elif prod * sign < 0:
            return False
    return True

def find_poly_intersection(poly, l1, l2):
    """
    Check if the line defined by l1, l2 passes inside
    a planar conver polygon poly.
    Return the coordinates of the point of intersection if exists,
    None otherwise.
    """
    s = find_line_x_plane_intersection(poly[0], poly[1], poly[2], l1, l2)
    if s is not None and is_in_polygon(poly, s):
        return s
    else:
        return None
        
def find_box_line_intersection(c, s, l1, l2):
    """
    Find points of intersection of a cube located at c
    with size (edge length) s, and the line defined by l1, l2
    """
    c = np.array(c)
    s /= 2
    l1 = np.array(l1)
    l2 = np.array(l2)
    faces = np.array([[[c[0] - s, c[1] - s, c[2] - s], [c[0] - s, c[1] + s, c[2] - s], [c[0] - s, c[1] + s, c[2] + s], [c[0] - s, c[1] - s, c[2] + s]], \
                      [[c[0] + s, c[1] - s, c[2] - s], [c[0] + s, c[1] + s, c[2] - s], [c[0] + s, c[1] + s, c[2] + s], [c[0] + s, c[1] - s, c[2] + s]], \
                      [[c[0] - s, c[1] - s, c[2] - s], [c[0] + s, c[1] - s, c[2] - s], [c[0] + s, c[1] - s, c[2] + s], [c[0] - s, c[1] - s, c[2] + s]], \
                      [[c[0] - s, c[1] + s, c[2] - s], [c[0] + s, c[1] + s, c[2] - s], [c[0] + s, c[1] + s, c[2] + s], [c[0] - s, c[1] + s, c[2] + s]], \
                      [[c[0] - s, c[1] - s, c[2] - s], [c[0] + s, c[1] - s, c[2] - s], [c[0] + s, c[1] + s, c[2] - s], [c[0] - s, c[1] + s, c[2] - s]], \
                      [[c[0] - s, c[1] - s, c[2] + s], [c[0] + s, c[1] - s, c[2] + s], [c[0] + s, c[1] + s, c[2] + s], [c[0] - s, c[1] + s, c[2] + s]]])
    intersections = []
    for face in faces:
        current = find_poly_intersection(face, l1, l2)
        flag = False
        if current is not None:
            for prev in intersections:
                if np.array_equal(current, prev):
                    flag = True
                    break
            if not flag:
                intersections.append(current)
    return intersections

def check_box_point_containment(box_center, box_size, point):
    offset = box_size / 2
    epsilon = 0.0001
    return point[0] >= box_center[0] - offset - epsilon and point[0] <= box_center[0] + offset + epsilon \
           and point[1] >= box_center[1] - offset - epsilon and point[1] <= box_center[1] + offset + epsilon \
           and point[2] >= box_center[2] - offset - epsilon and point[2] <= box_center[2] + offset + epsilon

def find_box_segment_intersection(c, s, l1, l2):
    """
    Find points of intersection of a cubical box located at c
    with size (edge length) s, and the segment defined by l1, l2
    """
    l1 = np.array(l1, dtype=np.single)
    l2 = np.array(l2, dtype=np.single)
    intersections = find_box_line_intersection(c, s, l1, l2)
    result = [p for p in intersections if check_box_point_containment(c, s, p) and max(np.linalg.norm(p - l1), np.linalg.norm(p - l2)) <= np.linalg.norm(l2 - l1) ]
    return result

# checks if p1 and p2 are points within the voxel centered at point c
def is_line_segment_in_box(c, p1, p2):
    return check_box_point_containment(c, box_size, p1) and check_box_point_containment(c, box_size, p2)

def is_point_in_box(point, voxel):
    return check_box_point_containment(voxel, voxel_size, point)
def find_voxels(vertices):
    prev = vertices[0]
    segments = [[]]
    voxels = [vertices[0]]
    seg_len = [0]
    idx = 0
    while idx < len(vertices):
        curr = vertices[idx]
        #Go on while curr is still inside the last voxel
        if is_point_in_box(curr, voxels[-1]):
            segments[-1].append(curr)
            seg_len[-1] += np.linalg.norm(np.array(curr) - np.array(prev))
            prev = curr
            idx += 1
        #Find the next voxel (probably neighbor)
        else:
            next_v = get_next_voxel(voxels[-1], curr, voxel_size)
            voxels.append(next_v)
            intersect = find_box_segment_intersection(next_v, voxel_size, prev, curr)
            #print (intersect, next_v, prev, curr)
            intersect = intersect[0]
            # if intersect == []:
            #     print (voxels[-2], prev, next_v, curr)
            # else:
            #     intersect = intersect[0]
            seg_len[-1] += np.linalg.norm(intersect - prev)
            prev = intersect
            seg_len.append(0)
            segments.append([])
    return voxels, segments, seg_len

def get_next_voxel(prev, point, size):
    size /= 2
    x_offset = (point[0] - prev[0]) / size
    y_offset = (point[1] - prev[1]) / size
    z_offset = (point[2] - prev[2]) / size
    #print (x_offset, y_offset, z_offset)
    x_offset = math.ceil(x_offset) if x_offset >= 0 else math.floor(x_offset)
    y_offset = math.ceil(y_offset) if y_offset >= 0 else math.floor(y_offset)
    z_offset = math.ceil(z_offset) if z_offset >= 0 else math.floor(z_offset)
    x_offset = math.floor(x_offset / 2) if x_offset / 2 >= 0 else math.ceil(x_offset / 2)
    y_offset = math.floor(y_offset / 2) if y_offset / 2 >= 0 else math.ceil(y_offset / 2)
    z_offset = math.floor(z_offset / 2) if z_offset / 2 >= 0 else math.ceil(z_offset / 2)
    next_v = [prev[0] + x_offset * voxel_size, prev[1] + y_offset * voxel_size, prev[2] + z_offset * voxel_size]
    #print (prev, next_v, point, x_offset, y_offset, z_offset)
    return next_v

surface_data = bpy.data.curves.new('wook', 'SURFACE')
surface_data.dimensions = '3D'
spline = surface_data.splines.new(type='NURBS')
spline.points.add(len(points))  #   already has a a default zero vector

for p, new_co in zip(spline.points, points):
    p.co = new_co

spline.order_v = 4
spline.order_u = 4
spline.resolution_v= 4
spline.resolution_u = 4
spline.use_endpoint_u = True

surf = bpy.data.objects.new('NURBS_OBJ', surface_data)
bpy.context.collection.objects.link(surf)

# calling methods to get vertices and edges from NURBS

verts, edges = get_verts_edges(surf)

# calling methods to get voxel intersections
vertices = []
for v in verts:
	vertices.append(list(v))

voxels, segments, lengths = find_voxels(vertices)

# creating mesh in blender
for v in voxels:
	bpy.ops.mesh.primitive_cube_add(size = voxel_size, location=(v[0],v[1],v[2]))
	print("voxel", "[", v[0], ", ", v[1], ", ",v[2], "], ")

bpy.context.evaluated_depsgraph_get().update()
