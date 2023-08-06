import gdspy
import math
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import time
import csv
import cv2



from numba import jit, njit
import numba 

@jit(nopython=True)
def pointinpolygon(xy,poly):
	x, y = xy
	n = len(poly)
	inside = False
	p2x = 0.0
	p2y = 0.0
	xints = 0.0
	p1x,p1y = poly[0]
	for i in numba.prange(n+1):
		p2x,p2y = poly[i % n]
		if y > min(p1y,p2y):
			if y <= max(p1y,p2y):
				if x <= max(p1x,p2x):
					if p1y != p2y:
						xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
					if p1x == p2x or x <= xints:
						inside = not inside
		p1x,p1y = p2x,p2y

	return inside

nm = 1e-3
d2r = np.pi/180
n_HSQ = 1.43

def correct(s):
	if s[-3:] == "*^6":
		return nm*float(s[:-3])*1e6
	else:
		return nm*float(s)

def write_from_file(cell, file, shift_149=False, shift_y=False, global_x_offet = 0*nm, global_y_offet = 0*nm, fix_double_write=False):
	polygons = []
	with open(file) as f:
		for idx, line in enumerate(f):
			data = line.split('\t')
			rotation = 0

			if len(data) == 4:
				x_c = correct(data[0])
				y_c = correct(data[1])
				x_s = correct(data[2])
				y_s = correct(data[3])
			elif len(data) == 5:
				x_c = correct(data[0])
				y_c = correct(data[1])
				x_s = correct(data[2])
				y_s = correct(data[3])
				rotation = float(data[-1])
			else:
				raise ValueError(str(data))

			if shift_y:
				y_c += 4000*nm
			elif shift_149:
				if x_c < 10000*nm:
					y_c += 4000*nm
			
			if x_s > 0 and y_s > 0:
				ellipse = gdspy.Round((global_x_offet+x_c, global_y_offet+y_c), [x_s,y_s], tolerance=0.001)
				if rotation != 0:
					ellipse = ellipse.rotate(rotation, center=(global_x_offet+x_c, global_y_offet+y_c))

				if fix_double_write:
					polygons.append(ellipse)
				else:
					cell.add(ellipse)

	if fix_double_write:
		if len(polygons) >0:
			merged = gdspy.boolean(polygons, None, 'or')
			cell.add(merged)

def make_alignment_mark(cell, r,c, x_bl, y_bl, cross_only=False):
	# r: row number, c: column number
	#	a3 a2 X  b3 b2
	#	a1 a0 X  b1 b0
	#   X  X  X  X  X
	#	c3 c2 X  d3 d2  
	#	c1 c0 X  d1 d0
	# a: higher bits of r, b: lower bits of r
	# c: higher bits of c, d: lower bits of c
	cross_pts = [(x_bl+200*nm, y_bl+0),      (x_bl+200*nm, y_bl+200*nm), (x_bl, 		y_bl+200*nm), (x_bl, 		y_bl+300*nm), (x_bl+200*nm, y_bl+300*nm), (x_bl+200*nm, y_bl+500*nm),\
				 (x_bl+300*nm, y_bl+500*nm), (x_bl+300*nm, y_bl+300*nm), (x_bl+500*nm,  y_bl+300*nm), (x_bl+500*nm, y_bl+200*nm), (x_bl+300*nm, y_bl+200*nm), (x_bl+300*nm, y_bl+0)]
	cross = gdspy.Polygon(cross_pts)
	
	if cross_only:
		cell.add(cross)
	else:
		polygon_list = []
		row_str = format(r, '#010b')
		if row_str[-1] == '1':
			polygon_list.append([(x_bl+400*nm, y_bl+300*nm),   (x_bl+400*nm, y_bl+400*nm),  (x_bl+500*nm, y_bl+400*nm), (x_bl+500*nm, y_bl+300*nm)])
		if row_str[-2] == '1':
			polygon_list.append([(x_bl+300*nm, y_bl+300*nm),   (x_bl+300*nm, y_bl+400*nm),  (x_bl+400*nm, y_bl+400*nm), (x_bl+400*nm, y_bl+300*nm)])
		if row_str[-3] == '1':
			polygon_list.append([(x_bl+400*nm, y_bl+400*nm),   (x_bl+400*nm, y_bl+500*nm),  (x_bl+500*nm, y_bl+500*nm), (x_bl+500*nm, y_bl+400*nm)])
		if row_str[-4] == '1':
			polygon_list.append([(x_bl+300*nm, y_bl+400*nm),   (x_bl+300*nm, y_bl+500*nm),  (x_bl+400*nm, y_bl+500*nm), (x_bl+400*nm, y_bl+400*nm)])

		if row_str[-5] == '1':
			polygon_list.append([(x_bl+100*nm, y_bl+300*nm),   (x_bl+100*nm, y_bl+400*nm),  (x_bl+200*nm, y_bl+400*nm), (x_bl+200*nm, y_bl+300*nm)])
		if row_str[-6] == '1':
			polygon_list.append([(x_bl       , y_bl+300*nm),   (x_bl       , y_bl+400*nm),  (x_bl+100*nm, y_bl+400*nm), (x_bl+100*nm, y_bl+300*nm)])
		if row_str[-7] == '1':
			polygon_list.append([(x_bl+100*nm, y_bl+400*nm),   (x_bl+100*nm, y_bl+500*nm),  (x_bl+200*nm, y_bl+500*nm), (x_bl+200*nm, y_bl+400*nm)])
		if row_str[-8] == '1':
			polygon_list.append([(x_bl       , y_bl+400*nm),   (x_bl       , y_bl+500*nm),  (x_bl+100*nm, y_bl+500*nm), (x_bl+100*nm, y_bl+400*nm)])
		
		col_str = format(c, '#010b')
		if col_str[-1] == '1':
			polygon_list.append([(x_bl+400*nm, y_bl    ),   	(x_bl+400*nm, y_bl+100*nm),  (x_bl+500*nm, y_bl+100*nm), (x_bl+500*nm, y_bl    )])
		if col_str[-2] == '1':
			polygon_list.append([(x_bl+300*nm, y_bl    ),   	(x_bl+300*nm, y_bl+100*nm),  (x_bl+400*nm, y_bl+100*nm), (x_bl+400*nm, y_bl    )])
		if col_str[-3] == '1':
			polygon_list.append([(x_bl+400*nm, y_bl+100*nm),   	(x_bl+400*nm, y_bl+200*nm),  (x_bl+500*nm, y_bl+200*nm), (x_bl+500*nm, y_bl+100*nm)])
		if col_str[-4] == '1':
			polygon_list.append([(x_bl+300*nm, y_bl+100*nm),   	(x_bl+300*nm, y_bl+200*nm),  (x_bl+400*nm, y_bl+200*nm), (x_bl+400*nm, y_bl+100*nm)])

		if col_str[-5] == '1':
			polygon_list.append([(x_bl+100*nm, y_bl    ),    	(x_bl+100*nm, y_bl+100*nm),  (x_bl+200*nm, y_bl+100*nm), (x_bl+200*nm, y_bl    )])
		if col_str[-6] == '1':
			polygon_list.append([(x_bl       , y_bl    ),   	(x_bl    	, y_bl+100*nm),  (x_bl+100*nm, y_bl+100*nm), (x_bl+100*nm, y_bl    )])
		if col_str[-7] == '1':
			polygon_list.append([(x_bl+100*nm, y_bl+100*nm),   	(x_bl+100*nm, y_bl+200*nm),  (x_bl+200*nm, y_bl+200*nm), (x_bl+200*nm, y_bl+100*nm)])
		if col_str[-8] == '1':
			polygon_list.append([(x_bl       , y_bl+100*nm),   	(x_bl    	, y_bl+200*nm),  (x_bl+100*nm, y_bl+200*nm), (x_bl+100*nm, y_bl+100*nm)])

		all_boxes = gdspy.PolygonSet(polygon_list)
		merged = gdspy.boolean(cross,all_boxes, 'or')
		cell.add(merged)


def write_alignment_marks(cell, Nx_Start, Nx_end, Ny_start, Ny_end, x0=0*nm, y0=0*nm):
	for r in range(Ny_start, Ny_end+1):
		for c in range(Nx_Start, Nx_end+1):
			x_bl = x0+c*8000*nm
			y_bl = y0+r*8000*nm

			if r in [0,125-1] and c in [0,125-1]:
				make_alignment_mark(cell, r, c, x_bl, y_bl, cross_only=True)
			else:
				make_alignment_mark(cell, r, c, x_bl, y_bl, cross_only=False)

def rc2files(r,c):
	# we have 496 um by 496 um region of shapes divided to 300 files.
	num_per_file = 62*62/296
	block_num = 31*c+r if r<31 else 31*62+31*c+r-30
	
	this_block_file = int(block_num//num_per_file)

	if r==31:
		bottom_block_file = int((31*c+31)//num_per_file)
	elif r==0:
		# no bottom, but use this_block here:
		bottom_block_file = this_block_file
	else:
		bottom_block_file = int((block_num-1)//num_per_file)
	
	if r==30:
		top_block_file = int((31*62+31*c+1)//num_per_file)
	elif r==61:
		top_block_file = this_block_file
	else:
		top_block_file = int((block_num+1)//num_per_file)


	if c==0:
		left_block_file = this_block_file
	else:
		left_block_file = int((block_num-31)//num_per_file)

	if c==61:
		right_block_file = this_block_file
	else:
		right_block_file = int((block_num+31)//num_per_file)

	# print(block_num, this_block_file+1,top_block_file+1,bottom_block_file+1,left_block_file+1,right_block_file+1)
	# +1 since the files starts with index 1
	return list(set([this_block_file+1,top_block_file+1,bottom_block_file+1,left_block_file+1,right_block_file+1]))

def gen_uint8_map(r, c, margin, resolution=4*nm, path='500um_txt/Design_sample', x0=50*nm, y0=50*nm):
	# generate local map with bottom left alignment mark at (r, c), with a margin

	# The GDSII file is called a library, which contains multiple cells.
	lib = gdspy.GdsLibrary()

	# Geometry must be placed in cells.
	cell = lib.new_cell('FIRST')
	
	g_x_off = 0 if c<62 else 496 # in um
	g_y_off = 0 if r<62 else 496 # in um

	Nx_start = c
	Nx_end = c+1
	Ny_start = r
	Ny_end = r+1

	file_indices = rc2files(r%62,c%62)

	for i in file_indices:
		if i>300:
			gdspy.current_library.remove('FIRST')
			return None,True
		if i==149:
			write_from_file(cell, path+'Design_sample_'+str(i)+'.txt', shift_149=True, global_x_offet = g_x_off, global_y_offet = g_y_off, fix_double_write=True)
		elif i>=150:
			write_from_file(cell, path+'Design_sample_'+str(i)+'.txt', shift_y=True, global_x_offet = g_x_off, global_y_offet = g_y_off, fix_double_write=True)
		else:
			write_from_file(cell, path+'Design_sample_'+str(i)+'.txt', global_x_offet = g_x_off, global_y_offet = g_y_off)

	write_alignment_marks(cell, Nx_start, Nx_end, Ny_start, Ny_end, x0=x0, y0=y0)

	# write a uint8 map:
	size = int((margin+margin+8)/resolution)
	local_map = np.zeros((size,size))

	x_min = x0 + c*8000*nm - margin
	x_max = x0 + (c+1)*8000*nm + margin
	y_min = y0 + r*8000*nm - margin
	y_max = y0 + (r+1)*8000*nm + margin
	
	# print("x_min,x_max,y_min,y_max: ", x_min,x_max,y_min,y_max)
	# print("len(cell.polygons): ", len(cell.polygons))
	for p_set in cell.polygons:
		# print("len(p_set.polygons): ", len(p_set.polygons))
		for p_numpy in p_set.polygons:
			p = gdspy.Polygon(p_numpy)
			bbox = p.get_bounding_box()
			# # print("bbox: ", bbox)
			if not(bbox[0][0]>x_max or bbox[1][0]<x_min or bbox[0][1]>y_max or bbox[1,1]<y_min):
				# get the local indices for that shape:
				x_idx_min = int((bbox[0][0] - x_min)/resolution)
				x_idx_max = int((bbox[1][0] - x_min)/resolution)
				y_idx_min = int((y_max - bbox[1][1])/resolution)
				y_idx_max = int((y_max - bbox[0][1])/resolution)
				for x in range(x_idx_min, x_idx_max+1):
					for y in range(y_idx_min, y_idx_max+1):
						thisx = x_min + x*resolution
						thisy = y_max - y*resolution

						if pointinpolygon((thisx, thisy), p_numpy):
						# if gdspy.inside([(thisx, thisy)], [p]):
							if x<size and y<size:
								local_map[x,y] = 255

	# for i in tqdm(range(size)):
	# 	for j in range(size):
	# 		x = x0 + c*8000*nm - margin + j * resolution
	# 		y = y0 + (r+1)*8000*nm + margin - i * resolution
	# 		if gdspy.inside([(x,y)], cell.polygons):
	# 			local_map[i,j] = 1
	gdspy.current_library.remove('FIRST')
	return local_map,False
	# file_name = "test_rc_local_map"
	# lib.write_gds(file_name+'.gds')
	# cell.write_svg('500um_by_500um.svg')

if __name__ == '__main__':
	start = time.time()

	r = 47
	c = 0
	margin = 1 # leave 4 um of margin 
	resolution=4*nm
	local_map = gen_uint8_map(r,c, margin, resolution=resolution, x0=50*nm, y0=50*nm)
	cv2.imwrite(f'local_map_r{r}_c{c}.png', local_map)

	end = time.time()
	# print("Run time: ",end - start, ' s')

