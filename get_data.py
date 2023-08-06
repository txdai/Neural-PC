import gdspy
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import cv2
import scipy
import os
import time
import itertools

from samsung_local_map import gen_uint8_map

nm = 1e-3

def decode_one_alignment(align_code):
	results = set()

    # Create a list of all points to be considered
	points = [[(1, 0), (0, 1), (1, 1)], [(0, 3), (1, 4), (1, 3)], [(3, 0), (4, 1), (3, 1)], [(3, 4), (4, 3), (3, 3)]]

	valid_points = [point_pair for point_pair in points if align_code[point_pair[0]] == 1 and align_code[point_pair[1]] == 1]

	row = (
		align_code[1, 4]
		+ 2 * align_code[1, 3]
		+ 4 * align_code[0, 4]
		+ 8 * align_code[0, 3]
		+ 16 * align_code[1, 1]
		+ 32 * align_code[1, 0]
		+ 64 * align_code[0, 1]
		+ 128 * align_code[0, 0]
	)
	col = (
		align_code[4, 4]
		+ 2 * align_code[4, 3]
		+ 4 * align_code[3, 4]
		+ 8 * align_code[3, 3]
		+ 16 * align_code[4, 1]
		+ 32 * align_code[4, 0]
		+ 64 * align_code[3, 1]
		+ 128 * align_code[3, 0]
	)
	results.add((int(row), int(col)))

    # Generate all combinations of third elements for valid points
	for valid_combination in itertools.product(*[[0, 1] for _ in valid_points]):
		for i, point_pair in enumerate(valid_points):
			align_code[point_pair[2]] = valid_combination[i]
			row = (
				align_code[1, 4]
				+ 2 * align_code[1, 3]
				+ 4 * align_code[0, 4]
				+ 8 * align_code[0, 3]
				+ 16 * align_code[1, 1]
				+ 32 * align_code[1, 0]
				+ 64 * align_code[0, 1]
				+ 128 * align_code[0, 0]
			)
			col = (
				align_code[4, 4]
				+ 2 * align_code[4, 3]
				+ 4 * align_code[3, 4]
				+ 8 * align_code[3, 3]
				+ 16 * align_code[4, 1]
				+ 32 * align_code[4, 0]
				+ 64 * align_code[3, 1]
				+ 128 * align_code[3, 0]
			)
			results.add((int(row), int(col)))

	if align_code[2, :].all() != 1 or align_code[:, 2].all() != 1:
		return list(results), False
	else:
		return list(results), True


def decode_four_alignments(align_codes):
    # Decode all possible alignment results for each corner
	tl_results, tl_valid = decode_one_alignment(align_codes[0])
	tr_results, tr_valid = decode_one_alignment(align_codes[1])
	bl_results, bl_valid = decode_one_alignment(align_codes[2])
	br_results, br_valid = decode_one_alignment(align_codes[3])

	valid_results = set()

    # Iterate over all possible combinations of alignment results
	for tl, tr, bl, br in itertools.product(tl_results, tr_results, bl_results, br_results):
        # Check the conditions for each combination
		if not (tl_valid and tr_valid and bl_valid and br_valid):
			if tl_valid:
				valid_results.add((tl[0] - 1, tl[1]))
			if tr_valid:
				valid_results.add((tr[0] - 1, tr[1] - 1))
			if bl_valid:
				valid_results.add((bl[0], bl[1]))
			if br_valid:
				valid_results.add((br[0], br[1] - 1))
		else:
			if tl[0] != tr[0] or bl[0] != br[0]:
				continue
			if tl[1] != bl[1] or tr[1] != br[1]:
				continue
			if tl[0] != bl[0] + 1:
				continue
			if tl[1] + 1 != tr[1]:
				continue

			valid_results.add((bl[0], bl[1]))

	return list(valid_results)

def get_alignment_info(img, search_range=150, debug_plot=False):
	######## to save time, first downscale the image:
	compress = cv2.pyrDown(cv2.pyrDown(cv2.pyrDown(img)))
	if debug_plot:
		cv2.imwrite('compress.png', compress)

	######## from the binary map get the alignment marks:
	cross = np.zeros((15,15), dtype=np.uint8)
	cross[7-1:7+2, :] = 1
	cross[:, 7-1:7+2] = 1
	convolved = scipy.signal.convolve2d(compress.astype(np.float32), cross, mode='same')

	convolved = convolved/np.max(convolved)*255
	convolved = convolved.astype(np.uint8)
	if debug_plot:
		cv2.imwrite('final_convolved.png', convolved)

	######## get largest four regions:
	alignment_coords = []

	top1 = np.unravel_index(convolved.argmax(), convolved.shape)
	alignment_coords.append(top1)
	convolved[max(top1[0]-5,0):min(top1[0]+5, convolved.shape[1]), max(top1[1]-5,0):min(top1[1]+5, convolved.shape[1])] = 0

	top2 = np.unravel_index(convolved.argmax(), convolved.shape)
	alignment_coords.append(top2)
	convolved[max(top2[0]-5,0):min(top2[0]+5, convolved.shape[1]), max(top2[1]-5,0):min(top2[1]+5, convolved.shape[1])] = 0

	top3 = np.unravel_index(convolved.argmax(), convolved.shape)
	alignment_coords.append(top3)
	convolved[max(top3[0]-5,0):min(top3[0]+5, convolved.shape[1]), max(top3[1]-5,0):min(top3[1]+5, convolved.shape[1])] = 0

	top4 = np.unravel_index(convolved.argmax(), convolved.shape)
	alignment_coords.append(top4)
	convolved[max(top4[0]-5,0):min(top4[0]+5, convolved.shape[1]), max(top4[1]-5,0):min(top4[1]+5, convolved.shape[1])] = 0

	if debug_plot:
		color = cv2.cvtColor(convolved, cv2.COLOR_GRAY2BGR)
		for i in alignment_coords:
			cv2.circle(color, (i[1], i[0]), 3, (0,255,0), 2)
		cv2.imwrite('compress_annotate.png', color)

	# convert the corrdinates back to the original precision:
	alignment_coords_original = [(i[0]*8, i[1]*8) for i in alignment_coords]

	# sort coords so it is [tl, tr, bl, br]:
	alignment_coords_original.sort(key = lambda x: 5*x[0]+x[1])

	if debug_plot:
		img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR )
		for i in alignment_coords_original:
			cv2.circle(img_color, (i[1], i[0]), 9, (0,255,0), 4)
		cv2.imwrite('original_annotate.png', img_color)

	# in the original image, search near the top corrdinates to get the accurate coordinates:
	cross_original = np.zeros((135,135), dtype=np.uint8)
	cross_original[67-13:67+14, :] = 1
	cross_original[:, 67-13:67+14] = 1

	corrected_align_coord_original = []
	align_codes = []
	for loc in alignment_coords_original:
		start_x = max(0,loc[0]-search_range//2)
		end_x = min(loc[0]+search_range//2,final.shape[0])
		start_y = max(0,loc[1]-search_range//2)
		end_y = min(loc[1]+search_range//2,final.shape[1])
		
		crop = final[start_x:end_x, start_y:end_y]
		crop = cv2.erode(crop, np.ones((9, 9)), iterations=1)
		relative_conv = scipy.signal.convolve2d(crop.astype(np.float32), cross_original, mode='same')
		relative_align_coord = np.unravel_index(relative_conv.argmax(), relative_conv.shape)

		this_center_x = relative_align_coord[0] + start_x
		this_center_y = relative_align_coord[1] + start_y
		corrected_align_coord_original.append((this_center_x, this_center_y))

		# and get the alignment encoding (where it is in the gds file:)
		align_code = np.zeros((5,5))
		for i in range(5):
			for j in range(5):
				tl_x = this_center_x-67+27*i
				tl_y = this_center_y-67+27*j
				box = final[tl_x:tl_x+27, tl_y:tl_y+27]
				align_code[i,j] = np.sum(box) > 0.7*27*27*255
				if debug_plot and np.sum(box) > 0.7*27*27*255:
					cv2.rectangle(img_color, (tl_y, tl_x), (tl_y+27, tl_x+27), (0,0,255), 1)
		if debug_plot:
			print(align_code)
		align_codes.append(align_code)

	if debug_plot:
		cv2.imwrite('original_box.png', img_color)

	bl = decode_four_alignments(align_codes)

	return corrected_align_coord_original, bl, align_codes


def get_code_mask(align_code, grid_size=27): # grid size could be fine tuned based on actual SEM taken, better to be an odd number?
	mask = np.zeros((grid_size*5,grid_size*5))

	for i in range(5):
		for j in range(5):
			if align_code[i,j] == 1:
				mask[i*grid_size:(i+1)*grid_size, j*grid_size:(j+1)*grid_size] = 1

	return mask


def precise_align(SEM, align_coords, align_codes, search_range=150, grid_size=27, debug_plot=False):
	precise_a_coords = []
	masks = []
	for i in range(4):
		mask = get_code_mask(align_codes[i], grid_size=grid_size)
		masks.append(mask)
		loc = align_coords[i]
		start_x = max(0,loc[0]-search_range//2)
		end_x = min(loc[0]+search_range//2,SEM.shape[0])
		start_y = max(0,loc[1]-search_range//2)
		end_y = min(loc[1]+search_range//2,SEM.shape[1])
		
		crop = SEM[start_x:end_x, start_y:end_y]
		relative_conv = scipy.signal.convolve2d(crop.astype(np.float32), mask[::-1,::-1], mode='same')
		# cv2.imwrite("conv"+str(i)+".png", relative_conv/np.max(relative_conv)*255)
		relative_align_coord = np.unravel_index(relative_conv.argmax(), relative_conv.shape)

		this_center_x = relative_align_coord[0] + start_x
		this_center_y = relative_align_coord[1] + start_y
		precise_a_coords.append((this_center_x, this_center_y))

	if debug_plot:
		before_overlay = cv2.cvtColor(SEM, cv2.COLOR_GRAY2BGR)
		for i in range(4):
			mask = masks[i]
			loc = align_coords[i]
			before_overlay[round(loc[0]-grid_size*2.5):round(loc[0]-grid_size*2.5)+5*grid_size, round(loc[1]-grid_size*2.5):round(loc[1]-grid_size*2.5)+5*grid_size] = \
			0.5*before_overlay[round(loc[0]-grid_size*2.5):round(loc[0]-grid_size*2.5)+5*grid_size, round(loc[1]-grid_size*2.5):round(loc[1]-grid_size*2.5)+5*grid_size]+0.5*mask[:,:,None]*np.array([0,255,0])

		after_overlay = cv2.cvtColor(SEM, cv2.COLOR_GRAY2BGR)
		for i in range(4):
			mask = masks[i]
			loc = precise_a_coords[i]
			after_overlay[round(loc[0]-grid_size*2.5):round(loc[0]-grid_size*2.5)+5*grid_size, round(loc[1]-grid_size*2.5):round(loc[1]-grid_size*2.5)+5*grid_size] = \
			0.5*after_overlay[round(loc[0]-grid_size*2.5):round(loc[0]-grid_size*2.5)+5*grid_size, round(loc[1]-grid_size*2.5):round(loc[1]-grid_size*2.5)+5*grid_size]+0.5*mask[:,:,None]*np.array([0,255,0])

		cv2.imwrite("before_overlay.png", before_overlay)
		cv2.imwrite("after_overlay.png", after_overlay)

	return precise_a_coords

if __name__ == '__main__':
	start = time.time()
	path1 = "/media/group_scratch/shared/samsung_data/SEM_data/Sample2_dose12_1_processed/"
	path2 = "/media/group_scratch/shared/samsung_data/SEM_data/Sample2_dose12_2_processed/"
	path3 = "/media/group_scratch/shared/samsung_data/SEM_data/Sample2_dose34_1_processed/"
	path4 = "/media/group_scratch/shared/samsung_data/SEM_data/Sample2_dose34_2_processed/"
	for path in [path3, path4]:
		imgnames = list(os.listdir(path))
		for imgname in tqdm(imgnames):
			with open("./log.txt","a") as f:
				f.write(f'processing: {imgname}\n')
			if imgname[0] == 'C':
				continue
			img = path + imgname
			final = cv2.imread(img, cv2.IMREAD_UNCHANGED)
			align_coords, bl, align_codes = get_alignment_info(final, debug_plot=False)
			with open("./log.txt","a") as f:
				f.write(f'decoding code, all possible combinations: {bl}\n')
			if len(bl)!=0:
				align_coords = precise_align(final, align_coords, align_codes, search_range=150, grid_size=28, debug_plot=False)
			# in order to work with numpy and the way we create gds, let's flip it:
			final = final.T
			align_coords = [[i[1],i[0]] for i in align_coords]
			align_coords.sort(key = lambda x: 5*x[0]+x[1])
			if len(bl)!=0:
				for bl_valid in bl:
					r,c = bl_valid
					with open("./log.txt","a") as f:
						f.write(f'trying with {r}-{c}\n')
			
					margin = 1 # how much margin to leave outside 4 alignemnt marks in um
					resolution=3*nm
					dosage = c//62 + 2*(r//62) + 1
					local_map,error_output = gen_uint8_map(r,c, margin, path='/media/group_scratch/shared/samsung_data/gen_gds/500um_txt/', resolution=resolution, x0=50*nm, y0=50*nm) # please keep x0, y0 = 50nm, as was in the gds data generation

					if error_output:
						with open("./log.txt","a") as f:
							f.write(f'no gds file\n')
						continue
					# annotate the four alignment centers of the generated local map:
					offsetx = 0.25
					offsety = -0.25
					align_x_min, align_x_max = round((margin+offsetx)/resolution), round((margin+8+offsetx)/resolution)
					align_y_min, align_y_max = round((margin+offsety)/resolution), round((margin+8+offsety)/resolution)
					centers = [(align_x_min, align_y_min), (align_x_min, align_y_max), (align_x_max, align_y_min), (align_x_max, align_y_max)]
					
					
					filpped_center = np.asarray([[i[1],i[0]] for i in centers], dtype=np.float32)
					filpped_coords = np.asarray([[i[1],i[0]] for i in align_coords], dtype=np.float32)
					M = cv2.getPerspectiveTransform(filpped_coords, filpped_center)
					transformed_SEM = cv2.warpPerspective(final.astype(np.float32), M, local_map.shape)
					
                    # difference between the two maps, should always be almost all negative if alignment is correct
					difference1 = (
                        transformed_SEM[align_x_min:align_x_max, align_y_min:align_y_max]
                        - local_map[align_x_min:align_x_max, align_y_min:align_y_max]
                    )
					difference2 = (
                        local_map[align_x_min:align_x_max, align_y_min:align_y_max]
                        - transformed_SEM[align_x_min:align_x_max, align_y_min:align_y_max]
                    )
					positive_count = (difference1 > 0).sum() + (difference2 > 0).sum()
					if positive_count > 700000:
						with open("./log.txt", "a") as f:
							f.write(f"different pixels: {positive_count}, image does not match\n")
						continue
					else:
						with open("./log.txt", "a") as f:
							f.write(f"image matched, succeed, dosage {dosage} at {r} {c}\n")
						cv2.imwrite(
							f"./data/dosage{dosage}/SEM/{r}-{c}.png", cv2.transpose(transformed_SEM[align_x_min:align_x_max, align_y_min:align_y_max])
						)
						cv2.imwrite(
							f"./data/dosage{dosage}/GDS/{r}-{c}.png", cv2.transpose(local_map[align_x_min:align_x_max, align_y_min:align_y_max])
						)
						break
	end = time.time()
	print("Run time: ",end - start, ' s')
