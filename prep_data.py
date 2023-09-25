import os
import cv2
import random
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import sys

from zernike4 import ZernikePolynomials
from tile import ImageTile

GREEN = '\u001b[38;5;10m'
YELLOW = '\u001b[38;5;11m'
RED = '\u001b[38;5;9m'
BLUE = '\u001b[38;5;12m'
ENDC = '\u001b[0m'

ROOT_DATA_DIR = r"C:\dev\Datasets\AstrographyImages"
DEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TILE_DATA_DIR = os.path.join(os.path.dirname(__file__), "tiledimages")

BATCH_SIZE = 10 # Amount of images per process
OUTPUT_PER_IMAGE = 1
PROCESSES = 10
ACCEPTED_FORMATS = ['.jpg', '.jpeg', '.tif', '.tiff']

def distort_wrapper(input_path, output_path, distortions, opi):
	state_holder = []
	for _ in range(opi):
		state_holder.append(distort(input_path, output_path, distortions))

	return state_holder

def distort(input_path, output_path, distortions):
	random.seed = time.process_time()
	if input_path == output_path:
		print(f"{RED}Error loading image '{input_path}': input_path is same as output_path{ENDC}")
		return 'SAMEDEST'

	try:
		if os.path.splitext(input_path)[1] in ACCEPTED_FORMATS:
			# First tile the image into 512x512 images
			filename = os.path.splitext(os.path.basename(input_path))[0]
			output_dir = os.path.join(TILE_DATA_DIR, filename)
			# Make sure the directory exists
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)

			# Create tiler object
			tiler = ImageTile(input_path, output_dir)
			# Split image into tiled images
			tiler.tile_image(debug=False)
		else:
			# Not an allowed format
			print(f"{RED}Error loading image '{input_path}': Incorrect format (not {ACCEPTED_FORMATS}){ENDC}")
			return 'FORMAT'
	except Exception as e:
		print(f"{RED}Error loading image '{input_path}': {e}{ENDC}")
		return 'ERROR'
	
	for file_path in os.listdir(output_dir):
		file_path = os.path.join(output_dir, file_path)
		image = cv2.imread(file_path,  cv2.IMREAD_COLOR)

		if 'gaussian_blur' in distortions:
			image = cv2.GaussianBlur(image, (33, 33), 0)

		if 'zernike_blur' in distortions:
			# m = azmuthal frequency of Zernike polynomial
			# n = radical order
			# n_max = The maximum order of the Zernike polynomials that can be used
			n, m = 0, 0
			n_max = 10
			kernel_size = 5
			zernike = ZernikePolynomials(n_max)
			image = zernike.apply_psf_to_image(image, n, m, kernel_size)
		
		if 'contrast' in distortions:
			try:
				alpha = 0.5  # contrast control; smaller values lead to less contrast
				image = cv2.convertScaleAbs(image, alpha=alpha)

			except Exception as e:
				print(f"{RED}Error applying contrast to image '{input_path}': {e}{ENDC}")
				return 'ERROR'

		if 'trailing' in distortions:
			try:
				rows, cols, ch = image.shape
				src_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0]])
				dst_points = np.float32([[cols * 0.1, 0], [cols * 0.1, rows - 1], [cols - 1, 0]])  # 10% shift to right
				matrix = cv2.getAffineTransform(src_points, dst_points)
				image = cv2.warpAffine(image, matrix, (cols, rows))

			except Exception as e:
				print(f"{RED}Error applying trailing to image '{input_path}': {e}{ENDC}")
				return 'ERROR'
		
		if 'haze' in distortions:
			try:
				intensity = 0.1
				haze = np.ones(image.shape, image.dtype) * 210  # this creates a white image
				image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)  # blending the images

			except Exception as e:
				print(f"{RED}Error applying haze to image '{input_path}': {e}{ENDC}")
				return 'ERROR'
		
		# Save processed image to new path
		image = image.astype(np.uint8)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		cv2.imwrite(output_path, image)
	return 'SUCCESS'

# Get the color associated with the distort states (returns)
def get_state_color(state):
	state_dict = {
		'SUCCESS':GREEN,
		'FORMAT':YELLOW,
		'SAMEDEST':YELLOW,
		'ERROR':RED,
	}

	return state_dict[state]

# For Multiprocessing - passing args
def process_batch_wrapper(args):
	batch_files, process_id, opi, debug = args
	start_time = time.perf_counter()

	image_time_avg, states = process_batch(batch_files, opi, debug)

	end_time = time.perf_counter()
	elapsed_time = round(end_time - start_time, 2)

	base_color = GREEN if RED not in states else RED
	print(f"{base_color}Process: {BLUE}{process_id:3}",
	   		f"{base_color}Num Files: {BLUE}{len(batch_files):3}",
			f"{base_color}Total: {BLUE}{elapsed_time:6.2f}s",
			f"{base_color}Per Image: {BLUE}{image_time_avg:5.2f}s{ENDC}")

# Traverse the input dataset directory and distort images
def process_batch(files, opi, debug):
	
	image_time_total = 0
	for file_ in files:
		start_time = time.perf_counter()

		input_image_path = os.path.join(ROOT_DATA_DIR, file_)
		output_image_path = os.path.join(DEST_DATA_DIR, file_)

		if debug:
			print(f"From: {input_image_path} ----- To: {output_image_path}")
		
		distortions = ['gaussian_blur']

		state_holder = distort_wrapper(input_image_path, output_image_path, distortions, opi)

		end_time = time.perf_counter()
		image_time_total += (end_time - start_time)
	
	image_time_avg = round(image_time_total / len(files), 2)
	
	return image_time_avg, state_holder

def get_batches():
	# Get all paths for an image's current path and its new path

	image_list = []
	if not os.path.exists(DEST_DATA_DIR):
		os.makedirs(DEST_DATA_DIR)
		print(f"Created Folder: {DEST_DATA_DIR}")
	for root, dirs, files in os.walk(ROOT_DATA_DIR):
		for d in dirs:
			path = os.path.join(DEST_DATA_DIR, d)
			if not os.path.exists(path):
				os.makedirs(path)
				print(f"Created Folder: {path}")
		# Copy each file to the destination directory
		for f in files:
			# Get the relative path from the root directory
			relative_path = os.path.relpath(os.path.join(root, f), ROOT_DATA_DIR)
			image_list.append(relative_path)

	# Split files into batches
	file_batches = [image_list[i:i+BATCH_SIZE] for i in range(0, len(image_list), BATCH_SIZE)]

	print(f"{GREEN}Total number of files: {BLUE}{len(image_list)}{ENDC}")
	print(f"{GREEN}Total number of batches: {BLUE}{len(file_batches)}{ENDC}")

	return file_batches


def main_wrapper(debug=False):
	
	file_batches = get_batches()

	# Pool only takes one argument, so you have to form it into a tuple
	args_list = [(batch_files, i, OUTPUT_PER_IMAGE, debug) for i, batch_files in enumerate(file_batches)]
	
	if not debug:
		try:
			with mp.Pool(processes=PROCESSES) as pool:
				pool.imap_unordered(process_batch_wrapper, args_list)
				pool.close()
				pool.join()
		except Exception as e:
			print(f"{RED}Error running Processes in Pool: {e}{ENDC}")
	
	else:
		args = args_list[0]
		process_batch_wrapper(args)


if __name__ == "__main__":
	os.system('color')
	start_time = time.perf_counter()

	debug = bool(sys.argv[1])

	main_wrapper(debug=debug)

	end_time = time.perf_counter()
	elapsed_time = round(end_time - start_time, 2)
	print(f"{GREEN}Dataset creation took a total of {BLUE}{elapsed_time:6.2f}s{ENDC}")