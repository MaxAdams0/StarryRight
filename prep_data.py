import os
import cv2
import random
import time
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import argparse

from zernike4 import ZernikePolynomial
from tile import ImageTile

GREEN = '\u001b[38;5;10m'
YELLOW = '\u001b[38;5;11m'
RED = '\u001b[38;5;9m'
BLUE = '\u001b[38;5;12m'
ENDC = '\u001b[0m'

ROOT_DATA_DIR = None
DEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
TILE_DATA_DIR = os.path.join(os.path.dirname(__file__), "tiledimages")


# Fixed Hyperparameters
MIN_IMAGE_SIZE = 1048
OUTPUT_IMAGE_SIZE = 512
ACCEPTED_FORMATS = ['.jpg', '.jpeg', '.tif', '.tiff']
# Command-line adjustable Hyperparameters
BATCH_SIZE = None # Amount of images per process
FILE_LIMIT = None # -1 = disabled
OUTPUT_PER_IMAGE = None
PROCESSES = None
DEBUG = None

def process_image(input_path, output_dir, distortions):
	# To seed randoms to it is different every run
	random.seed = time.process_time()

	try:
		if os.path.splitext(input_path)[1] in ACCEPTED_FORMATS:
			# First tile the image into 512x512 images
			# Create tiler object
			tiler = ImageTile(input_path, MIN_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)
			# Split image into tiled images
			images = tiler.tile_image(debug=DEBUG)
			if len(images) == 0: # only caused by min_image_size not reached
				#print(f"{YELLOW}Image {input_path} is too small (<{MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}){ENDC}")
				return 'MINSIZE'
			images = tiler.choose_images(images, OUTPUT_PER_IMAGE)
		else:
			# Not an allowed format from the specified list of files
			print(f"{RED}Error loading image '{input_path}': Incorrect format (not {ACCEPTED_FORMATS}){ENDC}")
			return 'FORMAT'
	except Exception as e:
		print(f"{RED}Error loading image '{input_path}': {e}{ENDC}")
		return 'ERROR'
	
	# image_touple is a group of images and their names from the ImageTile.choose_images() function
	for image_touple in images:
		filename, image = image_touple

		output_path = os.path.join(output_dir, filename)

		"""
		if 'gaussian_blur' in distortions:
			image = cv2.GaussianBlur(image, (33, 33), 0)
		"""
		
		if 'zernike_blur' in distortions:
			try:
				# Phasemap parameters
				n = 3
				m = 1
				grid_size = 33

				# Initialize a Zernike Polynomial of order n,m
				zernike = ZernikePolynomial(n, m)
				# Generate phasemap & plot
				phasemap = zernike.generate_phasemap(grid_size=33)
				#zernike.plot_phasemap(phasemap)

				# PSF Parameters
				wavelength = 1  # Wavelength of light (micrometers) (visible light is 0.38-0.7)
				aperture_radius = 100  # Radius of the aperture (normalized to 1)
				psf_size = grid_size
				pixel_size = 0.01  # Size of each pixel in the detector

				# Generate PSF & plot
				psf = zernike.generate_psf(phasemap, wavelength, aperture_radius, psf_size, pixel_size)
				# Apply rotation to the PSF for randomization purposes
				theta = random.randint(0,360)
				rotated_psf = zernike.rotate_psf(psf, theta)
				#zernike.plot_psf(rotated_psf)

				image = zernike.apply_psf_to_image(image, rotated_psf)
			except Exception as e:
				print(f"{RED}Error applying zernike blur to image: {filename}")
				print(f"{e}{ENDC}")
				return 'ERROR'
		
		if 'contrast' in distortions:
			try:
				alpha = 0.5  # contrast control; smaller values lead to less contrast
				image = cv2.convertScaleAbs(image, alpha=alpha)
			except Exception as e:
				print(f"{RED}Error applying contrast to image: {filename}")
				print(f"{e}{ENDC}")
				return 'ERROR'

		if 'trailing' in distortions:
			try:
				rows, cols, ch = image.shape
				src_points = np.float32([[0, 0], [0, rows - 1], [cols - 1, 0]])
				dst_points = np.float32([[cols * 0.1, 0], [cols * 0.1, rows - 1], [cols - 1, 0]])  # 10% shift to right
				matrix = cv2.getAffineTransform(src_points, dst_points)
				image = cv2.warpAffine(image, matrix, (cols, rows))
			except Exception as e:
				print(f"{RED}Error applying trailing to image: {filename}")
				print(f"{e}{ENDC}")
				return 'ERROR'
		
		if 'haze' in distortions:
			try:
				intensity = 0.1
				haze = np.ones(image.shape, image.dtype) * 210  # this creates a white image
				image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)  # blending the images
			except Exception as e:
				print(f"{RED}Error applying haze to image: {filename}")
				print(f"{e}{ENDC}")
				return 'ERROR'
		
		# Save processed image to new path
		image = image.astype(np.uint8)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		cv2.imwrite(output_path, image)
	return 'SUCCESS'

def get_state_color(state):
	state_dict = {
		'SUCCESS':GREEN,
		'FORMAT':YELLOW,
		'SAMEDEST':YELLOW,
		'MINSIZE':YELLOW,
		'ERROR':RED,
	}

	return state_dict[state]

# For Multiprocessing - passing args
def process_batch_wrapper(args):
	batch_files, process_id = args
	start_time = time.perf_counter()

	image_time_avg, state = process_batch(batch_files)
	
	color = get_state_color(state)

	end_time = time.perf_counter()
	elapsed_time = round(end_time - start_time, 2)

	print(f"{color}Process: {BLUE}{process_id:3}",
	   		f"{color}Num Files: {BLUE}{len(batch_files):2}",
			f"{color}Total: {BLUE}{elapsed_time:6.2f}s",
			f"{color}Per Image: {BLUE}{image_time_avg:5.2f}s{ENDC}")

# Traverse the input dataset directory and distort images
def process_batch(files):
	image_time_total = 0
	for file_ in files:
		start_time = time.perf_counter()

		input_image_path = os.path.join(ROOT_DATA_DIR, file_)
		
		distortions = ['zernike_blur']

		state = process_image(input_image_path, DEST_DATA_DIR, distortions)

		end_time = time.perf_counter()
		image_time_total += (end_time - start_time)
	
	image_time_avg = round(image_time_total / len(files), 2)
	
	return image_time_avg, state

def get_batches():
	# Get all paths for an image's current path and its new path
	image_list = []
	if not os.path.exists(DEST_DATA_DIR):
		os.makedirs(DEST_DATA_DIR)
		print(f"Created Folder: {DEST_DATA_DIR}")
	for root, dirs, files in os.walk(ROOT_DATA_DIR):
		# Copy each file to the destination directory
		for f in files:
			# Get the relative path from the root directory
			relative_path = os.path.relpath(os.path.join(root, f), ROOT_DATA_DIR)
			image_list.append(relative_path)

	# Split files into batches
	num_of_files = FILE_LIMIT if FILE_LIMIT != -1 else len(image_list)
	file_batches = [image_list[i:i+BATCH_SIZE] for i in range(0, num_of_files, BATCH_SIZE)]

	print(f"{GREEN}Total number of files: {BLUE}{num_of_files}{ENDC}")
	print(f"{GREEN}Total number of batches: {BLUE}{len(file_batches)}{ENDC}")

	return file_batches

def main_wrapper():
	
	file_batches = get_batches()

	# Pool only takes one argument, so you have to form it into a tuple
	args_list = [(batch_files, i) for i, batch_files in enumerate(file_batches)]
	
	if not DEBUG:
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


def set_constants():
	global DEBUG, FILE_LIMIT, BATCH_SIZE, ROOT_DATA_DIR, PROCESSES

	# Instantiate the command-line argument parser and create potential arguments
	parser = argparse.ArgumentParser(
		description='Dataset preparation tool for VikX. Not meant for external use, but such use is permitted.'
	)
	parser.add_argument('-d', '--debug', 
					action='store_true', 
					help='Enable debug mode - may hurt performance'
	)
	parser.add_argument('-l', '--file_limit', 
					type=int, 
					default=-1, 
					help='The cap for how many files are processed'
	)
	parser.add_argument('-b', '--batch_size',
					type=int, 
					default=7, 
					help='The amount of files per batch'
	)
	parser.add_argument('-i', '--input', 
					type=str, 
					default=r"C:\dev\Datasets\AstrographyImages", 
					help='The input directory'
	)
	parser.add_argument('-o', '--opi', 
					type=str, 
					default=3, 
					help='The number of images processed per input image'
	)
	parser.add_argument('-p', '--processes', 
					type=str, 
					default=12, 
					help='The number of distortion threads - DO NOT set more than thread count of your cpu. May lead to deadlocks & context-switching overhead strain'
	)
	args = parser.parse_args()

	# Note: very odd implementation by the argparse devs, but if there is not a "--[argname]" argument
	# 	listed above, the args.[argname] will throw an error.
	DEBUG = args.debug
	FILE_LIMIT = args.file_limit
	BATCH_SIZE = args.batch_size
	ROOT_DATA_DIR = args.input
	PROCESSES = args.processes


if __name__ == "__main__":
	os.system('color')
	# Set command line arguments as constants (hyperparameters)
	set_constants()

	start_time = time.perf_counter()
	main_wrapper(debug=DEBUG)
	end_time = time.perf_counter()
	
	elapsed_time = round(end_time - start_time, 2)
	print(f"{GREEN}Dataset creation took a total of {BLUE}{elapsed_time:6.2f}s{ENDC}")