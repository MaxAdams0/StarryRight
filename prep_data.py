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

# Terminal colors
GREEN = '\u001b[38;5;10m'
YELLOW = '\u001b[38;5;11m'
RED = '\u001b[38;5;9m'
BLUE = '\u001b[38;5;12m'
ENDC = '\u001b[0m'

# Fixed Hyperparameters
MIN_IMAGE_SIZE = 1048
OUTPUT_IMAGE_SIZE = 512
ACCEPTED_FORMATS = ['.jpg', '.jpeg', '.tif', '.tiff']

def process_image(params, input_path, output_dir, distortions):
	# To seed randoms to it is different every run
	random.seed = time.process_time()

	if os.path.splitext(input_path)[1] in ACCEPTED_FORMATS:
		try:
			# First tile the image into 512x512 images
			# Create tiler object
			tiler = ImageTile(input_path, MIN_IMAGE_SIZE, OUTPUT_IMAGE_SIZE)
			# Split image into tiled images
			images = tiler.tile_image()
			if images is None: # only caused by min_image_size not reached
				#print(f"{YELLOW}Image {input_path} is too small (<{MIN_IMAGE_SIZE}x{MIN_IMAGE_SIZE}){ENDC}")
				return 'MINSIZE'
			chosen_images = tiler.choose_images(images, params.tiles)
		except Exception as e:
			print(f"{RED}Error loading image: {input_path}")
			print(f"{e}{ENDC}")
			return 'ERROR'
	else:
		# Not an allowed format from the specified list of files
		print(f"{RED}Error loading image '{input_path}': Incorrect format (not {ACCEPTED_FORMATS}){ENDC}")
		return 'FORMAT'
	
	# image_touple is a group of images and their names from the ImageTile.choose_images() function
	for image_touple in chosen_images:
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
				grid_size = 32
				upscale_factor = 5 # multiplier to grid_size

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

def process_batch(args):
	# Extract variables from arguments
	params, batch_files, process_id = args

	image_time_total = 0
	for file_path in batch_files:
		start_time = time.perf_counter()

		input_image_path = os.path.join(params.input, file_path)
		# List of every distortion applied to the image
		# This should be inputted later, its stupid to force someone to edit this
		distortions = ['zernike_blur']
		# The state is the return statements, which is just a letter code
		state = process_image(params, input_image_path, params.output, distortions)

		end_time = time.perf_counter()
		image_time_total += (end_time - start_time)
	image_time_avg = round(image_time_total / len(batch_files), 2)
	
	# This is stupid but whatever
	color = get_state_color(state)

	print(f"{color}Process: {BLUE}{process_id:3}",
	   		f"{color}Num Files: {BLUE}{len(batch_files):2}",
			f"{color}Per Image: {BLUE}{image_time_avg:5.2f}s{ENDC}"
	)

def get_batches(params):
	# Get all paths for an image's current path and its new path
	image_list = []
	if not os.path.exists(params.output):
		os.makedirs(params.output)
	for root, dirs, files in os.walk(params.input):
		# Copy each file to the destination directory
		for f in files:
			# Get the relative path from the root directory
			relative_path = os.path.relpath(os.path.join(root, f), params.input)
			image_list.append(relative_path)

	# Split files into batches
	num_of_files = params.file_limit if params.file_limit != -1 else len(image_list)
	file_batches = [image_list[i:i+params.batch_size] for i in range(0, num_of_files, params.batch_size)]

	print(f"{GREEN}Total number of files: {BLUE}{num_of_files}{ENDC}")
	print(f"{GREEN}Total number of batches: {BLUE}{len(file_batches)}{ENDC}")

	return file_batches

def main_wrapper(params):
	
	file_batches = get_batches(params)

	# Pool only takes one argument, so you have to form it into a tuple
	args_list = [(params, batch_files, i) for i, batch_files in enumerate(file_batches)]
	
	if not params.debug:
		try:
			with mp.Pool(processes=params.processes) as pool:
				pool.imap_unordered(process_batch, args_list)
				pool.close()
				pool.join()
		except Exception as e:
			print(f"{RED}Error running Processes in Pool:")
			print(f"{e}{ENDC}")
	
	else:
		args = args_list[0]
		process_batch(args)

def set_constants():
	# Instantiate the command-line argument parser and create potential arguments
	parser = argparse.ArgumentParser(
		description='Dataset preparation tool for VikX. Not meant for external use, but such use is permitted.'
	)
	parser.add_argument('-i', '--input', 
					type=str, 
					default="C:\dev\Datasets\AstrographyImages", 
					help='The input directory'
	)
	parser.add_argument('-o', '--output', 
					type=str, 
					default=os.path.join(os.path.dirname(__file__), "dataset"), 
					help='The output directory'
	)
	parser.add_argument('-l', '--file_limit', 
					type=int, 
					default=-1, 
					help='The cap for how many files are processed'
	)
	parser.add_argument('-t', '--tiles', 
					type=int, 
					default=3, 
					help='The number of images processed per input image'
	)
	parser.add_argument('-b', '--batch_size',
					type=int, 
					default=5, 
					help='The amount of files per batch'
	)
	parser.add_argument('-p', '--processes', 
					type=int, 
					default=10, 
					help='The number of distortion threads - DO NOT set more than thread count of your cpu. May lead to deadlocks & context-switching overhead strain'
	)
	parser.add_argument('-d', '--debug', 
					action='store_true', 
					help='Enable debug mode - may hurt performance'
	)
	args = parser.parse_args()

	# Note: very odd implementation by the argparse devs, but if there is not a "--[argname]" argument
	# 	listed above, the args.[argname] will throw an error.
	return args

if __name__ == "__main__":
	os.system('color')
	# Set command line arguments as constants (hyperparameters)
	params = set_constants()

	print(
		f"{GREEN}Running with args:",
		f"{GREEN}i='{BLUE}{params.input}'",
		f"{GREEN}o='{BLUE}{params.output}'",
		f"{GREEN}f={BLUE}{params.file_limit}",
		f"{GREEN}t={BLUE}{params.tiles}",
		f"{GREEN}b={BLUE}{params.batch_size}",
		f"{GREEN}p={BLUE}{params.processes}{ENDC}",
		f"{GREEN}d={BLUE}{params.debug}",
	)

	start_time = time.perf_counter()
	main_wrapper(params)
	end_time = time.perf_counter()
	
	elapsed_time = round(end_time - start_time, 2)
	print(f"{GREEN}Dataset creation took a total of {BLUE}{elapsed_time:6.2f}s{ENDC}")