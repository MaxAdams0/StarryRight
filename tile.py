import cv2
import matplotlib.pyplot as plt
import os
import random

# Image path: C:\dev\Datasets\AstrographyImages\Hubble100\heic0206a.tif
"""
TILE_HEIGHT = 512
TILE_WIDTH = 512
IMAGE_DISPLAY_SCALE = 0.25
PYTHON_FILE_PATH = os.path.dirname(__file__)
TILES_DIR = os.path.join(os.path.dirname(__file__), "TiledImages")
"""

class ImageTile:

	def __init__(self, image_path:str, min_input_size:int, tile_size:int):
		self.image_path = image_path
		self.tile_size = tile_size
		self.min_input_size = min_input_size

	def get_image_info(self):
		# Get naming data about the input image file
		root = os.path.splitext(self.image_path)[0]
		filename = os.path.splitext(os.path.basename(self.image_path))[0]
		ext = os.path.splitext(self.image_path)[1]
		
		return root, filename, ext

	def tile_image(self, debug:bool=False):
		image_info = self.get_image_info()
		image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
		height, width, ch = image.shape

		if height < self.min_input_size or width < self.min_input_size:
			return None

		# Get the number of tiles horizontally and vertically
		num_tiles_width = (width//self.tile_size)+1
		num_tiles_height = (height//self.tile_size)+1
		# Get the individual tile sizes, ranked by x1,y1 and x2,y2 for the topL and botR corners
		tile_x1s = [i*self.tile_size
						if (i*self.tile_size <= width-self.tile_size)
						else width-self.tile_size
					for i in range(num_tiles_width)
		]
		tile_x2s = [i*self.tile_size+self.tile_size
						if (i*self.tile_size+self.tile_size <= width) # Make sure tile is not OOB
						else width # If it is, shift to image border
					for i in range(num_tiles_width)
		]
		tile_y1s = [i*self.tile_size
						if (i*self.tile_size <= height-self.tile_size)
						else height-self.tile_size
					for i in range(num_tiles_height)
		]
		tile_y2s = [i*self.tile_size+self.tile_size
						if (i*self.tile_size+self.tile_size <= height) # Make sure tile is not OOB
						else height # If it is, shift to image border
					for i in range(num_tiles_height)
		]

		images = []

		# Create and save each tile
		for row in range(num_tiles_width):
			for col in range(num_tiles_height):
				x1, y1, x2, y2 = tile_x1s[row], tile_y1s[col], tile_x2s[row], tile_y2s[col]
				name = image_info[1] + f"_{row}_{col}" + image_info[2]
				images.append((name, image[y1:y2, x1:x2]))

		if debug:
			debug_image = image.copy()
			for row in range(num_tiles_width):
				for col in range(num_tiles_height):
					x1, y1, x2, y2 = tile_x1s[row], tile_y1s[col], tile_x2s[row], tile_y2s[col]
					cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangles

			# Display the image with rectangles drawn
			plt.imshow(debug_image)
			plt.show()
		
		return images
	
	def choose_images(self, images, keep_num):
		chosen = []
		for i in range(keep_num):
			selection = random.randint(0, len(images)-1)
			if selection in chosen:
				i-=1
			else:
				chosen.append(images[selection])
		
		return chosen

	def stitch_image(input_dir, output_dir):
		return True