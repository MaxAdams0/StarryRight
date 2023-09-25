import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# Image path: C:\dev\Datasets\AstrographyImages\Hubble100\heic0206a.tif
"""
TILE_HEIGHT = 512
TILE_WIDTH = 512
IMAGE_DISPLAY_SCALE = 0.25
PYTHON_FILE_PATH = os.path.dirname(__file__)
TILES_DIR = os.path.join(os.path.dirname(__file__), "TiledImages")
"""

class ImageTile:

	def __init__(self, image_path:str, output_dir:str, target_width:int=512, target_height:int=512):
		self.image_path = image_path
		self.output_dir = output_dir
		self.tile_width = target_width
		self.tile_height = target_height

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

		# Get the number of tiles horizontally and vertically
		num_tiles_width = (width//self.tile_width)+1
		num_tiles_height = (height//self.tile_height)+1
		# Get the individual tile sizes, ranked by x1,y1 and x2,y2 for the topL and botR corners
		tile_x1s = [i*self.tile_width
						if (i*self.tile_width <= width-self.tile_width)
						else width-self.tile_width
					for i in range(num_tiles_width)
		]
		tile_x2s = [i*self.tile_width+self.tile_width
						if (i*self.tile_width+self.tile_width <= width) # Make sure tile is not OOB
						else width # If it is, shift to image border
					for i in range(num_tiles_width)
		]
		tile_y1s = [i*self.tile_height
						if (i*self.tile_height <= height-self.tile_height)
						else height-self.tile_height
					for i in range(num_tiles_height)
		]
		tile_y2s = [i*self.tile_height+self.tile_height
						if (i*self.tile_height+self.tile_height <= height) # Make sure tile is not OOB
						else height # If it is, shift to image border
					for i in range(num_tiles_height)
		]

		try:
			# Create and save each tile
			for row in range(num_tiles_width):
				for col in range(num_tiles_height):
					x1, y1, x2, y2 = tile_x1s[row], tile_y1s[col], tile_x2s[row], tile_y2s[col]
					cv2.imwrite(
						filename=os.path.join(self.output_dir, f"{row}-{col}{image_info[2]}"), 
						img=image[y1:y2, x1:x2]
					)
			print(f"Save all tiles to {self.output_dir}")
		except Exception as e:
			print(f"ERROR while tiling: {e}")

		if debug:
			debug_image = image.copy()
			for row in range(num_tiles_width):
				for col in range(num_tiles_height):
					x1, y1, x2, y2 = tile_x1s[row], tile_y1s[col], tile_x2s[row], tile_y2s[col]
					cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green rectangles

			# Display the image with rectangles drawn
			plt.imshow(debug_image)
			plt.show()

	def stitch_image(input_dir, output_dir):
		return True


"""
# First tile the image into 512x512 images
	tiler = ImageTile(input_path, output_directory)
	tiler.tile_image(debug=False)
"""