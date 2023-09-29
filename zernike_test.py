from zernike4 import ZernikePolynomial
import cv2
import random
"""
GOOD VALUES:
(n,m)
(1,1)
(2,0)
(2,2)
(3,1)
(4,0)
"""

# Phasemap parameters
n = 3
m = 1
grid_size = 33

# Initialize a Zernike Polynomial of order n,m
zernike = ZernikePolynomial(n, m)
# Generate phasemap & plot
phasemap = zernike.generate_phasemap(grid_size=33)
zernike.plot_phasemap(phasemap)

# PSF Parameters
wavelength = random.randint(2,3)  # Wavelength of light (micrometers) (visible light is 0.38-0.7)
aperture_radius = 100  # Radius of the aperture (normalized to 1)
psf_size = grid_size
pixel_size = 0.01  # Size of each pixel in the detector

# Generate PSF & plot
psf = zernike.generate_psf(phasemap, wavelength, aperture_radius, psf_size,  pixel_size)
# Apply rotation to the PSF for randomization purposes
theta = random.randint(0,360)
rotated_psf = zernike.rotate_psf(psf, theta)
zernike.plot_psf(rotated_psf)

# Read image and apply the PSF
image = cv2.imread('galaxies_heic0615a.jpg', cv2.IMREAD_COLOR)
blurred_image = zernike.apply_psf_to_image(image, rotated_psf)
cv2.imwrite("new.jpg", blurred_image)