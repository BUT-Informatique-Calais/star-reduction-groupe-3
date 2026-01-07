# https://www.youtube.com/watch?v=qgJgh0a9qxU

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from photutils.detection import DAOStarFinder
import sys
from astropy.stats import sigma_clipped_stats

# DAOStarFinder detects stars in an image by using DAOFIND (Stetson 1987) algorithm.
# Stars are significant peaks above the background noise. There are not perfect bright points, because
# of seeing and other artifacts such as the optical system of the telescope or the camera that took
# the image. Its intensity reaches a peak in its center and then decreases at the edges.

# https://www.youtube.com/watch?v=qgJgh0a9qxU
# https://docs.astropy.org/en/latest/api/astropy.stats.sigma_clipped_stats.html#astropy.stats.sigma_clipped_stats
# https://photutils.readthedocs.io/en/2.3.0/api/photutils.detection.DAOStarFinder.html

# Opens and reads the FITS file
fits_file = sys.argv[1]
hdul = fits.open(fits_file)
data = hdul[0].data
print(f"Data shape: {data.shape}\n{data}")

hdul.close()

# Converts to 2D image if necessary
if data.ndim == 3:
    img = np.mean(data, axis=0)
else:
    img = data

print(f"Image: {img}")

# Calculates image statistics with sigma clipping
# Sigma-clipping avoids calculating a gross mean that contains outliers (stars, hot pixels, extreme noise) which is usually the case in astronomical images
# Pixels with an higher value than the specified sigma are considered outliers
# mean: mean value of the image, usually not used for this kind of analysis
# median: useful to subtract the background from outliers
# std: standard deviation which represents noise level, useful to detect stars (also called threshold)
mean, median, std = sigma_clipped_stats(img, sigma=3.0)
'''
print(f"Mean: {mean}, Median: {median}, Std Dev: {std}")
'''

# Detects stars using DAOStarFinder
# treshold: the higher the value, the less stars will be detected
# fwhm stands for Full Width at Half Maximum: the apparent size of the star (in pixels). It is
# usually a value between 2 and 3 pixels.
daofind = DAOStarFinder(fwhm=3.0, threshold=2.5 * std)
sources = daofind(img - median) # Detects stars

# Creates binary mask for stars
mask = np.zeros_like(img, dtype=np.uint8)

if sources is None or len(sources) == 0:
    print("Not stars detected! Try with a different threshold.")
else:
    # Gets flux range for normalization
    # min_flux: lowest star detected
    # max_flux: highest star detected
    # A flux is the brightness of a star (a mean of each pixel in a star)
    min_flux = np.min(sources['flux'])
    max_flux = np.max(sources['flux'])

    # If only one star detected, set normalized flux to 0.5 to avoid division by zero (crash!)
    if max_flux == min_flux:
        normalized_flux = 0.5

    # Draws each detected star as a white circle with variable size
    for i in range(len(sources)):
        # Gets star position
        x = int(round(sources['xcentroid'][i]))
        y = int(round(sources['ycentroid'][i]))
        flux = sources['flux'][i]
        
        # Calculates circle radius based on flux (1 to 5 pixels)
        # A bigger circle means a bigger star
        normalized_flux = (flux - min_flux) / (max_flux - min_flux)
        radius = int(1 + normalized_flux * 4)
        
        # Checks if star center is within image bounds
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            # Draws white circle for star
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    # Checks if pixel is inside circle
                    if dx*dx + dy*dy <= radius*radius:
                        if 0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]:
                            mask[ny, nx] = 255

print(f"Stars: {sources}")

# Saves mask with OpenCV
cv.imwrite('./results/mask.png', mask)

# Save mask with matplotlib (for better visualization)
plt.figure(figsize=(8, 8))
plt.imshow(mask, cmap='gray', origin='lower', vmin=0, vmax=255)
plt.title("Binary mask of stars")
plt.colorbar()
plt.savefig("./results/binary_mask.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"Number of stars detected: {len(sources)}")