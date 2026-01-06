from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

# https://docs.astropy.org/en/stable/io/fits/index.html
# Open and read the FITS file
fits_file = './examples/test_M31_raw.fits'
hdul = fits.open(fits_file) # Returns an HDUList object.
'''
The open function returns an object called an HDUList which is a list-like collection of HDU objects.
An HDU (Header Data Unit) is the highest level component of the FITS file structure, consisting of a header and (typically) a data array or table.
'''

# Displays information about the file
hdul.info()
'''
Filename: ./examples/test_M31_raw.fits
No.    Name      Ver    Type      Cards   Dimensions   Format
  0  PRIMARY       1 PrimaryHDU      54   (1141, 1052, 3)   float32
'''

# https://docs.astropy.org/en/latest/io/fits/usage/headers.html
# Displays header information
# FITS headers consist of a list of 80 byte “cards”, where a card contains a keyword, a value, and a comment.

# Access the data from the primary HDU
data = hdul[0].data

# Access header information
header = hdul[0].header

# Handle both monochrome and color images
# ndim = number of dimensions (3 for color, height and height)
if data.ndim == 3:
    # Color image - need to transpose to (height, width, channels)
    # (height, width, channels) is the standard format for OpenCV
    if data.shape[0] == 3:  # If channels are first: (3, height, width)
        data = np.transpose(data, (1, 2, 0))
    # If already (height, width, 3), no change needed
    
    # Normalizes the entire image to [0, 1] for matplotlib
    # Every data value will be between 0 and 1 in order for Matplotlib to display the image
    data_normalized = (data - data.min()) / (data.max() - data.min())
    
    # Save the data as a png image (no cmap for color images)
    plt.imsave('./results/original.png', data_normalized)
    
    # Normalizes each channel separately to [0, 255] for OpenCV
    image = np.zeros_like(data, dtype='uint8')
    # For every channel (R, G, B)
    for i in range(data.shape[2]):
        # Normalizes each channel separately
        # Multiplies it by 255 and then converts to uint8 (standard for OpenCV)
        # uint8: unsigned 8-bit integer (0 to 255) represents pixel intensity
        channel = data[:, :, i]
        image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
else:
    # Monochrome image (only height and width)
    # cmap='gray': tells Matplotlib to use a grayscale colormap to represent pixel intensity
    plt.imsave('./results/original.png', data, cmap='gray')
    
    # Convert to uint8 for OpenCV
    image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

# Define a kernel for erosion
# Returns a 3x3 matrix, filled with unsigned interger 1.
# It defines the area where the erosion will be applied.
kernel = np.ones((3,3), np.uint8)
# Perform erosion
eroded_image = cv.erode(image, kernel, iterations=1)

# Save the eroded image 
cv.imwrite('./results/eroded.png', eroded_image)

# Close the file
hdul.close()