from astropy.io import fits
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import sys

FILENAMES = ['./examples/test_M31_raw.fits', './examples/test_M31.fits', './examples/HorseHead.fits']

# https://numpy.org/doc/2.2/reference/generated/numpy.ones.html#numpy.ones

def convert_to_uint8 (data):
    if data.ndim == 3:
        if data.shape[0] == 3:
            data = np.transpose(data, (1, 2, 0))

        data_normalized = (data - data.min()) / (data.max() - data.min())

        plt.imsave('./results/original.png', data_normalized)

        image = np.zeros_like(data, dtype='uint8')

        for i in range(data.shape[2]):
            channel = data[:, :, i]
            image[:, :, i] = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype('uint8')
    else:
        plt.imsave('./results/original.png', data, cmap='gray')

        image = ((data - data.min()) / (data.max() - data.min()) * 255).astype('uint8')

    return image

if __name__ == "__main__":
    # Opens and reads a FITS file from the command line
    fits_file = sys.argv[1]
    hdul = fits.open(fits_file) # Returns an HDUList object.

    data = hdul[0].data
    print(f"Data shape: {data.shape}\n{data}")

    image = convert_to_uint8(data)
    print(f"Image shape: {image.shape}\n{image}")
    kernel = np.ones((int(sys.argv[2]), int(sys.argv[3])), np.uint8)

    img_eroded = cv.erode(image, kernel, iterations=int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    print(f"Eroded image shape: {img_eroded.shape}\n{img_eroded}")

    cv.imwrite('./results/eroded.png', img_eroded)

    print("Eroded image saved at ./results/eroded.png")

# Close the file
hdul.close()

# The kernel is a matrix of ones with a specifed size. For each pixel in the image, the kernel is applied to it.
# The neighboring pixels are then used to calculate the new value of the pixel, if there are inside the kernel range.
# Because it chooses the minimum value, bright areas tend to be darker after each iteration.

# When increasing the number of iterations, the kernel is applied multiple times to the image. The pixels tend to reach a stable color after a few iterations.
# It goes darker and darker. In the final result, pixels are forming squares.