This document is a report of our research over time.

It helps with understanding the subject matter, ensuring that each member has a point of reference and can quickly access definitions to speed up the development process.

## First steps: How did we do it?

### FITS (Flexible Image Transport System)

Universal standard for sharing astronomical data

**What we understood:**
* Data storage (in table form) is 16 to 32 bits for total mathematical precision.
* It contains a metadata header (celestial coordinates, exposure time, sensor temperature, telescope used).
* It contains several extensions (images of different filters, spectra, or data tables) in a single file.
* A .FITS file needs to be converted into uint8 (8-bit unsigned integer) to work with OpenCV.

_Useful links:_
- https://docs.astropy.org/en/latest/io/fits/index.html
- https://docs.astropy.org/en/latest/io/fits/usage/headers.html
- https://learn.astropy.org/tutorials/FITS-images.html
- https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html
- https://community.openastronomy.org/t/fits-image-processing-with-opencv-python/594

### Erosion

**What we understood:**
* That the kernel takes the average of an X by X square/rectangle and assigns the average to the central value of that square/rectangle.
* That OpenCV is of type uint8 and ranges from [0,255] compared to Matplotlib which is of type float and ranges from [0,1].
* Brightness is measured from [0,255].

_Useful links:_
- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- https://en.wikipedia.org/wiki/Erosion_(morphology)

### Morphological Operations

**What we understood:**
* The kernel is a matrix of ones of a specified size. The kernel is applied to each pixel in the image.
* Neighboring pixels are then used to calculate the new pixel value (if they are within the kernel range).
* Since it chooses the minimum value, bright areas tend to become less visible after each iteration.
* As the number of iterations increases, the kernel is applied to the image several times. This creates huge artifacts that are not good for accurate scientific processing.

Dilation is the opposite of erosion.
* Combining erosion and dilation creates a closing operation.
* Combining dilation and erosion creates an opening operation.

_Useful links:_
* https://en.wikipedia.org/wiki/Closing_(morphology)

### Color Conversion

When using openCV, colors are inverted. The fix we use is the following one:

* `image = cv.cvtColor(image, cv.COLOR_RGB2BGR)` converts the color channel order.