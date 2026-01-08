This document is a report of our research over time.

It helps with understanding the subject matter, ensuring that each member has a point of reference and can quickly access definitions to speed up the development process.

## Making our First steps: What we learned (05/01 - 06/01)

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
- https://www.universalis.fr/encyclopedie/astrometrie/
- https://en.wikipedia.org/wiki/Closing_(morphology)

### Color Conversion

**What we understood:**
* Color images are processed channel by channel (R, G, B).
* OpenCV uses a reversed order. It was a mess to properly sort it out!

**What difficulty we encountered:**

When using OpenCV, colors are inverted. The fix we use is the following one:

* `image = cv.cvtColor(image, cv.COLOR_RGB2BGR)` converts the color channel order.

_Useful links:_
- https://docs.opencv.org/4.x/d8/d01/group__imgproc__color__conversions.html#gaf86c09fe702ed037c03c2bc603ceab14

## Making an Algorithm: What we learned (07/01 - 08/01)

Our algorithm (and tool) is named StarEX (for StarEXtraction). The `algorithm.py` file contains a class StarEX with methods to process images.

### DAOStarFinder

**What we understood:**

* Star detection is performed using the image's luminance, with the DAOStarFinder from the photutils library.
* Each detected star has:
    - A position with x and y coordinates
    - A brightness value (also called flux)
* Detection sensitivity depends on:
    - `fwhm`: apparent star size (pixels)
    - `threshold_sigma`: the higher the value, the less stars are detected

_Useful links:_
- https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
- https://en.wikipedia.org/wiki/Full_width_at_half_maximum
- https://en.wikipedia.org/wiki/Gaussian_function
- https://photutils.readthedocs.io/en/stable/api/photutils.detection.IRAFStarFinder.html
- https://photutils.readthedocs.io/en/stable/api/photutils.psf.GaussianPSF.html
- https://github.com/laingmic005/aperture-photometry/blob/main/L2_photutils-detection.ipynb

### Binary Mask

**What we understood:**

* A binary mask is created from detected star positions (also called sources).
* Using 0 and 1 values only, black pixels represent the background and nebulae and white ones represent stars.
* The radius of each star mask is limited by two variables `r_min` and ``r_max`. Must be useful to achieve multi-size reduction?

**What difficulty we encountered:**
* Using square (or even rectangle) kernels were not accurate, creating visible boxes as if the results were in low quality.
* For accuracy, each star is represented with a circular kernel to avoid artefacts.

_Useful links:_
- https://photutils.readthedocs.io/en/latest/user_guide/detection.html
- https://www.youtube.com/watch?v=8162hq_5ZkQ
- https://github.com/laingmic005/aperture-photometry

### Gaussian Mask

**What we understood:**

* A Gaussian blur is applied to the binary mask to smooth transitions instead of sharp edges.
* This avoids visible artifacts in the final result!
* To apply the mask, values range from 0 to 1.

_Useful links:_
- https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
- https://en.wikipedia.org/wiki/Gaussian_blur

### Eroded image (also called ✨ Starless ✨)

**What we understood:**

* Morphologicial opening removes stars from the image (does erosion + dilation).
* It avoids undesirable holes and shape distortions.
* Only the background and nebula remain.

_Useful links:_
- https://en.wikipedia.org/wiki/Mathematical_morphology
- https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f

### Star Reduction

**What we understood:**
* The final result uses this beautiful formula: `Ifinal = (M * Ierode) + ((1 - M) * Ioriginal)`
    * M = Gaussian mask,
    * Ierode = Starless image,
    * Ioriginal = Original image.
* It reduces star brightness, preserves background details and avoids hard edges!
* For better results, `reduction_strength` ensure more control over resulted output.

**What difficulty we encountered:**
* Colors were not counted in. Maybe we did something wrong at first, but we manage to fix it by applying this formula over each channel!
* Each channel gets star removal and mask blending, before being stacked back into a final colored image!
* For masks, it has to be mono-channel (2D array). A luminance layer was used, called Luma to pay tribute to Mario's cute companions in the games Super Mario Galaxy 1 and 2! ✨

## Making choices: What we did (08/01)

Due to time constraints, we decided to focus our efforts on a high-performance tool rather than a graphical abstraction.

> With AI, creating interfaces in Qt is very quick and easy: no interest there. On the other hand, making the algorithm fast and efficient... hm, interesting.

The tool will be CLI-based, accessible via command line. The goal is to have no graphical interface, or at most image visualization only. The rest: commands, only commands!