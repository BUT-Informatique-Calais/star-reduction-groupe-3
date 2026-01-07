# 🌠 StarEX (beta)

![Python](https://img.shields.io/badge/Python-3.8+-green)

A simple but useful tool to achieve star reduction on .FITS files, using the Astropy.io and OpenCV libraries.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

## Authors

- COLIN Noé [@Kiizer](https://www.github.com/Kiizer861)
- MELOCCO David [@ThFoxY](https://www.github.com/ThFoxY)
- LECLERCQ-SPETER Simon [@Koshy](https://www.github.com/KoshyMVP)

## Installation

**A stable version of Python is required (< 3.13)**. A higher version may cause undesirable results or errors.

* See requirements.txt for full dependency list

### Libraries

* Astropy.io **7.2.0** - [User Guide](https://docs.astropy.org/en/stable/index_user_docs.html)
* Matplotlib **3.10.8** - [API Reference](https://matplotlib.org/stable/api/index.html)
* NumPy **2.20** - [Docs](https://numpy.org/doc/)
* OpenCV for Python **4.12.0.88** - [Modules](https://docs.opencv.org/4.x/index.html)
* Qt for Python **6.9.1** - [Docs](https://doc.qt.io/qtforpython-6.9/)

### Virtual Environment

It is recommended to create a virtual environment before installing dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Troubleshooting for Powershell:
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Dependencies
```bash
pip install -r requirements.txt
```

Or install dependencies manually (specified above):
```bash
pip install [package-name]
```

## Examples files
Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.
- Example 1: `examples/HorseHead.fits` (Black and white FITS image file for testing)
- Example 2: `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3: `examples/test_M31_raw.fits` (Color FITS image file for testing)

## First steps: How did we do it?

### FITS (Flexible Image Transport System)

Universal standard for sharing astronomical data

**What we understood:**
* Data storage (in table form) is 16 to 32 bits for total mathematical precision.
* It contains a metadata header (celestial coordinates, exposure time, sensor temperature, telescope used).
* It contains several extensions (images of different filters, spectra, or data tables) in a single file.

### Erosion

**What we understood:**
* That the kernel takes the average of an X by X square/rectangle and assigns the average to the central value of that square/rectangle
* That OpenCV is of type uint8 and ranges from [0,255] compared to Matplotlib which is of type float and ranges from [0,1]
* Pixel intensity is measured from [0,255]

### Morphological Operations

**What we understood:**
* The kernel is a matrix of ones of a specified size. The kernel is applied to each pixel in the image.
* Neighboring pixels are then used to calculate the new pixel value (if they are within the kernel range).
* Since it chooses the minimum value, bright areas tend to become darker after each iteration.
* As the number of iterations increases, the kernel is applied to the image several times. The pixels tend to reach a stable color after a few iterations.
* It goes darker and darker. In the final result, pixels are forming squares.

Dilation is the opposite of erosion.
* Combining erosion and dilation creates a closing operation.
* Combining dilation and erosion creates an opening operation.

### Color Conversion

When using openCV, colors are inverted. The fix we use is the following one:

* `image = cv.cvtColor(image, cv.COLOR_RGB2BGR)` converts the color channel order.

### Useful References

Here's some docs and articles we use

* https://en.wikipedia.org/wiki/Closing_(morphology)
* https://numpy.org/doc/2.2/reference/generated/numpy.ones.html#numpy.ones
* https://docs.astropy.org/en/stable/io/fits/index.html
* https://docs.astropy.org/en/latest/io/fits/usage/headers.html
* https://www.youtube.com/watch?v=qgJgh0a9qxU