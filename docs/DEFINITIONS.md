## Useful definitions to know!

Here are some (very) useful and quick definitions when it comes to astrometric and image processing.

> This document to keep tracks of new scientific terms and to help contribute to StarEX!

### Astrometry:
A branch of astronomy that determines the position of celestial bodies on the celestial sphere by measuring angles.

### Open Source Computer Vision Library (openCV):
An open source computer vision and machine learning software library.

### Atmospheric diffusion or turbulence (known as seeing):
The effect of light scattering produced by the brightest stars.
__See Figure 1 - (a) M45 detail__

### Nebula:
A cluster of interstellar gas and dust.

### Star reduction:
Reducing the apparent diameter of stars in order to extract them from a nebula. __See Figure 1 - (b) Sadr (γ Cygni)__

### Point source:
Light emitted in all directions (as in the case of a star).

### PSF (Point Spread Function):
The spread of a star's image captured by an optical instrument, caused by the diffraction of light.

### FWHM (Full Width at Half Maximum):
A measure of star size in pixels. Lower values indicate sharper stars.

### Sigma (σ):
Standard deviation of background noise. Used to define detection thresholds. In other words: what can be considered as a star?

### Threshold sigma:
The number of σ above background noise required to detect a star.

### Flux:
Total brightness of a detected star.

### Luminance (Luma):
Grayscale intensity extracted from a color image for analysis.

### Binary mask:
A black-and-white image where stars are marked as white and background as black.

### Gaussian blur:
A smoothing filter used to soften mask edges and avoid harsh transitions.

### Morphological opening:
An erosion followed by dilation, used to remove small bright structures (stars).

### Kernel:
A small matrix used for morphological operations.

### Kernel radius:
The size of the morphological structuring element.

### Erosion:
Shrinks bright regions in a mask or image.

### Dilation:
Expands bright regions.

### Inpainting:
A method that fills masked areas using surrounding pixel data. Very useful!

### Starless image:
An image where stars have been removed or minimized.

### Reduction strength:
Controls how strongly stars are visible (0 = none, 1 = maximum)

### Multiscale processing:
Uses different mask sizes depending on star magnitude and brightness.

### Tile processing:
Splitting large images into smaller blocks for faster processing.

### Overlap:
Shared pixel between tiles to avoid visible artefacts.

### DAOStarFinder:
A star detection algorithm based on PSF fitting.

### Sigma clipping:
A statistical method to estimate background noise.

### FITS (Flexible Image Transport System):
Standard astronomy file format for scientific images.

### BSCALE / BZERO:
Header values used to calibrate raw FITS data.

### HDU (Header Data Unit):
A FITS file structure containing metadata and image data.

### Bit depth:
Nubmer of intensity levels in an image (8-bit, 16-bit, ...).

### Background noise:
Random pixel variations unrelated to real objets.

### Linear blending:
Smoothly merging overlapping tiles.

### Parallel processing:
Running multiple computations simultaneously using CPU cores.

### Batch processing:
Processing multiple images at once, with same parameters.

### Telea method:
Fast inpainting algorithm used by OpenCV.

### INPAINT_NS:
Alternative inpainting algorithm (Navier-Stokes).