from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import os
from tabulate import tabulate
from typing import Tuple, Optional


# TODO: Add Optional types
# =========================
# FILESYSTEM UTILITIES
# =========================
def does_file_exist(filepath: str) -> bool:
    '''Checks whether a file exists

    Args:
        filepath (str): The path to the file

    Returns:
        bool: True if the file exists
    '''
    return os.path.isfile(filepath)

def does_dir_exist(dirpath: str) -> bool:
    '''Checks whether a directory exists

    Args:
        dirpath (str): The path to the directory

    Returns:
        bool: True if the directory exists
    '''
    return os.path.isdir(dirpath)

def is_file_fits_format(filepath: str) -> bool:
    '''Checks if a file is in FITS format

    Args:
        filepath (str): The path to the file

    Returns:
        bool: True if the file extension matches FITS formats.
    '''
    return filepath.lower().endswith(('.fits', '.fit', '.fts'))

# =========================
# FITS UTILITIES
# =========================
def get_hdu_list(filepath: str) -> fits.HDUList:
    '''Gets an HDUList from a FITS file.

    Args:
        filepath (str): The path to the FITS file

    Returns:
        fits.HDUList: Opened HDU list.
    '''
    if not does_file_exist(filepath):
        raise Exception("File not found.")
    if not is_file_fits_format(filepath):
        raise Exception("Invalid file format. Please provide a FITS file.")

    hdul = fits.open(filepath)
    return hdul

def get_hdu_data(hdul: fits.HDUList, index: int=0) -> NDArray[np.floating]:
    '''Gets the data from an HDUList

    Args:
        hdul (fits.HDUList): Opened HDU list.
        index (int, optional): Index of the HDU. Defaults to 0.

    Returns:
        ndarray: Data from the HDU
    '''
    return hdul[index].data

def get_hdu_header(hdul: fits.HDUList, index: int=0) -> fits.Header:
    '''Gets the header from an HDUList

    Args:
        hdul (fits.HDUList): Opened HDU list.
        index (int, optional): Index of the HDU. Defaults to 0.

    Returns:
        fits.Header: Header from the HDU
    '''
    return hdul[index].header

def create_fits_header_table(hdul: fits.HDUList, index: int=0):
    '''Creates a table that contains FITS header info

    Args:
        hdul (fits.HDUList): Opened HDU list
        index (int, optional): Index of the HDU. Defaults to 0.

    Returns:
        str: A string containing the table
    '''
    header = get_hdu_header(hdul, index)
    table = [[key, value] for key, value in header.items()]
    return tabulate(table, headers=["Key", "Value"], tablefmt="grid")

def has_color_channels(data: NDArray[np.floating]) -> bool:
    '''Checks if the FITS file has color channels

    Args:
        data (ndarray): The FITS image

    Returns:
        bool: True if the FITS file has color channels
    '''
    return data.ndim == 3

# =========================
# IMAGE NORMALIZATION/SAVING
# =========================
def normalize_minmax(data: NDArray[np.floating], eps: float=1e-8) -> NDArray[np.float32]:
    '''Normalize image data to the [0, 1] range using min-max scaling.

    Args:
        data (ndarray): The image to normalize
        eps (float, optional): A small number to avoid division by zero. Defaults to 1e-8.

    Returns:
        ndarray: The normalized image
    '''
    min_val = np.min(data)
    max_val = np.max(data)

    if max_val - min_val < eps:
        return np.zeros_like(data, dtype=np.float32)

    return (data - min_val) / (max_val - min_val)

def save_preview_image(data: NDArray[np.floating], filepath: str) -> None:
    '''Saves an image preview using Matplotlib

    Args:
        data (ndarray): The image to save
        filepath (str): The path to save the image
    '''
    if data.ndim == 3:
        plt.imsave(filepath, normalize_minmax(data))
    else:
        plt.imsave(filepath, data, cmap='gray')

def save_float_image(data: NDArray[np.floating], filepath: str) -> None:
    '''Save a floating-point image as an 8-bit PNG using OpenCV

    Args:
        data (ndarray): The image to save (H, W) or (H, W, C) or (C, H, W)
        filepath (str): The path to save the image
    '''
    # Ensure (H, W, C) format for color images
    if data.ndim == 3:
        if data.shape[0] == 3:
            # Convert (C, H, W) -> (H, W, C)
            data = np.transpose(data, (1, 2, 0))
        
        # Debug: Check channel differences
        print(f"[DEBUG] Channel stats: R={data[...,0].mean():.2f}, G={data[...,1].mean():.2f}, B={data[...,2].mean():.2f}")
        
        # Normalize each channel separately
        img = np.zeros_like(data, dtype=np.uint8)
        for c in range(3):
            img[..., c] = (normalize_minmax(data[..., c]) * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(filepath, img)
    else:
        # Monochrome
        cv.imwrite(filepath, (normalize_minmax(data) * 255).astype(np.uint8))

def save_combined_data(image1: NDArray[np.uint8], image2: NDArray[np.uint8], filepath: str) -> NDArray[np.uint8]:
    '''Save two images side by side for comparison

    Args:
        image1 (ndarray): The first image to save
        image2 (ndarray): The second image to save

    Returns:
        ndarray: The combined image
    '''
    combined = np.hstack((image1, image2))
    cv.imwrite(filepath, combined)
    return combined

def save_combined_images(image1_path: str, image2_path: str, filepath: str) -> NDArray[np.uint8]:
    '''Save two images side by side for comparison

    Args:
        image1_path (str): The path to the first image to save
        image2_path (str): The path to the second image to save

    Returns:
        ndarray: The combined image
    '''
    image1 = cv.imread(image1_path)
    image2 = cv.imread(image2_path)
    combined = np.hstack((image1, image2))
    cv.imwrite(filepath, combined)
    return combined

# =========================
# KERNEL TOOLS
# =========================
def create_square_kernel(size: int) -> NDArray[np.uint8]:
    '''Creates a square kernel of a given size

    Args:
        size (int): The size of the kernel

    Returns:
        ndarray: The kernel
    '''
    return np.ones((size, size), np.uint8)

def create_circular_kernel(radius):
    '''Creates a circular kernel of a given radius

    Args:
        radius (int): The radius of the kernel

    Returns:
        ndarray: The kernel
    '''
    size = 2 * radius + 1
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y <= radius*radius
    return mask.astype(np.uint8)

def fits_to_uint8_image(data: NDArray[np.floating]) -> NDArray[np.uint8]:
    '''Converts a FITS image to a uint8 OpenCV image

    Args:
        data (ndarray): The FITS image to convert

    Returns:
        ndarray: The converted image
    '''
    # Transposes to (height, width, channels)
    if data.ndim == 3 and data.shape[0] == 3:
        data = np.transpose(data, (1, 2, 0))

    # Has channels
    if data.ndim == 3:
        img = np.zeros_like(data, dtype=np.uint8)

        # For every channel
        for c in range(3):
            img[..., c] = (normalize_minmax(data[..., c]) * 255).astype(np.uint8)
        # Converts to BGR
        return cv.cvtColor(img, cv.COLOR_RGB2BGR)
    else:
        # Monochrome
        return (normalize_minmax(data) * 255).astype(np.uint8)

# =========================
# MASK UTILITIES
# =========================
def erode_mask(mask: NDArray[np.uint8], kernel: NDArray[np.uint8], iterations: int=1) -> NDArray[np.uint8]:
    '''Erodes a mask

    Args:
        mask (ndarray): The binary mask to erode
        kernel (ndarray): The kernel to use for erosion
        iterations (int, optional): The number of times to erode the binary mask. Defaults to 1.

    Returns:
        ndarray: The eroded mask
    '''
    return cv.erode(mask, kernel, iterations=iterations)

def dilate_mask(mask: NDArray[np.uint8], kernel: NDArray[np.uint8], iterations: int=1) -> NDArray[np.uint8]:
    '''Dilates a mask

    Args:
        mask (ndarray): The binary mask to dilate
        kernel (ndarray): The kernel to use for dilation
        iterations (int, optional): The number of times to dilate the binary mask. Defaults to 1.

    Returns:
        ndarray: The dilated mask
    '''
    return cv.dilate(mask, kernel, iterations=iterations)

def convert2luma(image: NDArray[np.floating]) -> NDArray[np.floating]:
    '''Converts an image to luma

    Args:
        image (ndarray): The image to convert

    Returns:
        ndarray: The converted image
    '''
    if image.ndim == 2:
        return image.astype(np.float32)
    elif image.ndim == 3:
        # C x H x W
        if image.shape[0] == 3:
            return np.mean(image, axis=0).astype(np.float32)
        # H x W x C
        elif image.shape[2] == 3:
            return np.mean(image, axis=2).astype(np.float32)
        else:
            # fallback: mean over last axis
            return np.mean(image, axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape {image.shape}")

def filter_noise(image: NDArray[np.floating], sigma: float=3.0) -> Tuple[float, float, float]:
    '''Filters noise using sigma clipping
    
    Args:
        image (ndarray): The image to estimate the background from
        sigma (float, optional): The sigma value to use for sigma clipping. Defaults to 3.0.

    Returns:
        tuple: A tuple containing the mean, median, and standard deviation
    '''
    mean, median, std = sigma_clipped_stats(image, sigma=sigma)
    return mean, median, std

def detect_stars_dao(image: NDArray[np.floating], fwhm: float=3.0, threshold_sigma: float=2.5) -> tuple:
    '''Detects stars in an image using DAOStarFinder

    Args:
        image (ndarray): The image to detect stars in
        fwhm (float, optional): The full width at half maximum of the stars. Defaults to 3.0.
        threshold_sigma (float, optional): The threshold sigma value to use for star detection. Defaults to 2.5.

    Returns:
        tuple: A tuple containing the detected stars, the median, and the standard deviation
    '''
    mean, median, std = filter_noise(image)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * std)
    sources = daofind(image - median)
    return sources, median, std

def build_binary_mask(
    shape: Tuple[int, int], 
    sources, 
    r_min: int=2, 
    r_max: int=8
):
    '''Builds a binary mask for stars

    Args:
        shape (tuple): The shape of the mask
        sources (list): The detected stars
        r_min (int, optional): The minimum radius of the stars. Defaults to 2.
        r_max (int, optional): The maximum radius of the stars. Defaults to 8.

    Returns:
    '''
    mask = np.zeros(shape, dtype=np.uint8)

    if sources is None or len(sources) == 0:
        return mask

    flux = sources["flux"]
    f_min, f_max = flux.min(), flux.max()

    for src in sources:
        x = int(round(src["xcentroid"]))
        y = int(round(src["ycentroid"]))

        f = src["flux"]
        f_norm = (f - f_min) / (f_max - f_min + 1e-8)

        radius = int(r_min + (r_max - r_min) * np.sqrt(f_norm))

        cv.circle(mask, (x, y), radius, 255, -1)

    return mask

def apply_gaussian_blur(mask: NDArray[np.uint8], sigma: float=2.0):
    '''Applies a Gaussian blur to a mask

    Args:
        mask (ndarray): The mask to blur
        sigma (float, optional): The sigma value to use for the Gaussian blur. Defaults to 2.0.

    Returns:
        ndarray: The blurred mask
    '''
    mask_f = mask.astype(np.float32) / 255.0
    return cv.GaussianBlur(mask_f, ksize=(0, 0), sigmaX=sigma)

def get_starless_image(image: NDArray[np.floating], kernel_radius: int=4, iterations: int=1):
    '''Gets a starless image by using morphological opening. Morphological opening is a combination of erosion and dilation

    Args:
        image (ndarray): The image to get the starless image from
        kernel_radius (int, optional): The radius of the kernel to use for morphological opening. Defaults to 4.

    Returns:
        ndarray: The eroded image
    '''
    kernel = create_circular_kernel(kernel_radius)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations)

# =========================
# TESTING PLAYGROUND YAY!
# =========================
if __name__ == "__main__":
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir, "examples/test_M31_linear.fits")
    hdul = get_hdu_list(filepath)
    fits_data = get_hdu_data(hdul)
    hdul.close()

    fits_header = create_fits_header_table(hdul, 0)
    print(fits_header)