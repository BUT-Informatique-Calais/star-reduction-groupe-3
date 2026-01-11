import multiprocessing as mp
# For multiprocessing: runs algorithms in parallel along with the tiling process.
# It uses mp.cpu_count() to determine the number of workers (CPU cores).
#! Watch out! only works with bigger images (> 2000x2000) or it will be slower!
#! Due to time constraints, AI was used to explain how to use multiprocessing in Python. However, the thought process behind this algorithm was by us!

# Third-party imports
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# Third-party imports
from photutils.detection import DAOStarFinder
from pathlib import Path # For compatibility: it avoids struggle between Windows and UNIX paths (https://docs.python.org/3/library/pathlib.html)
from tabulate import tabulate
import cv2 as cv
import numpy as np

# Typing
from numpy.typing import NDArray
from typing import Tuple, Optional, Literal, List

# Avoids warnings when sources are None
import warnings
from photutils.utils import NoDetectionsWarning
warnings.filterwarnings("ignore", category=NoDetectionsWarning)

# TODO: Un-installed libraries must generate warnings/errors. See PIL in save_gif function!
# TODO: Separate all functions in different files (SOLID principles!)

# =========================
# FILESYSTEM UTILITIES
# =========================
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
def get_hdu_list(filepath: str, memmap: bool=False) -> fits.HDUList:
    '''Gets an HDUList from a FITS file.

    Args:
        filepath (str): The path to the FITS file
        memmap (bool, optional): If True, the HDUList will be memory-mapped. Avoids loading the entire FITS file into RAM (for bigger files). Defaults to False. 

    Returns:
        fits.HDUList: Opened HDU list.
    '''
    path_obj = Path(filepath) # Convert to Path object for compatibility
    if not path_obj.is_file():
        raise Exception("File not found.")

    if not is_file_fits_format(filepath):
        raise Exception("Invalid file format. Please provide a FITS file.")

    hdul = fits.open(filepath, memmap=memmap)
    return hdul

def get_hdu_data(hdul: fits.HDUList, index: int=0, astype: str="float32") -> NDArray[np.floating]:
    '''Gets the data from an HDUList

    Args:
        hdul (fits.HDUList): Opened HDU list.
        index (int, optional): Index of the HDU. Defaults to 0.
        astype (str, optional): The data type to convert to. Defaults to "float32".

    Returns:
        ndarray: Data from the HDU
    '''
    return hdul[index].data.astype(getattr(np, astype))

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

def read_fits(filepath: str, index: int=0, apply_bscale: bool = True):
    '''Reads a FITS file and returns the data and header

    Args:
        filepath (str): The path to the FITS file
        index (int, optional): Index of the HDU. Defaults to 0.
        apply_bscale (bool, optional): If True, applies BSCALE and BZERO correction to the data. Defaults to True.

    Returns:
        tuple: A tuple containing the data and header
    '''
    hdul = get_hdu_list(filepath, memmap=True) # memmap for faster reading: avoids loading the entire FITS file into RAM (for bigger files)
    header = get_hdu_header(hdul, index) # Gets the header of the first HDU (metadata: exposure time, date, etc.)
    data = get_hdu_data(hdul, index, "float32") # Gets the data of the first HDU (float32: standard precision for FITS files)
    hdul.close() # Frees up memory

    # Applies BSCALE and BZERO correction if requested
    if apply_bscale:
       bscale = header.get("BSCALE", 1.0) # Retrieves the scaling factor. Defaults to 1.0
       bzero = header.get("BZERO", 0.0) # Retrieves the calibration offset. Defaults to 0.0
       data = data * bscale + bzero # Applies the formula: real value = raw value * scaling factor + calibration offset

    # In a tuple: (data, header)
    return data, header

# =========================
# NORMALIZATION/CONVERSION
# =========================
def simple_normalize(data: NDArray[np.floating]) -> NDArray[np.floating]:
    '''Simplified normalization: min-max normalization (no clipping, no percentile) because it made me go mad! >:(

    Args:
        data (ndarray): The data to normalize

    Returns:
        ndarray: The normalized data
    '''
    data = data.astype(np.float32)

    # Replaces NaN/Inf by 0 (to avoid invalid pixels)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Min-max based normalization
    data_min = data.min()
    data_max = data.max()

    # Avoid division by 0
    if data_max - data_min < 1e-10:
        return np.zeros_like(data)

    # Normalization
    normalized = (data - data_min) / (data_max - data_min)
    return normalized

def convert_fits_to_png(
    data: NDArray[np.floating], 
    bitdepth: int = 16 # Similar to the format of the FITS file (16-bit or 8-bit). 8-bit is better to reduce file size, but 16-bit is more accurate!
) -> NDArray:
    '''Converts a FITS image to a PNG image

    Args:
        data (ndarray): The FITS image
        bitdepth (int, optional): The bitdepth of the PNG image. Defaults to 16.

    Returns:
        ndarray: The PNG image
    '''
    # Converts C, H, W to H, W, C
    if data.ndim == 3 and data.shape[0] == 3:
        arr = np.transpose(data, (1, 2, 0)).astype(np.float32)
    else:
        arr = data.astype(np.float32) if data.ndim == 2 else data

    # Simple normalization per channel
    if arr.ndim == 2:
        # Monochrome
        normalized = simple_normalize(arr)
    else:
        # Colors: processes each channel separately
        normalized = np.zeros_like(arr, dtype=np.float32)
        for c in range(arr.shape[2]):
            normalized[..., c] = simple_normalize(arr[..., c])

    # Converts to png8 or png16
    scaled = normalized * (2 ** bitdepth - 1)
    img = scaled.astype(np.uint16 if bitdepth == 16 else np.uint8)

    # RGB to BGR for OpenCV
    if img.ndim == 3:
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
    return img

# =========================
# IMAGE SAVING
# =========================
def save_display_image(display_img: NDArray, filepath: str):
    '''Save a display-ready image as a PNG file

    Args:
        display_img (ndarray): The image to save (H, W, C) or (C, H, W)
        filepath (str): The path to save the image
    '''
    cv.imwrite(filepath, display_img)

# TODO: Make... it... work... :c
# TIFF is a 32-bit format (like FITS!) so it should be optimal for saving float images.
# Unfortunately it's not working in this beta..
def save_float_tiff(
    data: NDArray[np.floating], 
    filepath: str, 
    normalize: bool = True
):
    '''Save float data as TIFF 32-bit (with or without normalization)

    Args:
        data (ndarray): The data to save
        filepath (str): The path to save the TIFF
        normalize (bool, optional): Whether to normalize the data. Defaults to True.
    '''
    # Normalizes data shape to H, W, C
    if data.ndim == 3 and data.shape[0] == 3:
         arr = np.transpose(data, (1, 2, 0)).astype(np.float32)
    else:
        arr = data if data.ndim == 2 else data.astype(np.float32)

    if normalize:
        # Applies asinh and then normalize to [0, 1]
        if arr.ndim == 2:
            normalized = simple_normalize(arr)
        else:
            normalized = np.zeros_like(arr, dtype=np.float32)
            for c in range(arr.shape[2]):
                normalized[..., c] = simple_normalize(arr[..., c])
            # RGB to BGR for OpenCV
            if normalized.ndim == 3:
                normalized = normalized[..., ::-1]

        cv.imwrite(filepath, normalized.astype(np.float32))
    else:
        # Saves raw data
        if arr.ndim == 3:
            arr = arr[..., ::-1]
        cv.imwrite(filepath, arr.astype(np.float32))

def save_float_image(
    data: NDArray[np.floating], 
    filepath: str, 
    raw: bool = False, 
    format_type: Literal["png8", "png16"] = "png16", 
    normalize_tiff: bool = True
):
    '''Save a float image in various formats

    Args:
        data (ndarray): The image to save
        filepath (str): The path to save the image
        raw (bool, optional): Whether to save the image as a FITS file. Defaults to False. Is it really useful?
        format_type (Literal["png8", "png16"], optional): The format to save the image. Defaults to "png16".
        normalize_tiff (bool, optional): Whether to normalize the TIFF image. Defaults to True.
    '''
    # Reconstructs the FITS file
    if raw:
        fits.writeto(filepath.replace(".png", ".fits").replace(".tiff", ".fits").replace(".tif", ".fits"), data, overwrite=True)
        return

    # if format_type == "tiff":
    #     tiff_path = filepath.replace(".png", ".tiff")
    #     save_float_tiff(data, tiff_path, normalize=normalize_tiff)
    if format_type == "png8":
        img = convert_fits_to_png(data, bitdepth=8)
        save_display_image(img, filepath)
    else:  # png16
        img = convert_fits_to_png(data, bitdepth=16)
        save_display_image(img, filepath)

def save_combined_images(image1: str, image2: str, filepath: str) -> NDArray[np.uint8]:
    '''Save two images side by side for comparison

    Args:
        image1 (str): The first image to save
        image2 (str): The second image to save
        filepath (str): The path to save the combined image

    Returns:
        ndarray: The combined image
    '''
    # Loads images
    img1 = cv.imread(image1, cv.IMREAD_UNCHANGED)
    img2 = cv.imread(image2, cv.IMREAD_UNCHANGED)

    # Ensures same shape
    if img1.shape[0] != img2.shape[0]:
        h = min(img1.shape[0], img2.shape[0]) # Gets the smaller height
        # Resizes both images
        img1 = cv.resize(img1, (int(img1.shape[1] * h / img1.shape[0]), h))
        img2 = cv.resize(img2, (int(img2.shape[1] * h / img2.shape[0]), h))

    # Creates combined image
    combined = np.hstack((img1, img2)) # Horizontal stack
    cv.imwrite(filepath, combined)
    return combined

def save_gif(image1: str, image2: str, filepath: str, duration: int=500):
    '''Saves an animated GIF alternating between two images (blink comparison)

    Args:
        image1 (str): The first image to save
        image2 (str): The second image to save
        filepath (str): The path to save the GIF
        duration (int): The duration per frame in milliseconds
    '''
    try:
        from PIL import Image # Very useful library for working with images!
    except ImportError:
        raise ImportError("Pillow is required to save a GIF! Install with `pip install Pillow`.")

    # Loads images
    img1 = cv.imread(image1, cv.IMREAD_UNCHANGED)
    img2 = cv.imread(image2, cv.IMREAD_UNCHANGED)

    # Ensures same shape
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = cv.resize(img1, (w, h))
        img2 = cv.resize(img2, (w, h))
    
    # Converts BGR to RGB for PIL
    if img1.ndim == 3:
        img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
        img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
    else:
        img1_rgb = img1
        img2_rgb = img2

    # Converts to PIL Images
    # Handles 16-bit images by normalizing to 8-bit for GIF
    # Decreases quality.. because it is a GIF!
    if img1_rgb.dtype == np.uint16:
        img1_rgb = (img1_rgb / 256).astype(np.uint8)
        img2_rgb = (img2_rgb / 256).astype(np.uint8)

    pil_img1 = Image.fromarray(img1_rgb)
    pil_img2 = Image.fromarray(img2_rgb)

    # Creates animated GIF
    pil_img1.save(
        filepath, 
        save_all=True, 
        append_images=[pil_img2], 
        duration=duration, 
        loop=0
    )

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

def create_circular_kernel(radius) -> NDArray[np.uint8]:
    '''Creates a circular kernel of a given radius

    Args:
        radius (int): The radius of the kernel

    Returns:
        ndarray: The kernel
    '''
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x * x + y * y <= radius * radius
    return mask.astype(np.uint8)

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

def convert_to_luma(image: NDArray[np.floating]) -> NDArray[np.floating]:
    '''Converts an image to luma to keep masks apart from color processing

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
            # Fallback: mean over last axis
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

    # Avoids warnings
    if sources is None or len(sources) == 0:
        return None, median, std

    return sources, median, std

# TODO: Might not be the most efficient..
def build_binary_mask(
    shape: Tuple[int, int], 
    sources, 
    r_min: int=2, 
    r_max: int=6
):
    '''Builds a binary mask for stars

    Args:
        shape (tuple): The shape of the mask
        sources (list): The detected stars
        r_min (int, optional): The minimum radius of the stars. Defaults to 2.
        r_max (int, optional): The maximum radius of the stars. Defaults to 6.

    Returns:
    '''
    # Create empty mask with values of 0
    mask = np.zeros(shape, dtype=np.uint8)

    # If no stars detected, return an empty mask
    if sources is None or len(sources) == 0:
        return mask

    # Get flux values
    flux = sources["flux"]
    f_min, f_max = flux.min(), flux.max()

    for src in sources:
        x = int(round(src["xcentroid"]))
        y = int(round(src["ycentroid"]))

        f = src["flux"]
        f_norm = (f - f_min) / (f_max - f_min + 1e-8)

        # Circular radius
        radius = int(r_min + (r_max - r_min) * np.sqrt(f_norm))

        cv.circle(mask, (x, y), radius, 255, -1)
    return mask

def apply_gaussian_blur(mask: NDArray[np.uint8], sigma: float=1.0) -> NDArray[np.floating]:
    '''Applies a Gaussian blur to a mask

    Args:
        mask (ndarray): The mask to blur
        sigma (float, optional): The sigma value to use for the Gaussian blur. Defaults to 1.0.

    Returns:
        ndarray: The blurred mask
    '''
    mask_f = mask.astype(np.float32) / 255.0
    return cv.GaussianBlur(mask_f, ksize=(0, 0), sigmaX=sigma)

def get_starless_image_with_opening(image: NDArray[np.floating], kernel_radius: int=4, iterations: int=1) -> NDArray[np.floating]:
    '''Gets a starless image by using morphological opening. Morphological opening is a combination of erosion and dilation

    Args:
        image (ndarray): The image to get the starless image from
        kernel_radius (int, optional): The radius of the kernel to use for morphological opening. Defaults to 4.
        iterations (int, optional): The number of times to apply morphological opening. Defaults to 1.

    Returns:
        ndarray: The eroded image
    '''
    kernel = create_circular_kernel(kernel_radius)
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)

def get_starless_image_inpainting(image: NDArray[np.floating], mask: NDArray[np.uint8], method: str="telea") -> NDArray[np.floating]:
    '''Gets a starless image by using inpainting (way better! it gets rid of black holes)

    Args:
        image (ndarray): The image to get the starless image from
        mask (ndarray): The binary mask to use for inpainting
        method (str, optional): The method to use for inpainting. Defaults to "telea".

    - telea: Uses cv.inpaint with INPAINT_TELEA flag
    - ns: Uses cv.inpaint with INPAINT_NS flag

    Returns:
        ndarray: The inpainted image
    '''
    # Normalizes data to uint8 for cv.inpaint
    img_norm = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

    # Applies inpainting with specified method
    inpaint_method = cv.INPAINT_TELEA if method == "telea" else cv.INPAINT_NS
    inpainted = cv.inpaint(img_norm, mask, inpaintRadius=3, flags=inpaint_method)

    # Reconverts to float32 with original scale
    result = inpainted.astype(np.float32) / 255.0 * (image.max() - image.min()) + image.min()
    return result

def get_starless_image(
    image: NDArray[np.floating], 
    mask: Optional[NDArray[np.uint8]] = None, 
    kernel_radius: int=4, 
    iterations: int=1, 
    method: Literal["opening", "inpainting"] = "opening"
) -> NDArray[np.floating]:
    '''Gets a starless image using chosen method

    Args:
        image (ndarray): The image to get the starless image from
        mask (ndarray, optional): The binary mask to use for inpainting. Defaults to None.
        kernel_radius (int, optional): The radius of the kernel to use for morphological opening. Defaults to 4.
        iterations (int, optional): The number of times to apply morphological opening. Defaults to 1.
        method (Literal["opening", "inpainting"], optional): The method to use for getting the starless image. Defaults to "opening".

    Returns:
        ndarray: The starless image
    '''
    if method == "inpainting":
        # inpainting method requires a mask
        if mask is None:
            raise ValueError("Inpainting requires a binary mask")
        return get_starless_image_inpainting(image, mask, method="telea")
    else:
        return get_starless_image_with_opening(image, kernel_radius, iterations)

# =========================
# TILE MULTI-PROCESSING
# =========================
def calculate_tiles(image_shape: Tuple[int, int], tile_size: int = 128, overlap: int = 64) -> List[Tuple[int, int, int, int]]:
    '''Calculates tile boundaries with overlap

    - Overlap is used to avoid edge artifacts. Each tile is processed separately and then blended together. If a star is close to the edge of a tile, it might be missed or unevenly processed!
    - Each tile is defined by: (y_start, y_end, x_start, x_end)

    Args:
        image_shape: (height, width) of the image
        tile_size: Size of each tile (default 128x128)
        overlap: Overlap between tiles to avoid edge artifacts (default 64px)

    Returns:
        List of tuples (y_start, y_end, x_start, x_end) for each tile
    '''
    # Extracts height and width from image (it is inverted)
    h, w = image_shape
    # List to contain all tiles coordinates
    tiles: List[Tuple[int, int, int, int]] = []

    # Vertical position
    y = 0
    while y < h:
        # End of the tile
        y_end = min(y + tile_size, h)

        # Horizontal position
        x = 0
        while x < w:
            # End of the tile
            x_end = min(x + tile_size, w)

            # Stores the tile boudaries
            tiles.append((y, y_end, x, x_end))

            # Adds overlap (to the right)
            x += tile_size - overlap

            # Stops if right edge exceeds width
            if x_end == w:
                break

        # Overlaps down
        y += tile_size - overlap

        # Stops if bottom edge exceeds height
        if y_end == h:
            break

    return tiles

def process_tile(args):
    '''Process one tile using the StarEX algorithm

    - Executed in parallel by multiprocessing. Amazing!

    Args:
        args: Tuple of (tile_bounds, image_data, algorithm_params, output_type)
  
    Returns:
        Tuple of (tile_bounds, processed_data)
    '''
    # Unpacks arguments
    tile_bounds, image_data, algorithm_params, output_type = args
    y_start, y_end, x_start, x_end = tile_bounds

    # Imports algorithm locally (required for multiprocessing)
    from algorithm import StarEX

    # Extracts tile
    if image_data.ndim == 3: # Color image (C, H, W)
        tile = image_data[:, y_start:y_end, x_start:x_end].copy()
    else: # Monochrome image (H, W)
        tile = image_data[y_start:y_end, x_start:x_end].copy()

    # Creates the StarEX algorithm instance with given parameters
    algorithm = StarEX(**algorithm_params)

    # Runs the algorithm on the tile
    result = algorithm.run(tile, output_type=output_type)

    # Returns tile boundaries and processed data
    return (tile_bounds, result[output_type])

def blend_tiles(output_shape: Tuple[int, ...], tile_results: List[Tuple[Tuple[int, int, int, int], NDArray]], overlap: int = 64) -> NDArray:
    '''Blend overlapping tiles using linear interpolation

    Args:
        output_shape: Shape of the final output image
        tile_results: List of (tile_bounds, processed_tile_data)
        overlap: Overlap size used during tiling

    Returns:
        Blended output image
    '''
    # Initializes output image and weight map
    if len(output_shape) == 3: # Color image
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape[1:], dtype=np.float32) # H, W
    else: # Monochrome image
        output = np.zeros(output_shape, dtype=np.float32)
        weights = np.zeros(output_shape, dtype=np.float32)

    # Processes each tile
    for tile_bounds, tile_data in tile_results:
        y_start, y_end, x_start, x_end = tile_bounds
        tile_h = y_end - y_start
        tile_w = x_end - x_start

        # Creates blending weight mask (linear ramp near edges)
        weight_mask = np.ones((tile_h, tile_w), dtype=np.float32)

        # Applies linear ramp on overlapping edges
        if overlap > 0:
            # Top edge
            if y_start > 0:
                ramp_h = min(overlap, tile_h)
                ramp = np.linspace(0, 1, ramp_h).reshape(-1, 1)
                weight_mask[:ramp_h, :] *= ramp

            # Bottom edge
            if y_end < output_shape[-2]:
                ramp_h = min(overlap, tile_h)
                ramp = np.linspace(1, 0, ramp_h).reshape(-1, 1)
                weight_mask[-ramp_h:, :] *= ramp

            # Left edge
            if x_start > 0:
                ramp_w = min(overlap, tile_w)
                ramp = np.linspace(0, 1, ramp_w).reshape(1, -1)
                weight_mask[:, :ramp_w] *= ramp

            # Right edge
            if x_end < output_shape[-1]:
                ramp_w = min(overlap, tile_w)
                ramp = np.linspace(1, 0, ramp_w).reshape(1, -1)
                weight_mask[:, -ramp_w:] *= ramp

        # Adds weighted tile to output
        if len(output_shape) == 3: # Color
            for c in range(output_shape[0]):
                output[c, y_start:y_end, x_start:x_end] += tile_data[c] * weight_mask
            weights[y_start:y_end, x_start:x_end] += weight_mask
        else: # Monochrome
            output[y_start:y_end, x_start:x_end] += tile_data * weight_mask
            weights[y_start:y_end, x_start:x_end] += weight_mask

    # Normalizes by weights
    weights = np.maximum(weights, 1e-8) # Avoids division by zero
    if len(output_shape) == 3:
        for c in range(output_shape[0]):
            output[c] = output[c] / weights
    else:
        output = output / weights

    return output

def process_image_tiled(
    image_data: NDArray, 
    algorithm_params: dict, 
    output_type: str, 
    tile_size: int = 128, 
    overlap: int = 64, 
    nb_workers: int = None
) -> NDArray:
    '''Process large image using tiling and multiprocessing

    The image is:
    1. Divided into tiles
    2. Each tile is processed separately
    3. The processed tiles are blended together

    Args:
        image_data: Input FITS image (H,W) or (C,H,W)
        algorithm_params: Dictionary of StarEX parameters
        output_type: Type of output to generate
        tile_size: Size of tiles (default 128)
        overlap: Overlap between tiles (default 64)
        nb_workers: Number of worker processes (default: CPU count)

    Returns:
        Processed image
    '''
    # Uses CPU count if nb_workers is not specified
    if nb_workers is None:
        nb_workers = mp.cpu_count()

    # Gets image shape
    if image_data.ndim == 3:
        image_shape = image_data.shape[1:] # (H, W)
    else:
        image_shape = image_data.shape

    # Calculates tile boundaries
    tiles = calculate_tiles(image_shape, tile_size, overlap)

    # Prepares arguments for worker processes
    worker_args = [(tile_bounds, image_data, algorithm_params, output_type) for tile_bounds in tiles]

    # Processes tiles in parallel
    with mp.Pool(processes=nb_workers) as pool:
        tile_results = pool.map(process_tile, worker_args)

    # Blends tiles together!
    output = blend_tiles(image_data.shape, tile_results, overlap)

    return output # Final processed image ;)

# =========================
# TESTING PLAYGROUND YAY!
# =========================
if __name__ == "__main__":
    pass