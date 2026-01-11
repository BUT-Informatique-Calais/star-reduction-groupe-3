# Third-party imports
import numpy as np
import cv2 as cv
from numpy.typing import NDArray

# Typing
from typing import Self, Literal, Dict, List, Tuple, Optional

# Local imports
import utils

# Available output types
OUTPUT_TYPES = Literal[
    "original", 
    "binary_mask", 
    "gaussian_mask", 
    "eroded", 
    "reduced", 
    "sources", 
    "all"
]

# TODO: Add unit tests?
class StarEX:
    '''StarEX : Star EXtraction (b1.2.1)'''

    def __init__(
        self, 
        fwhm=3.0, # Between 2.5 and 3.0
        threshold_sigma=2.5, # Between 2.5 and 3.0
        r_min=2, 
        r_max=8, 
        blur_strength=2.5, # Between 1.5 and 3.0
        reduction_strength=0.65, # Between 0.5 and 0.7 (1.0 is maximum reduction)
        kernel_radius=5, # Between 4 and 6
        iterations=1, 
        starless_method: Literal["opening", "inpainting"] = "opening", 
        multiscale: bool = False
    ) -> None:
        '''Initializes StarEX algorithm with default values

        Args:
            fwhm (float): Full Width at Half Maximum (2.5-3.5)
            threshold_sigma (float): Detection threshold (2.5-3.0)
            r_min (int): Minimum radius of mask
            r_max (int): Maximum radius of mask
            blur_strength (float): Gaussian blur sigma (1.5-3.0)
            reduction_strength (float): Star reduction strength (0.5-0.8)
            kernel_radius (int): Morphological kernel radius (4-6)
            iterations (int): Number of morphological iterations
            starless_method (str): Method for starless image ("opening" or "inpainting")
            multiscale (bool): Use multiscale eroded image with adaptive kernel
        '''
        self.fwhm = fwhm
        self.threshold_sigma = threshold_sigma
        self.r_min = r_min
        self.r_max = r_max
        self.blur_strength = blur_strength
        self.reduction_strength = reduction_strength
        self.kernel_radius = kernel_radius
        self.iterations = iterations
        self.starless_method = starless_method
        self.multiscale = multiscale

    # =========================
    # CORE STEPS
    # =========================
    def get_luminance(
        self: Self, 
        fits_data: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], bool, NDArray[np.floating] | None]:
        '''Extracts luminance from image data

        Args:
            fits_data (ndarray): The FITS image

        Returns:
            ndarray: The luminance image
        '''
        has_color = utils.has_color_channels(fits_data)
        luma = utils.convert_to_luma(fits_data) # Super Mario Galaxy reference
        colored = fits_data.astype(np.float32) if has_color else None
        return luma, has_color, colored

    def detect_sources(
        self: Self, 
        luma: NDArray[np.floating]
    ) -> List:
        '''Detects star sources in the luminance image

        Args:
            luma (ndarray): The luminance image

        Returns:
            list: List of detected sources
        '''
        sources, _, _ = utils.detect_stars_dao(
            luma, 
            fwhm=self.fwhm, 
            threshold_sigma=self.threshold_sigma
        )
        return sources

    def create_binary_mask(
        self: Self, 
        shape: Tuple[int, int], 
        sources: List
    ) -> NDArray[np.uint8]:
        '''Creates binary mask from detected sources

        Args:
            shape (tuple): The shape of the mask
            sources (list): The detected sources

        Returns:
            ndarray: The binary mask
        '''
        mask = utils.build_binary_mask(
            shape, 
            sources, 
            r_min=self.r_min, 
            r_max=self.r_max
        )
        return mask

    def create_gaussian_mask(
        self: Self, 
        binary_mask: NDArray[np.uint8]
    ) -> NDArray[np.floating]:
        '''Applies Gaussian blur to binary mask for smooth transitions
        
        Args:
            binary_mask (ndarray): The binary mask to blur

        Returns:
            ndarray: The blurred binary mask
        '''
        gaussian_mask = utils.apply_gaussian_blur(
            binary_mask, 
            sigma=self.blur_strength
        )
        return gaussian_mask

    def create_eroded_image(
        self: Self, 
        image: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.floating]:
        '''Creates starless (eroded) version of image

        Args:
            image (ndarray): The luminance image
            binary_mask (ndarray): The binary mask

        Returns:
            ndarray: The starless image
        '''
        eroded = utils.get_starless_image(
            image, 
            mask=binary_mask, 
            kernel_radius=self.kernel_radius, 
            iterations=self.iterations, # i forgor 💀
            method=self.starless_method # i forgor that too 💀
        )
        return eroded

    def create_multiscale_eroded_image(
        self: Self, 
        image: NDArray[np.floating], 
        sources: List, 
        binary_mask: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.floating]:
        '''Creates starless (eroded) version of image using adaptative kernel sizes per star magnitude

        Args:
            image (ndarray): The luminance image
            sources (list): The detected sources
            binary_mask (ndarray): The binary mask

        Returns:
            ndarray: The starless image
        '''
        # If no stars detected, return the original image
        if sources is None or len(sources) == 0:
            return image.copy()

        # Get flux values
        flux = sources["flux"]
        f_min, f_max = flux.min(), flux.max()

        # 3 categories: small, medium, large
        flux_range = f_max - f_min
        threshold_small = f_min + 0.33 * flux_range
        threshold_large = f_min + 0.66 * flux_range

        # 3 individual masks
        small_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        medium_mask = np.zeros_like(binary_mask, dtype=np.uint8)
        large_mask = np.zeros_like(binary_mask, dtype=np.uint8)

        for src in sources:
            x = int(round(src["xcentroid"]))
            y = int(round(src["ycentroid"]))

            f = src["flux"]

            f_norm = (f - f_min) / (flux_range + 1e-8)

            radius = int(self.r_min + (self.r_max - self.r_min) * np.sqrt(f_norm))

            # Gets the star category
            if f < threshold_small:
                cv.circle(small_mask, (x, y), radius, 255, -1)
            elif f < threshold_large:
                cv.circle(medium_mask, (x, y), radius, 255, -1)
            else:
                cv.circle(large_mask, (x, y), radius, 255, -1)

        # Processes each category separetely
        if self.starless_method == "inpainting":
            eroded_small = utils.get_starless_image(image, small_mask, method="inpainting")
            eroded_medium = utils.get_starless_image(image, medium_mask, method="inpainting")
            eroded_large = utils.get_starless_image(image, large_mask, method="inpainting")
        else:
            eroded_small = utils.get_starless_image(image, small_mask, method="opening")
            eroded_medium = utils.get_starless_image(image, medium_mask, method="opening")
            eroded_large = utils.get_starless_image(image, large_mask, method="opening")

        result = image.copy() # Combines all categories

        # Normalizes each mask
        small_mask_norm = small_mask.astype(np.float32) / 255.0
        medium_mask_norm = medium_mask.astype(np.float32) / 255.0
        large_mask_norm = large_mask.astype(np.float32) / 255.0

        # Applies erosion formula
        result = result * (1.0 - large_mask_norm) + eroded_large * large_mask_norm
        result = result * (1.0 - medium_mask_norm) + eroded_medium * medium_mask_norm
        result = result * (1.0 - small_mask_norm) + eroded_small * small_mask_norm

        return result

    def reduce_stars_single_channel(
        self: Self, 
        channel: NDArray[np.floating], 
        gaussian_mask: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None, 
        sources: Optional[List] = None
    ) -> NDArray[np.floating]:
        '''Reduces stars in a single channel

        Args:
            channel (ndarray): The channel to reduce
            gaussian_mask (ndarray): The gaussian mask
            binary_mask (ndarray): The binary mask
            sources (list): The detected sources

        Returns:
            ndarray: The reduced image
        '''
        if self.multiscale and sources is not None:
            eroded = self.create_multiscale_eroded_image(channel, sources, binary_mask)
        else:
            eroded = self.create_eroded_image(channel, binary_mask)

        # Applies reduction formula
        reduced = (
            (self.reduction_strength * gaussian_mask) * eroded
            + (1.0 - self.reduction_strength * gaussian_mask) * channel
        )
        return reduced

    def reduce_stars_multi_channel(
        self: Self, 
        colored: NDArray[np.floating], 
        gaussian_mask: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None, 
        sources: Optional[List] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        '''Reduces stars in multi-channel (color) image

        Args:
            colored (ndarray): The color image
            gaussian_mask (ndarray): The gaussian mask
            binary_mask (ndarray): The binary mask
            sources (list): The detected sources

        Returns:
            ndarray: The reduced image
        '''
        reduced_channels = []
        eroded_channels = []

        for c in range(colored.shape[0]):
            channel = colored[c]

            if self.multiscale and sources is not None:
                eroded_c = self.create_multiscale_eroded_image(channel, sources, binary_mask)
            else:
                eroded_c = self.create_eroded_image(channel, binary_mask)

            reduced_c = (
                (self.reduction_strength * gaussian_mask) * eroded_c
                + (1.0 - self.reduction_strength * gaussian_mask) * channel
            )

            reduced_channels.append(reduced_c)
            eroded_channels.append(eroded_c)

        # Puts channels back
        reduced = np.stack(reduced_channels, axis=0)
        eroded = np.stack(eroded_channels, axis=0)
        
        return reduced, eroded

    def run(
        self: Self, 
        fits_data: NDArray[np.floating], 
        output_type: OUTPUT_TYPES
    ) -> Dict[str, NDArray[np.floating] | List]:
        '''Runs the StarEX algorithm

        Args:
            fits_data (ndarray): The FITS image to process. Shape: (H, W) or (C, H, W)
            output_type (str): Type of output to generate

        Returns:
            dict: A dictionary containing only the required output
        '''
        # Stops the algorithm if the type is not valid
        if output_type not in OUTPUT_TYPES.__args__:
            raise ValueError("Type must be 'original', 'binary_mask', 'gaussian_mask', 'eroded', 'reduced', 'sources' or 'all'")

        has_color: bool = utils.has_color_channels(fits_data)

        # =========================
        # ORIGINAL IMAGE
        # =========================
        if output_type == "original":
            return {"original": fits_data.astype(np.float32)}

        # =========================
        # LUMINANCE
        # =========================
        luma, has_color, colored = self.get_luminance(fits_data)

        # =========================
        # SOURCES
        # =========================
        sources = self.detect_sources(luma)

        if output_type == "sources":
            return {"sources": sources}

        # =========================
        # BINARY MASK
        # =========================
        binary_mask = self.create_binary_mask(luma.shape, sources)

        # =========================
        # BINARY MASK
        # =========================
        if output_type == "binary_mask":
            return {"binary_mask": binary_mask}

        # =========================
        # GAUSSIAN MASK
        # =========================
        gaussian_mask = self.create_gaussian_mask(binary_mask)

        if output_type == "gaussian_mask":
            return {"gaussian_mask": gaussian_mask}

        # =========================
        # ERODED
        # =========================
        if not has_color:
            eroded = self.create_eroded_image(luma, binary_mask)

            # =========================
            # ERODED
            # =========================
            if output_type == "eroded":
                return {"eroded": eroded}

            # =========================
            # REDUCED
            # =========================
            reduced = self.reduce_stars_single_channel(luma, gaussian_mask, binary_mask, sources)
            return {"reduced": reduced}

        reduced, eroded = self.reduce_stars_multi_channel(colored, gaussian_mask, binary_mask, sources)

        if output_type == "eroded":
            return {"eroded": eroded}

        if output_type == "reduced":
            return {"reduced": reduced}

        # Error if output type is not handled
        raise RuntimeError(f"Unhandled output type: {output_type}")