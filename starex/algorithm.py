# Third-party imports
import numpy as np
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

class StarEX:
    '''StarEX : Star EXtraction (1.2.0)'''

    def __init__(
        self, 
        fwhm=3.0, # Between 2.5 and 3.0
        threshold_sigma=2.5, # Between 2.5 and 3.0
        r_min=2, 
        r_max=8, 
        mask_blur_sigma=2.5, # Between 1.5 and 3.0
        reduction_strength=0.65, # Between 0.5 and 0.7 (1.0 is maximum reduction)
        kernel_radius=5, # Between 4 and 6
        iterations=1, 
        starless_method: Literal["opening", "inpainting"] = "opening"
    ) -> None:
        '''Initializes StarEX algorithm with default values

        Args:
            fwhm (float): Full Width at Half Maximum (2.5-3.5)
            threshold_sigma (float): Detection threshold (2.5-3.0)
            r_min (int): Minimum radius of mask
            r_max (int): Maximum radius of mask
            mask_blur_sigma (float): Gaussian blur sigma (1.5-3.0)
            reduction_strength (float): Star reduction strength (0.5-0.8)
            kernel_radius (int): Morphological kernel radius (4-6)
            iterations (int): Number of morphological iterations
            starless_method (str): Method for starless image ("opening" or "inpainting")
        '''
        self.fwhm = fwhm
        self.threshold_sigma = threshold_sigma
        self.r_min = r_min
        self.r_max = r_max
        self.mask_blur_sigma = mask_blur_sigma
        self.reduction_strength = reduction_strength
        self.kernel_radius = kernel_radius
        self.iterations = iterations
        self.starless_method = starless_method

    # =========================
    # CORE STEPS
    # =========================
    def get_luminance(
        self: Self, 
        fits_data: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], bool, NDArray[np.floating] | None]:
        '''Extracts luminance from image data'''
        has_color = utils.has_color_channels(fits_data)
        luma = utils.convert_to_luma(fits_data)
        colored = fits_data.astype(np.float32) if has_color else None
        return luma, has_color, colored

    def detect_sources(
        self: Self, 
        luma: NDArray[np.floating]
    ) -> List:
        '''Detects star sources in the luminance image'''
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
        '''Creates binary mask from detected sources'''
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
        '''Applies Gaussian blur to binary mask for smooth transitions'''
        gaussian_mask = utils.apply_gaussian_blur(
            binary_mask, 
            sigma=self.mask_blur_sigma
        )
        return gaussian_mask

    def create_eroded_image(
        self: Self, 
        image: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.floating]:
        '''Creates starless (eroded) version of image'''
        eroded = utils.get_starless_image(
            image, 
            mask=binary_mask, 
            kernel_radius=self.kernel_radius, 
            iterations=self.iterations, # i forgor 💀
            method=self.starless_method # i forgor that too 💀
        )
        return eroded

    def reduce_stars_single_channel(
        self: Self, 
        channel: NDArray[np.floating], 
        gaussian_mask: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None
    ) -> NDArray[np.floating]:
        '''Reduces stars in a single channel'''
        eroded = self.create_eroded_image(channel, binary_mask)

        reduced = (
            (self.reduction_strength * gaussian_mask) * eroded
            + (1.0 - self.reduction_strength * gaussian_mask) * channel
        )
        return reduced

    def reduce_stars_multi_channel(
        self: Self, 
        colored: NDArray[np.floating], 
        gaussian_mask: NDArray[np.floating], 
        binary_mask: Optional[NDArray[np.uint8]] = None
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        '''Reduces stars in multi-channel (color) image'''
        reduced_channels = []
        eroded_channels = []

        for c in range(colored.shape[0]):
            channel = colored[c]

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
            reduced = self.reduce_stars_single_channel(luma, gaussian_mask, binary_mask)
            return {"reduced": reduced}

        reduced, eroded = self.reduce_stars_multi_channel(colored, gaussian_mask, binary_mask)

        if output_type == "eroded":
            return {"eroded": eroded}

        if output_type == "reduced":
            return {"reduced": reduced}

        # Error if output type is not handled
        raise RuntimeError(f"Unhandled output type: {output_type}")