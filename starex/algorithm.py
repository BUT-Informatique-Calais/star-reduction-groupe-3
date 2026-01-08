import numpy as np
from numpy.typing import NDArray
import utils
from typing import Self, Literal, Dict, List, Tuple

OUTPUT_TYPES = Literal[
    "original",
    "binary_mask",
    "gaussian_mask",
    "eroded",
    "reduced",
    "sources",
]

# TODO: Add docstrings and explicit types
class StarEX:
    '''StarEX : Star EXtraction (beta)'''
    def __init__(
        self, 
        fwhm=3.0, # Between 2.5 and 3.0
        threshold_sigma=2.5, # Between 2.5 and 3.0
        r_min=2, 
        r_max=8, 
        mask_blur_sigma=2.5, # Between 1.5 and 3.0
        reduction_strength=0.7, # Between 0.5 and 0.7 (1.0 is maximum reduction)
        kernel_radius=4, # Between 4 and 6
        iterations=1
    ) -> None:
        '''Initializes StarEX algorithm with default values

        Args:reshold_sigma (float, optional):
                Detection threshold: the higher the value, the less stars will be detected
            r_min (int, optional): Minimum radius of mask
            r_max
            fwhm (float, optional):
                Full Width at Half Maximum: the apparent size of the star (in pixels).
            th (int, optional): Maximum radius of mask
            mask_blur_sigma (float, optional): Avoids noise in the mask
            reduction_strength (float, optional): Reduction strength
            kernel_radius (int, optional): Kernel radius
            iterations (int, optional): Number of iterations
        '''
        self.fwhm = fwhm
        self.threshold_sigma = threshold_sigma
        self.r_min = r_min
        self.r_max = r_max
        self.mask_blur_sigma = mask_blur_sigma
        self.reduction_strength = reduction_strength
        self.kernel_radius = kernel_radius
        self.iterations = iterations

    def get_luminance(
        self, 
        fits_data: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], bool, NDArray[np.floating] | None]:
        '''Extracts luminance from image data
        
        Args:
            fits_data: Input image data (H, W) or (C, H, W)
            
        Returns:
            Tuple containing:
                - luma: Luminance channel
                - has_color: Boolean indicating if image has color
                - colored: Original colored data or None
        '''
        has_color = utils.has_color_channels(fits_data)
        luma = utils.convert2luma(fits_data)
        colored = fits_data.astype(np.float32) if has_color else None
        return luma, has_color, colored

    def detect_sources(
        self, 
        luma: NDArray[np.floating]
    ) -> List:
        '''Detects star sources in the luminance image
        
        Args:
            luma: Luminance channel
            
        Returns:
            List of detected star sources
        '''
        sources, _, _ = utils.detect_stars_dao(
            luma, 
            fwhm=self.fwhm, 
            threshold_sigma=self.threshold_sigma
        )
        return sources

    def create_binary_mask(
        self, 
        image_shape: Tuple[int, int], 
        sources: List
    ) -> NDArray[np.floating]:
        '''Creates binary mask from detected sources
        
        Args:
            image_shape: Shape of the image (H, W)
            sources: List of detected star sources
            
        Returns:
            Binary mask array
        '''
        mask = utils.build_binary_mask(
            image_shape, 
            sources, 
            r_min=self.r_min, 
            r_max=self.r_max
        )
        return mask

    def create_gaussian_mask(
        self, 
        binary_mask: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        '''Applies Gaussian blur to binary mask for smooth transitions
        
        Args:
            binary_mask: Binary mask array
            
        Returns:
            Gaussian blurred mask
        '''
        gaussian_mask = utils.apply_gaussian_blur(
            binary_mask, 
            sigma=self.mask_blur_sigma
        )
        return gaussian_mask

    def create_eroded_image(
        self, 
        image: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        '''Creates starless (eroded) version of image
        
        Args:
            image: Input image channel
            
        Returns:
            Eroded image with stars removed
        '''
        eroded = utils.get_starless_image(
            image,
            kernel_radius=self.kernel_radius
        )
        return eroded

    def reduce_stars_single_channel(
        self,
        channel: NDArray[np.floating],
        gaussian_mask: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        '''Reduces stars in a single channel
        
        Args:
            channel: Image channel to process
            gaussian_mask: Gaussian mask for blending
            
        Returns:
            Channel with reduced stars
        '''
        eroded = self.create_eroded_image(channel)
        reduced = (
            (self.reduction_strength * gaussian_mask) * eroded
            + (1.0 - self.reduction_strength * gaussian_mask) * channel
        )
        return reduced

    def reduce_stars_multi_channel(
        self,
        colored: NDArray[np.floating],
        gaussian_mask: NDArray[np.floating]
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        '''Reduces stars in multi-channel (color) image
        
        Args:
            colored: Multi-channel image (C, H, W)
            gaussian_mask: Gaussian mask for blending
            
        Returns:
            Tuple containing:
                - reduced: Image with reduced stars
                - eroded: Starless image
        '''
        reduced_channels = []
        eroded_channels = []

        for c in range(colored.shape[0]):
            print(colored.shape[0])
            channel = colored[c]
            
            eroded_c = self.create_eroded_image(channel)
            reduced_c = (
                (self.reduction_strength * gaussian_mask) * eroded_c
                + (1.0 - self.reduction_strength * gaussian_mask) * channel
            )
            
            reduced_channels.append(reduced_c)
            eroded_channels.append(eroded_c)

        # Puts channels back
        reduced = np.stack(reduced_channels, axis=0)
        print(f"Reduced: {reduced.shape}")
        
        eroded = np.stack(eroded_channels, axis=0)
        print(f"Eroded: {eroded.shape}")
        
        return reduced, eroded

    def run(
        self: Self, 
        fits_data: NDArray[np.floating], 
        output_type: OUTPUT_TYPES
    ) -> Dict[str, NDArray[np.floating] | List]:
        '''Runs the StarEX algorithm

        Args:
            fits_data (ndarray): The FITS image to process. Shape can be (H, W) or (C, H, W) in case of color image

        Returns:
            dict: A dictionary containing only the required output
        '''
        # Stops the algorithm if the type is not valid
        if output_type not in (
            "original",
            "binary_mask",
            "gaussian_mask",
            "eroded",
            "reduced",
            "sources",
        ):
            raise ValueError("Type must be 'original', 'binary_mask', 'gaussian_mask', 'eroded', 'reduced' or 'sources'")

        has_color: bool = utils.has_color_channels(fits_data)

        # 1. Prepares data
        # =========================
        # ORIGINAL IMAGE
        # =========================
         # float32: optimized for OpenCV and more precision. It is also better for image processing performance
        if output_type == "original":
            return {"original": fits_data.astype(np.float32)}

        # =========================
        # LUMINANCE
        # =========================
        luma: NDArray[np.floating] = utils.convert2luma(fits_data)
        colored: NDArray[np.floating] = fits_data.astype(np.float32) if has_color else None

        # 2. Detects stars
        # Detecting the stars is necessary to build the binary mask. It will allow us to reduce the stars locally
        sources, _, _ = utils.detect_stars_dao(
            luma, 
            fwhm=self.fwhm, 
            threshold_sigma=self.threshold_sigma
        )

        # =========================
        # SOURCES
        # =========================
        if output_type == "sources":
            return {"sources": sources}

        # 3. Binary mask
        mask = utils.build_binary_mask(
            luma.shape, 
            sources, 
            r_min=self.r_min, 
            r_max=self.r_max
        )

        # =========================
        # BINARY MASK
        # =========================
        if output_type == "binary_mask":
            return {"binary_mask": mask}

        # 4. Gaussian mask (M)
        # Blurring the mask (with a Gaussian blur) allows us to reduce the noise locally by interpolating the edges and having smooth transitions
        M = utils.apply_gaussian_blur(
            mask, 
            sigma=self.mask_blur_sigma
        )

        # =========================
        # GAUSSIAN MASK
        # =========================
        if output_type == "gaussian_mask":
            return {"gaussian_mask": M}

        # 5. Eroded image (Ierode)
        # It is better to use morphological opening than simple erosion. It avoids the problem of "holes" in the mask
        if not has_color:
            eroded = utils.get_starless_image(
                luma,
                kernel_radius=self.kernel_radius
            )

            # =========================
            # ERODED
            # =========================
            if output_type == "eroded":
                return {"eroded": eroded}

            # 6. Reduces image
            # This formula allows us to reduce the noise locally and the stars globally
            # Ifinal = (M * Ierode) + ((1.0 - M) * image)
            reduced = (
                (self.reduction_strength * M) * eroded
                + (1.0 - self.reduction_strength * M) * luma
            )

            # =========================
            # REDUCED
            # =========================
            return {"reduced": reduced}

        reduced_channels = []
        eroded_channels = []

        for c in range(colored.shape[0]):
            print(colored.shape[0])
            channel = colored[c]

            Ierode_c = utils.get_starless_image(
                channel, 
                kernel_radius=self.kernel_radius
            )
            
            Ifinal_c = ((self.reduction_strength * M) * Ierode_c) + ((1.0 - self.reduction_strength * M) * channel)

            reduced_channels.append(Ifinal_c)
            eroded_channels.append(Ierode_c)

        # Puts channels back
        reduced = np.stack(reduced_channels, axis=0)
        print(f"Reduced: {reduced.shape}")

        if output_type == "reduced":
            return {"reduced": reduced}

        eroded = np.stack(eroded_channels, axis=0)
        print(f"Eroded: {eroded.shape}")

        if output_type == "eroded":
            return {"eroded": eroded}

        # Error if output type is not handled
        raise RuntimeError(f"Unhandled output type: {output_type}")