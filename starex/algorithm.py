import numpy as np
import utils as utils

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
        kernel_radius=4 # Between 4 and 6
    ):
        '''Initializes StarEX algorithm with default values

        Args:
            fwhm (float, optional):
                Full Width at Half Maximum: the apparent size of the star (in pixels).
            threshold_sigma (float, optional):
                Detection threshold: the higher the value, the less stars will be detected
            r_min (int, optional): Minimum radius of mask
            r_max (int, optional): Maximum radius of mask
            mask_blur_sigma (float, optional): Avoids noise in the mask
            reduction_strength (float, optional): Reduction strength
            kernel_radius (int, optional): Kernel radius
        '''
        self.fwhm = fwhm
        self.threshold_sigma = threshold_sigma
        self.r_min = r_min
        self.r_max = r_max
        self.mask_blur_sigma = mask_blur_sigma
        self.reduction_strength = reduction_strength
        self.kernel_radius = kernel_radius

    def run(self, fits_data):
        # 1. Prepares data
        # float32: optimized for OpenCV and more precision. It is also better for image processing performance
        if fits_data.ndim == 3:
            # For each channel, a mean is calculated
            image = np.mean(fits_data, axis=0).astype(np.float32)
        else:
            image = fits_data.astype(np.float32)

        # 2. Detects stars
        # Detecting the stars is necessary to build the binary mask. It will allow us to reduce the stars locally
        sources, median, std = utils.detect_stars_dao(
            image, 
            fwhm=self.fwhm, 
            threshold_sigma=self.threshold_sigma
        )

        # 3. Binary mask
        mask = utils.build_binary_mask(
            image.shape, 
            sources, 
            r_min=self.r_min, 
            r_max=self.r_max
        )

        # 4. Blurs mask (M)
        # Blurring the mask allows us to reduce the noise locally by interpolating the edges and having smooth transitions
        M = utils.apply_gaussian_blur(mask, sigma=self.mask_blur_sigma)

        # 5. Eroded image (Ierode)
        # It is better to use morphological opening than simple erosion. It avoids the problem of "holes" in the mask
        Ierode = utils.get_starless_image(
            image,
            kernel_radius=self.kernel_radius
        )

        # 6. Reduces image
        # This formula allows us to reduce the noise locally and the stars globally
        # Ifinal = (M * Ierode) + ((1.0 - M) * image)
        Ifinal = ((self.reduction_strength * M) * Ierode) + ((1.0 - self.reduction_strength * M) * image)

        # 7. Returns results as images (apart from sources)
        return {
            'original': image, 
            'binary_mask': mask, 
            'gaussian_mask': M, 
            'eroded': Ierode, 
            'reduced': Ifinal, 
            'sources': sources
        }