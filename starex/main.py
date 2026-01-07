import sys, os
import cv2 as cv
import utils as utils
from algorithm import StarEX

# TODO: Add docstrings and explicit types
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[HOW] Correct syntax is: python main.py <image.fits>")
        sys.exit(1)

    filepath = sys.argv[1]
    results_dir = "results"
    os.makedirs('results', exist_ok=True)

    # Check if file exists
    if not utils.does_file_exist(filepath):
        print("[ERR] File not found.")
        sys.exit(1)
    # Check if file is in FITS format
    if not utils.is_file_fits_format(filepath):
        print("[ERR] Invalid file format. Please provide a FITS file.")
        sys.exit(1)

    hdul = utils.get_hdu_list(filepath)
    fits_data = utils.get_hdu_data(hdul)
    hdul.close()

    # Run algorithm
    algorithm = StarEX()
    result = algorithm.run(fits_data)

    # Save results
    utils.save_float_image(
        result["original"],
        os.path.join(results_dir, "original.png")
    )

    utils.save_float_image(
        result["reduced"],
        os.path.join(results_dir, "reduced.png")
    )

    utils.save_float_image(
        result["binary_mask"],
        os.path.join(results_dir, "binary_mask.png")
    )

    utils.save_float_image(
        result["gaussian_mask"],
        os.path.join(results_dir, "gaussian_mask.png")
    )

    utils.save_float_image(
        result["eroded"],
        os.path.join(results_dir, "eroded.png")
    )

    nb_stars = 0 if result["sources"] is None else len(result["sources"])
    print(f"[OK] Stars detected: {nb_stars}")