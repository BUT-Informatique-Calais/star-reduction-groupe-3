import argparse
import os
from algorithm import StarEX
import utils
from typing import List
from numpy.typing import NDArray

# TODO: Be more explicit with help messages
def collect_files(path: str) -> List[str]:
    '''Collects all FITS files in a directory or a single file

    Args:
        path (str): The path to the file or directory

    Returns:
        list: A list of FITS files, or only one!
    '''
    # Checks if the path is a file or directory
    if os.path.isdir(path):
        if not utils.does_dir_exist(path):
            raise Exception("[ERR] Directory not found. Maybe check your spelling?")
        # If it's a directory, collect all FITS files
        files = [os.path.join(path, f) for f in os.listdir(path) if utils.is_file_fits_format(f)]
    else:
        if not utils.does_file_exist(path):
            raise Exception("[ERR] File not found. Maybe check your spelling?")
        files = [path]

    return files

if __name__ == "__main__":
    '''Batch processing for FITS images using StarEX algorithm'''

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        prog="StarEX Batch Processing Tool", 
        description="Thank you for using StarEX Batch Processing Tool.\nPlease provide a FITS image or directory of FITS images. To know more, use -h or --help!",
        epilog="Syntax: python main.py /path/to/fits --type binary_mask" # This is displayed at the end of the help message
    )
    # =========================
    # MANDATORY ARGS
    # =========================
    parser.add_argument("paths", nargs='+', metavar="PATH", help="File(s) or directory(ies) of FITS images")

    # =========================
    # OPTIONAL ARGS
    # =========================
    # TODO: Generates every output types
    parser.add_argument(
        "-t", "--type",
        choices=["original", "binary_mask", "gaussian_mask", "eroded", "reduced", "sources"],
        default="reduced",
        help="Type of image to output. Defaults to 'reduced'"
    )
    parser.add_argument("-f", "--fwhm", type=float, default=3.0, help="FWHM")
    parser.add_argument("-d", "--threshold_sigma", type=float, default=2.5, help="Detection threshold")
    parser.add_argument("-r", "--reduction_strength", type=float, default=0.7, help="Reduction strength")
    parser.add_argument("-k", "--kernel_radius", type=int, default=4, help="Kernel radius")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("-c", "--comparison", type=str, nargs=2, metavar=("FILE1", "FILE2"), help="Compare two images")
    # TODO: Put a list of files
    parser.add_argument("-o", "--outdir", type=str, default="results", help="Directory to save results")

    # Parse arguments
    args = parser.parse_args()
    
    # Get files from all provided paths
    files = []
    for path in args.paths:
        files.extend(collect_files(path))
    
    # Create output directory
    if args.comparison is not None:
        comparison_dir = os.path.join(args.outdir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        # Gets the number of files to add to the filename
        # TODO: Comparison with every file
        nb_files = len([f for f in os.listdir(comparison_dir) if f.startswith("comparison_") and f.endswith(".png")])
        combined = utils.save_combined_images(args.comparison[0], args.comparison[1], os.path.join(comparison_dir, f"comparison_{nb_files+1}.png"))
        print(f"[INFO] Comparison image saved at {os.path.join(comparison_dir, f'comparison_{nb_files+1}.png')}")

    # Initialize algorithm
    algorithm = StarEX(
        fwhm=args.fwhm, 
        threshold_sigma=args.threshold_sigma, 
        reduction_strength=args.reduction_strength, 
        kernel_radius=args.kernel_radius
    )

    # Process files
    for filepath in files:
        hdul = utils.get_hdu_list(filepath)
        fits_data = utils.get_hdu_data(hdul)
        hdul.close()
        result = algorithm.run(fits_data, output_type=args.type)

        # TODO: Create a table with sources
        if args.type == "sources":
            print(f"[OK] {filepath}: {len(result['sources'])} stars detected")
        else:
            # Retrieves filename from path
            basename = os.path.splitext(os.path.basename(filepath))[0]
            output_data = result[args.type]

            # Creates directories for each type of image (apart from sources)
            os.makedirs(os.path.join(args.outdir, args.type), exist_ok=True)

            # Saves image in the directory type
            save_path = os.path.join(args.outdir, args.type, f"{basename}_{args.type}.png")
            utils.save_float_image(output_data, save_path) 

            # Prints results
            print(f"[OK] {filepath}: Saved to {save_path}")