import os, sys, time
from datetime import datetime
import platform, subprocess # For compatibility: gets operating system (https://docs.python.org/fr/3.9/library/platform.html) and runs a command to open a file or directory
from pathlib import Path # For compatibility: it avoids struggle between Windows and UNIX paths (https://docs.python.org/3/library/pathlib.html)

# Third-party imports
from tqdm import tqdm # Progress bar
import click # Command line interface

# Typing
from typing import List, Dict

# Local imports
from algorithm import StarEX # Algorithm
import utils

# TODO: Add Matplotlib support to generate useful plots

# TODO: Can search recursively inside subdirectories
def collect_files(paths: List[str]) -> List[str]:
    '''Collects all FITS files from a list of file or directory paths

    - Directories are scanned non-recursively
    - Only .fits files are considered: .fits, .fts, .fit

    Args:
        path (List[str]): A list of file or directory paths

    Returns:
        List[str]: A list of FITS file paths, or only one!
    '''
    files = [] # List of FITS files
    # For every path in the specified parameter list
    for path in paths:
        path_obj = Path(path) # Convert to Path object for compatibility

        # If it's a directory
        if path_obj.is_dir():
            # Iterates through all FITS files in directory using a list comprehension
            fits_files = [
                # Checks for each file if it's a FITS file
                # TODO: Can search recursively inside subdirectories
                str(f) for f in path_obj.iterdir() if utils.is_file_fits_format(str(f))
            ]
            files.extend(fits_files) # Adds FITS files to list

        # If it's a file
        elif path_obj.is_file():
            # If it is a FITS file
            if utils.is_file_fits_format(str(path)):
                files.append(str(path)) # Adds FITS file to list
            else:
                click.secho(
                    f"[WARNING] {path} is not a FITS file. Skipping...", fg="yellow"
                )
        else:
            click.secho(
                f"[ERR] {path} was not found. Maybe check for typos?", fg="red", bold=True, err=True
            )
    return sorted(files)

# TODO: Try it on macOS and Linux
def open_file_or_directory(path: str) -> bool:
    '''Opens a file or directory in the default file explorer

    - Includes: Windows, macOS and Linux

    Args:
        path (str): The path to the file or directory

    Returns:
        bool: True if the file or directory was opened
    '''
    try:
        system = platform.system() # Gets operating system

        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin": # macOS
            subprocess.run(["open", path], check=True)
        elif system == "Linux":
            subprocess.run(["xdg-open", path], check=True)
        return True
    except Exception as e:
        click.secho(
            f"[ERR] Failed to open {path}: {e}", fg="red", bold=True, err=True
        )
        return False

# =============================================
# CLI COMMANDS
# Click uses decorators:
#
# @batch.command() adds a command
# @click.argument() adds an argument (required)
# @click.option() adds an option (optional)
#
# Then, it gets associated with a function.
# =============================================
@click.group() # Command group
@click.version_option(version="b1.2.1", prog_name="StarEX") # Version
def batch():
    '''StarEX - Lightweight Star Extraction & Removal Tool for FITS images.

    This CLI allows:
    - Star detection (using DAOStarFinder),
    - Mask generation (using detected stars and Gaussian blur),
    - Star reduction,
    - Image comparison (or blinking within a GIF),
    - FITS header viewing (experimental)
    '''
    pass

# Main command
@batch.command()
# Requires a single path to a FITS file, a list of paths to FITS files, or a directory containing FITS files
# nargs: -1 means it accepts any number of arguments (multiple paths)
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=True, file_okay=True))

# Output-type: what type of image to output
@click.option("-otype", "--output_type", "output_type", 
                type=click.Choice([
                    "original", 
                    "binary_mask", 
                    "gaussian_mask", 
                    "eroded", 
                    "reduced", 
                    "sources", 
                    "all"
                ]), default="reduced", 
                help="Type of image output to generate.")

# Outdir (for Output Directory): directory to save results into
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results", 
                help="Output directory.")

# Format: output image format
@click.option("--format", "format_type", 
                type=click.Choice(["png8", "png16"]), # TIFF won't work :c
                default="png16", 
                help="Output format: png8 (8-bit PNG) or png16 (16-bit PNG). TIFF (32-bit) is not yet supported!")

# TODO: TIFF format is not supported..
# Normalize: normalize output image for TIFF files
# @click.option("--no-norm", "no_normalize", 
#                 is_flag=True, 
#                 default=False, 
#                 help="Disable normalization for TIFF32 (raw float data). Only applies to tiff32 format! Defaults to False")

# Starless method: which method to use for starless image
@click.option("--starless-method", "starless_method", 
                type=click.Choice(["opening", "inpainting"]), 
                default="opening", 
                help="Method for starless image: opening (fast) or inpainting (better quality).")

# FWHM (for Full Width at Half Maximum): the apparent size of a star
@click.option("-f", "--fwhm", "full_width_half_max", 
                type=float, 
                default=3.0, 
                help="Full width at half maximum: the apparent size of a star.")

# Threshold sigma: to which threshold a star gets detected as a star
@click.option("-t", "--threshold_sigma", "threshold_sigma", 
                type=float, 
                default=2.5, 
                help="Threshold sigma value to use for star detection. The higher the value, the less stars will be detected.")

# Reduction strength: the higher the value, the less stars will be visible
@click.option("-r", "--reduction_strength", "reduction_strength", 
                type=float, 
                default=0.65, 
                help="Star reduction strength: attenuates visible stars between 0 and 1.")

# Kernel radius: morphological kernel radius
@click.option("-k", "--kernel_radius", "kernel_radius", 
                type=int, 
                default=4, 
                help="Morphologicial kernel radius: uses a circular kernel of a given radius.")

# Iterations: number of morphological iterations
@click.option("-i", "--iterations", "iterations", 
                type=int, 
                default=1, 
                help="Number of morphological iterations: how many times morphological operations are applied.")

# Multiscale: adaptive kernel sizes
@click.option("--multiscale/--no-multiscale",
                default=False,
                help="Use adaptive kernel sizes based on star magnitude.")

# Tiling: enable tiled processing
@click.option("--tiling/--no-tiling",
                default=False,
                help="Enable tiled processing for large images.")

# Tile size: size of each tile
@click.option("--tile-size", "tile_size",
                type=int,
                default=128,
                help="Size of tiles for tiled processing.")

# Tile overlap: overlap between tiles
@click.option("--tile-overlap", "tile_overlap",
                type=int,
                default=64,
                help="Overlap between tiles in pixels.")

# Workers: number of worker processes
@click.option("--workers", "workers",
                type=int,
                default=None,
                help="Number of worker processes. Defaults to CPU count")

# Date: add date to output filename to avoid overwriting
@click.option("--date/--no-date", 
                default=False, 
                help="Add date to output filename. Avoids overwriting.")

# Open: open the resulting image (or directory if multiple files)
@click.option("--open/--no-open", 
              default=False, 
              help="Open the resulting image (or directory if multiple files) after processing.")

# Main function: processes FITS files
def process(
    paths, 
    output_type, 
    outdir, 
    format_type, 
    starless_method, 
    full_width_half_max, 
    threshold_sigma, 
    reduction_strength, 
    kernel_radius, 
    iterations, 
    multiscale, 
    tiling, 
    tile_size, 
    tile_overlap, 
    workers, 
    date, 
    open
):
    '''Processes FITS images using StarEX algorithm

    Loads FITS files and processes them using the following steps:
    1. Load FITS image(s)
    2. Detect stars using DAOStarFinder
    3. Generate mask using sources (detected stars) and Gaussian blur
    4. Apply star reduction
    5. Export results

    For larger images: supports tiled processing with multiprocessing.
    '''

    # Collects all FITS files
    files = collect_files(paths)

    if not files:
        click.secho(
            "[ERR] No FITS files found.", fg="red", bold=True, err=True
        )
        sys.exit(1)

    click.secho(
        f"[OK] Found {len(files)} FITS files.", fg="green", bold=True
    )

    # Maps format type to its file extension
    # format_ext_map: Dict[str, str] = {
    #     "png8": ".png", 
    #     "png16": ".png", 
    #     "tiff": ".tiff"
    # }
    # file_ext = format_ext_map[format_type]

    # Initializes algorithm
    algorithm = StarEX(
        fwhm=full_width_half_max, 
        threshold_sigma=threshold_sigma, 
        reduction_strength=reduction_strength, 
        kernel_radius=kernel_radius, 
        iterations=iterations, 
        starless_method=starless_method, 
        multiscale=multiscale
    )

    # Starts global timer: keeps track of total processing time
    global_start_time = time.time()

    # Defines the types to be processed
    if output_type == "all":
        need_to_be_processed: List[str] = [
            "original", 
            "binary_mask", 
            "gaussian_mask", 
            "eroded", 
            "reduced", 
            "sources"
        ]
    else:
        need_to_be_processed: List[str] = [output_type]

    # Handles each type of output
    for current_type in need_to_be_processed:
        # Creates the output directory for this type (except for sources)
        # TODO: Sources generates Matplotlib charts
        if current_type != "sources":
            output_dir = Path(outdir) / current_type
            output_dir.mkdir(parents=True, exist_ok=True)

        processed_files: List[str] = [] # Keeps track of processed files

        # Sets progress bar description
        desc: str = "Detecting stars" if current_type == "sources" else f"Processing {current_type}"

        # tqdm: adds a nice progress bar!
        with tqdm(files, desc, unit="file") as pbar:
            # Processes each FITS file
            for filepath in pbar:
                filename = Path(filepath).stem # Gets the name of the file without the extension
                pbar.set_postfix(file=filename[:20]) # Sets the progress bar postfix as the filename (maximum of 20 characters)

                # Loads FITS files with explicit context manager
                with utils.get_hdu_list(filepath) as hdul:
                    fits_data = utils.get_hdu_data(hdul)

                # Starts timer
                start_time = time.time()

                # Runs StarEX algorithm with or without tiling
                try:
                    # Checks if tiling is enabled and if the current_type can be tiled properly (not masks!)
                    if tiling and current_type not in ["sources", "binary_mask", "gaussian_mask"]:
                        # Prepares algorithm parameters
                        parameters = {
                            'fwhm': full_width_half_max, 
                            'threshold_sigma': threshold_sigma, 
                            'reduction_strength': reduction_strength, 
                            'kernel_radius': kernel_radius, 
                            'iterations': iterations, 
                            'starless_method': starless_method, 
                            'multiscale': multiscale
                        }

                        click.secho(
                            f"[OK] Using tiling with parameters: {parameters}", fg="cyan", bold=True
                        )

                        # Processes image using tiling
                        output_data = utils.process_image_tiled(
                            fits_data, 
                            parameters, 
                            current_type, 
                            tile_size, 
                            tile_overlap, 
                            workers
                        )

                        result = {current_type: output_data}
                    else:
                        # Processing without tiling
                        result = algorithm.run(fits_data, output_type=current_type)
                except Exception as e:
                    click.secho(
                        f"\n[ERR] Error processing {filename}: {e}", fg="red", bold=True, err=True
                    )
                    continue # Skips to the next file

                # Ends timer
                end_time = time.time()
                elapsed_time = end_time - start_time

                # Handles sources output, which are detected stars
                if current_type == "sources":
                    # Counts the number of sources
                    nb_sources = len(result['sources']) if result['sources'] is not None else 0

                    click.secho(
                        f"\n[OK] {filename}: {nb_sources} stars detected! (processed in {elapsed_time:.2f}s)", fg="yellow", bold=True
                        )
                    click.secho(
                        f"{'=' *100}", fg="cyan", bold=True
                    )
                else:
                    # Saves image output
                    output_data = result[current_type]

                    if date:
                        save_path = output_dir / f"{filename}_{current_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
                    else:
                        save_path = output_dir / f"{filename}_{current_type}.png"

                    # Applies normalization if tiff32
                    # normalize_tiff = not no_normalize if format_type == "tiff" else True

                    # Saves image
                    utils.save_float_image(
                        output_data, 
                        str(save_path), 
                        raw=False, # Reconstructs FITS file but it is not very useful.. because the FITS file are already available :(
                        format_type=format_type, 
                    )

                    click.secho(
                        # Absolute path because it's easier to find!
                        f"\n[OK] Results saved to: {output_dir.absolute()} (processed in {elapsed_time:.2f}s)", fg="green", bold=True
                    )
                    click.secho(
                        f"{'-' *100}", fg="cyan", bold=True
                    )
                processed_files.append(save_path) # Adds processed file to list

    # Calculates total processing time
    total_elapsed_time = time.time() - global_start_time
    
    click.secho(
        f"\n{'=' *100}", fg="cyan", bold=True
    )
    click.secho(
        f"[DONE] Total processing time: {total_elapsed_time:.2f}s ({len(files)} file(s) processed)", 
        fg="cyan", bold=True
    )
    click.secho(
        f"{'=' *100}\n", fg="cyan", bold=True
    )

    # Open results if requested
    # TODO: Optimize this nested if statement
    if open and output_type != "sources":
        if len(processed_files) == 1:
            # Only one file was processed: opens it
            if open_file_or_directory(str(processed_files[0])):
                click.secho(
                    f"[OK] Results opened: {processed_files[0].name}", fg="green", bold=True
                )
            else:
                click.secho(
                    f"[ERR] Failed to open {processed_files[0].name}", fg="red", bold=True
                )
        elif len(processed_files) > 1:
            # A directory with multiple files was processed: opens it
            if open_file_or_directory(str(output_dir)):
                click.secho(
                    f"[OK] Results opened: {output_dir}", fg="green", bold=True
                )
            else:
                click.secho(
                    f"[ERR] Failed to open {output_dir}", fg="red", bold=True
                )
        else:
            click.secho(
                "[ERR] No results to open.", fg="red", bold=True
            )

# Comparison: compares two processed FITS images
# TODO: Limited to two images in .png, could be raw FITS files?
@batch.command()
# Requires two file paths
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))

# Outdir (for Output Directory): directory to save results into
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results/comparison", 
                help="Output directory for comparison results.")

# Blink: animated GIF alternating between images
@click.option("-b", "--blink", "blink",
                is_flag=True,
                default=False,
                help="Generate animated GIF alternating between images.")

# Blink duration: duration per frame in milliseconds
@click.option("--blink-duration", "blink_duration",
                type=int,
                default=500,
                help="Duration per frame in milliseconds for blink mode.")

# Date: add date to output filename to avoid overwriting
@click.option("--date/--no-date", 
                default=False, 
                help="Add date to output filename. Avoids overwriting.")

# Open: open the resulting image or animated GIF after comparison
@click.option("--open/--no-open", 
              default=False, 
              help="Open the resulting image or animted GIF after comparison.")

def compare(file1, file2, outdir, blink, blink_duration, date, open):
    '''Compares two FITS images side-by-side'''

    # Creates output directory
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Counts existing comparison
    index = len(list(output_dir.glob("*")))

    if blink:
        # Generates animated GIF
        if date:
            output_path = output_dir / f"comparison_{index + 1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.gif"
        else:
            output_path = output_dir / f"comparison_{index + 1}.gif"

        # Generates animated GIF comparison
        utils.save_gif(file1, file2, str(output_path), duration=blink_duration)
        click.secho(
            f"[OK] Animated GIF comparison saved to: {output_path.absolute()}", fg="green", bold=True
        )
    else:
        # Generates static image side-by-side
        if date:
            output_path = output_dir / f"comparison_{index + 1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        else:
            output_path = output_dir / f"comparison_{index + 1}.png"

        # Creates comparison (images are side-by-side)
        utils.save_combined_images(file1, file2, str(output_path))

        click.secho(
            f"[OK] Comparison saved to: {output_path.absolute()}", fg="green", bold=True
        )

    # Opens result if requested
    if open:
        if open_file_or_directory(str(output_path)):
            click.secho(f"[OK] Result opened: {output_path.name}", fg="green")

# TODO: Display multiple FITS files header (using HDUList?)
# View command: displays FITS header information in a formatted table by using Tabulate library
@batch.command()
# Requires a file path
@click.argument("file", required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True))

def view(file):
    '''Views FITS header information in a formatted table'''
    click.echo(utils.create_fits_header_table(utils.get_hdu_list(file), 0))

if __name__ == '__main__':
    batch()