import os, sys
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

def collect_files(paths: List[str]) -> List[str]:
    '''Collects all FITS files from given paths (files or directories)

    Args:
        path (List[str]): A list of file or directory paths

    Returns:
        list: A list of FITS files, or only one!
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

def open_file_or_directory(path: str) -> bool:
    '''Opens a file or directory in the default file explorer

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
            f"[ERR] Failed to open {path}. Error: {e}", fg="red", bold=True, err=True
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
# Then, it gets associated with the function.
# =============================================
@click.group() # Command group
@click.version_option(version="1.2.0", prog_name="StarEX") # Version
def batch():
    '''StarEX - Lightweight Star Extraction & Removal Tool for FITS images'''
    pass

# Main command
@batch.command()
# Requires a single path to a FITS file, a list of paths to FITS files, or a directory containing FITS files
# nargs: -1 means it accepts any number of arguments (multiple paths)
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=True, file_okay=True))

# Output-type: what type of image to output (default: reduced)
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
                help="Type of image to output. Defaults to 'reduced'")

# Outdir (for Output Directory): directory to save results into
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results", 
                help="Directory to save results into")

# Format: output image format
@click.option("--format", "format_type", 
                type=click.Choice(["png8", "png16"]), # Removed "tiff" because it wont work :c
                default="png16", 
                help="Output format: png8 (8-bit PNG), png16 (16-bit PNG), tiff32 (32-bit TIFF). Defaults to png16")

# Normalize: normalize output image for TIFF files
@click.option("--no-norm", "no_normalize", 
                is_flag=True, 
                default=False, 
                help="Disable normalization for TIFF32 (raw float data). Only applies to tiff32 format! Defaults to False")

# Starless method: opening (fast) or inpainting (better quality)
@click.option("--starless-method", "starless_method", 
                type=click.Choice(["opening", "inpainting"]), 
                default="opening", 
                help="Method for starless image: opening (fast) or inpainting (better quality). Defaults to opening")

# FWHM (for Full Width at Half Maximum): the apparent size of a star
@click.option("-f", "--fwhm", "full_width_half_max", 
                type=float, 
                default=3.0, 
                help="Full width at half maximum: the apparent size of a star. Defaults to 3.0")

# Threshold sigma: to which threshold a star gets detected as a star
@click.option("-t", "--threshold_sigma", "threshold_sigma", 
                type=float, 
                default=2.5, 
                help="Threshold sigma value to use for star detection. The higher the value, the less stars will be detected. Defaults to 2.5")

# Reduction strength: the higher the value, the less stars will be visible
@click.option("-r", "--reduction_strength", "reduction_strength", 
                type=float, 
                default=0.65, 
                help="Star reduction strength. The value is between 0 and 1. Defaults to 0.65")

# Kernel radius: morphological kernel radius
@click.option("-k", "--kernel_radius", "kernel_radius", 
                type=int, 
                default=4, 
                help="Morphologicial kernel radius. Uses a circular kernel of a given radius. Defaults to 4")

# Iterations: number of morphological iterations
@click.option("-i", "--iterations", "iterations", 
                type=int, 
                default=1, 
                help="Number of morphological iterations: how many times morphological operations are applied. Defaults to 1")

# Date: add date to output filename to avoid overwriting
@click.option("--date/--no-date", 
                default=True, 
                help="Add date to output filename. Defaults to True")

# Open: open the resulting image (or directory if multiple files) or not
@click.option("--open/--no-open", 
              default=True, 
              help="Open the resulting image (or directory if multiple files) after processing. Defaults to True")

# Main function: processes FITS files
def process(paths, output_type, outdir, format_type, no_normalize, starless_method, full_width_half_max, threshold_sigma, reduction_strength, kernel_radius, iterations, date, open):
    '''Processes FITS images using StarEX algorithm'''

    # Collects all FITS files from given paths (files or directories)
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
    format_ext_map: Dict[str, str] = {
        "png8": ".png", 
        "png16": ".png", 
        "tiff": ".tiff"
    }
    file_ext = format_ext_map[format_type]

    # Initializes algorithm
    algorithm = StarEX(
        fwhm=full_width_half_max, 
        threshold_sigma=threshold_sigma, 
        reduction_strength=reduction_strength, 
        kernel_radius=kernel_radius, 
        iterations=iterations, 
        starless_method=starless_method
    )

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

        processed_files: List[str] = [] # What files have been processed

        # Progress bar
        desc: str = "Detecting stars" if current_type == "sources" else f"Processing {current_type}"
        with tqdm(files, desc, unit="file") as pbar:
            # Processes each FITS file
            for filepath in pbar:
                filename = Path(filepath).stem # Gets the name of the file without the extension
                pbar.set_postfix(file=filename[:20]) # Sets the progress bar postfix as the filename (maximum of 20 characters)

                # Loads FITS files
                hdul = utils.get_hdu_list(filepath)
                fits_data = utils.get_hdu_data(hdul)
                hdul.close()

                # Runs StarEX algorithm
                result = algorithm.run(fits_data, output_type=current_type)

                # Handles sources output, which are detected stars
                if current_type == "sources":
                    # Counts the number of sources
                    nb_sources = len(result['sources']) if result['sources'] is not None else 0
                    click.secho(
                        f"\n[OK] {filename}: {nb_sources} stars detected!", fg="yellow", bold=True
                        )
                    click.secho(
                        "------------------------------------------------------------------------------------------------------------------", fg="blue", bold=True
                    )
                else:
                    # Saves image output
                    output_data = result[current_type]

                    if date:
                        save_path = output_dir / f"{filename}_{current_type}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{file_ext}"
                    else:
                        save_path = output_dir / f"{filename}_{current_type}{file_ext}"

                    # Applies normalization if tiff32
                    normalize_tiff = not no_normalize if format_type == "tiff" else True

                    utils.save_float_image(
                        output_data, 
                        str(save_path), 
                        raw=False, 
                        format_type=format_type, 
                        normalize_tiff=normalize_tiff
                    )

                    click.secho(
                        f"\n[OK] Results saved to: {output_dir.absolute()}", fg="green", bold=True
                    )
                    click.secho(
                        "------------------------------------------------------------------------------------------------------------------", fg="blue", bold=True
                    )
                    processed_files.append(save_path) # Adds processed file to list

    # Open results if requested
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
# TODO: Add -b for blink to generate a gif with both images alterning
# TODO: Limited to two images in .png, could be FITS files?
@batch.command()
# Requires two file paths
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))

# Outdir (for Output Directory): directory to save results into
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results/comparison", 
                help="Directory to save comparison into")

# Date: add date to output filename to avoid overwriting
@click.option("--date", "date", 
                default=True, 
                help="Add date to output filename. Defaults to True")

# Open: open the resulting image (or directory if multiple files) or not
@click.option("--open/--no-open",
              default=True,
              help="Open the resulting image after comparison. Defaults to True")

def compare(file1, file2, outdir, date, open):
    '''Compares two FITS images side-by-side'''

    # Creates output directory
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Counts existing comparison (has .png extension)
    nb_files = len(list(output_dir.glob("*.png"))) + len(list(output_dir.glob("*.tiff")))
    if date:
        output_path = output_dir / f"comparison_{nb_files + 1}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    else:
        output_path = output_dir / f"comparison_{nb_files + 1}.png"

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