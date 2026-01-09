import os, sys
import platform, subprocess # For compatibility: gets operating system (https://docs.python.org/fr/3.9/library/platform.html) and runs a command to open a file or directory
from pathlib import Path # For compatibility: it avoids struggle between Windows and UNIX paths (https://docs.python.org/3/library/pathlib.html)
from tqdm import tqdm
import click
from typing import List
from algorithm import StarEX
import utils
# TODO: Sorts out imports

# TODO: Add explicit types because I'm lazy to do it
def collect_files(paths: List[str]) -> List[str]:
    '''Collects all FITS files from given paths (files or directories)

    Args:
        path (List[str]): A list of file or directory paths

    Returns:
        list: A list of FITS files, or only one!
    '''
    files = []
    for path in paths:
        path_obj = Path(path) # Convert to Path object for compatibility

        # If it's a directory
        if path_obj.is_dir():
            # Collects all FITS files in directory
            fits_files = [
                str(f) for f in path_obj.iterdir() if utils.is_file_fits_format(str(f))
            ]
            files.extend(fits_files)

        # If it's a file
        elif path_obj.is_file():
            if utils.is_file_fits_format(str(path)):
                files.append(str(path))
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
        system = platform.system()

        if system == "Windows":
            os.startfile(path)
        elif system == "Darwin": # macOS
            subprocess.run(["open", path], check=True)
        elif system == "Linux": # Linux
            subprocess.run(["xdg-open", path], check=True)

        return True
    except Exception as e:
        click.secho(
            f"[ERR] Failed to open {path}. Error: {e}", fg="red", bold=True, err=True
        )
        return False

# =========================
# CLI COMMANDS
# =========================
@click.group()
@click.version_option(version="beta", prog_name="StarEX")
def batch():
    '''StarEX - Lightweight Star Extraction & Removal Tool for FITS images'''
    pass

@batch.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=True, file_okay=True))
@click.option("--output_type", "output_type", 
                type=click.Choice([
                    "original", 
                    "binary_mask", 
                    "gaussian_mask", 
                    "eroded", 
                    "reduced", 
                    "sources"
                ]), default="reduced", 
                help="Type of image to output. Defaults to 'reduced'")
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results", 
                help="Directory to save results")
@click.option("-f", "--fwhm", "full_width_half_max", 
                type=float, 
                default=3.0, 
                help="Full width at half maximum")
@click.option("-t", "--threshold_sigma", "threshold_sigma", 
                type=float, 
                default=2.5, 
                help="Threshold sigma value to use for star detection")
@click.option("-r", "--reduction_strength", "reduction_strength", 
                type=float, 
                default=0.5, 
                help="Star reduction strength (0-1)")
@click.option("-k", "--kernel_radius", "kernel_radius", 
                type=int, 
                default=4, 
                help="Morphologicial kernel radius")
@click.option("-i", "--iterations", "iterations", 
                type=int, 
                default=1, 
                help="Number of morphological iterations")
@click.option("--open/--no-open", 
              default=True, 
              help="Open results after processing. Defaults to True")
def process(paths, output_type, outdir, full_width_half_max, threshold_sigma, reduction_strength, kernel_radius, iterations, open):
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

    # Initializes algorithm
    algorithm = StarEX(
        fwhm=full_width_half_max,
        threshold_sigma=threshold_sigma,
        reduction_strength=reduction_strength,
        kernel_radius=kernel_radius,
        iterations=iterations
    )

    # Creates output directory
    # TODO: Add datestamp?
    output_dir = Path(outdir) / output_type
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_files = []

    with tqdm(files, desc="Processing", unit="file") as pbar:
        for filepath in pbar:
            filename = Path(filepath).stem
            pbar.set_postfix(file=filename)

            # Loads FITS files
            hdul = utils.get_hdu_list(filepath)
            fits_data = utils.get_hdu_data(hdul)
            hdul.close()

            # Runs StarEX algorithm
            result = algorithm.run(fits_data, output_type=output_type)

            # Handles sources output
            if output_type == "sources":
                nb_sources = len(result['sources']) if result['sources'] is not None else 0
                click.echo(f"{filename} : {nb_sources} stars detected!")
            else:
                # Saves image output
                output_data = result[output_type]
                save_path = output_dir / f"{filename}_{output_type}.png"
                utils.save_float_image(output_data, str(save_path))
                processed_files.append(save_path)

    click.secho(
        f"Results saved to: {output_dir.absolute()}", fg="green", bold=True
    )

    # Open results if requested
    if open and output_type != "sources":
        if len(processed_files) == 1:
            # Only one file was processed: opens it
            if open_file_or_directory(str(processed_files[0])):
                click.secho(
                    f"Results opened: {processed_files[0].name}", fg="green", bold=True
                )
        elif len(processed_files) > 1:
            # A directory with multiple files was processed: opens it
            if open_file_or_directory(str(output_dir)):
                click.secho(
                    f"Results opened: {output_dir}", fg="green", bold=True
                )

@batch.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option("-o", "--outdir", "outdir", 
                type=click.Path(), 
                default="results/comparison", 
                help="Output directory for comparison")
@click.option("--open/--no-open",
              default=True,
              help="Open result after creating comparison. Defaults to True")
def compare(file1, file2, outdir, open):
    '''Compare two FITS images side-by-side'''

    # Creates output directory
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Counts existing comparison
    nb_files = len(list(output_dir.glob("*.png")))
    output_path = output_dir / f"comparison_{nb_files + 1}.png"

    # Creates comparison
    utils.save_combined_images(file1, file2, str(output_path))

    click.secho(
        f"Comparison saved to: {output_path.absolute()}", fg="green", bold=True
    )

    # Opens result if requested
    if open:
        if open_file_or_directory(str(output_path)):
            click.secho(f"✅ Opened: {output_path.name}", fg="green")

# TODO: Needs some additionnal work to be useful
@batch.command()
@click.argument("filename", required=True, type=click.Path(exists=True, dir_okay=False, file_okay=True))
def view(filename):
    '''View FITS header information in a formatted table'''
    click.echo(utils.create_fits_header_table(utils.get_hdu_list(filename), 0))

if __name__ == '__main__':
    batch()