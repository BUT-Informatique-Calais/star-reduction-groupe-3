# StarEX - Terminal Commands

To get started, follow the examples below and discover the effectiveness of the **StarEX algorithm**!

---

## Available commands

### 1. `process` - Processing FITS images

Processes FITS images using the StarEX algorithm. It is its main feature and has a variety of options!

#### Syntax
```bash
python batch.py process [OPTIONS] PATHS...
```

#### Arguments (mandatory)
- `PATHS` : A file, multiple files or a directory that contains FITS files

#### Options (optional)

| Option | Shortcut | Type | Defaults to | Description |
|--------|-----------|------|--------|-------------|
| `--output_type` | - | Choice | `reduced` | Image type to process |
| `--outdir` | `-o` | Path | `results` | Output directory |
| `--fwhm` | `-f` | Float | `3.0` | Full Width at Half Maximum |
| `--threshold_sigma` | `-t` | Float | `2.5` | Sigma threshold for star detection |
| `--reduction_strength` | `-r` | Float | `0.5` | Star reduction strength (0-1) |
| `--kernel_radius` | `-k` | Int | `4` | Morphological kernel radius |
| `--iterations` | `-i` | Int | `1` | Number of morphological iterations |
| `--open/--no-open` | - | Flag | `--open` | Open results after processing |

#### Possible values for `--output_type`
- `original` : original image
- `binary_mask` : binary mask of detected stars
- `gaussian_mask` : mask with Gaussian blur applied
- `eroded` : eroded image (called starless)
- `reduced` : reduced image (stars are less bright)
- `sources` : list of detected stars

#### Examples
```bash
# Process an unique FITS file with default options
python batch.py process image.fits

# Process every FITS files within a specified directory
python batch.py process /path/to/dir

# Generates a binary mask
python batch.py process image.fits --output_type binary_mask

# Customized processing with basic options 
python batch.py process image.fits -f 4.0 -t 3.0 -r 0.8 -o results

# Process multiples files and directories
python batch.py process image1.fits image2.fits /dir/

# Disable automatic opening of the resulting file or directory
python batch.py process image.fits --no-open

# Customized processing with advanced options
python batch.py process image.fits -k 6 -i 2 --reduction_strength 0.9
```

---

### 2. `compare` - Compare two FITS images

Compare two resulting FITS image side-by-side in a single .png image.

#### Syntax
```bash
python batch.py compare [OPTIONS] FILE1 FILE2
```

#### Arguments (mandatory)
- `FILE1` : First image to compare
- `FILE2` : Second image to compare

#### Options (optional)

| Option | Shortcut | Type | Defaults to | Description |
|--------|-----------|------|--------|-------------|
| `--outdir` | `-o` | Path | `results/comparison` | Output directory for comparisons |
| `--open/--no-open` | - | Flag | `--open` | Open comparison directory after processing |

#### Examples
```bash
# Compare two processed FITS images
python batch.py compare original_image.png reduced_image.png

# Compare two processed FITS images and save result in a specified directory
python batch.py compare img1.png img2.png -o comparaisons/

# Compare to processed FITS images without opening result .png
python batch.py compare img1.png img2.png --no-open
```

---

### 3. `view` - View FITS header information

View in a formatted table a FITS file header.

#### Syntax
```bash
python batch.py view FILENAME
```

#### Arguments
- `FILENAME` : FITS file to view header from

#### Examples
```bash
# View FITS file header
python batch.py view image.fits
```

---

## General info

### Get the StarEX tool version
```bash
python batch.py --version
```

### Get help
```bash
# Help
python batch.py --help

# Help for a specified command
python batch.py process --help
python batch.py compare --help
python batch.py view --help
```

---

## Notes

- FITS file available formats: `.fits`, `.fit`, `.fts`
- Results are saved into .png images
- Cool progress bar when processing!
