# 🌠 StarEX (beta)

![Python](https://img.shields.io/badge/Python-3.8+-green)

A simple but useful tool to achieve star reduction on .FITS files, using the Astropy.io and OpenCV libraries.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/zP0O23M7)

## Authors

- COLIN Noé [@Kiizer](https://www.github.com/Kiizer861)
- MELOCCO David [@ThFoxY](https://www.github.com/ThFoxY)
- LECLERCQ-SPETER Simon [@Koshy](https://www.github.com/KoshyMVP)

## Installation

**A stable version of Python is required (< 3.13)**. A higher version may cause undesirable results or errors.

* See requirements.txt for full dependency list

### Libraries

* Astropy.io **7.2.0** - [User Guide](https://docs.astropy.org/en/stable/index_user_docs.html)
* Matplotlib **3.10.8** - [API Reference](https://matplotlib.org/stable/api/index.html)
* NumPy **2.20** - [Docs](https://numpy.org/doc/)
* OpenCV for Python **4.12.0.88** - [Modules](https://docs.opencv.org/4.x/index.html)
* Photutils **2.3.0** - [User Guide](https://photutils.readthedocs.io/en/stable/user_guide/index.html)
* Tabulate **0.9.0** - [Full Guide](https://www.datacamp.com/tutorial/python-tabulate)

### Virtual Environment

It is recommended to create a virtual environment before installing dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Troubleshooting for Powershell:
```bash
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Dependencies
```bash
pip install -r requirements.txt
```

Or install dependencies manually (specified above):
```bash
pip install [package-name]
```

## Examples files
Example files are located in the `examples/` directory. You can run the scripts with these files to see how they work.
- Example 1: `examples/HorseHead.fits` (Black and white FITS image file for testing)
- Example 2: `examples/test_M31_linear.fits` (Color FITS image file for testing)
- Example 3: `examples/test_M31_raw.fits` (Color FITS image file for testing)