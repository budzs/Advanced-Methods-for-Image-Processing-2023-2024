# Advanced Methods for Image Processing - Lab 3

## Introduction
This is the code repository for the third lab on Image Denoising

Gitlab pages are automatically generated from this repository, and include:
- The lab instructions
- A documentation of the API (generated from the docstrings)

Refer to [this page](https://gitlab.com/am4ip/am4ip-lab3) to access to the lab instructions. To configure
your environment, please refer to the [Installation](#installation) section.


## Installation
An environment with a pre-configured Python + Torch installation using GPUs is available. Please follow
[this link.](https://dept-info.labri.fr/~mansenca/public/CREMI_deeplearning_env.pdf)

Then, you have to add the [src](src) folder to your python path, so it can find the given API. In Linux
it is done by running the following command, after replacing with the correct path (e.g., using pwd command
in src folder): 
```bash
export PYTHONPATH = $PYTHONPATH:/absolute/path/to/src
```
Note that this should be done everytime if you are running scripts in a terminal.

In a Jupyter Notebook (and even in a script), it can be done manually as follows:
```python
import sys
sys.path.append("/absolute/path/to/src")
```
before doing any import of the am4ip library.

One can also configure its IDE for the project.

### In Pycharm:
- right Click on the src folder
- Mark Directory as > Sources Root
