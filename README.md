# ESCHR

[![Documentation][badge-docs]][link-docs]
[![codecov](https://codecov.io/gh/zunderlab/eschr/branch/main/graph/badge.svg)](https://codecov.io/gh/zunderlab/eschr)

[badge-tests]: https://img.shields.io/github/actions/workflow/status/zunderlab/eschr/test.yaml?branch=main
[link-tests]: https://github.com/zunderlab/eschr/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/eschr

ESCHR: A hyperparameter-randomized ensemble approach for robust clustering across diverse datasets

Please refer to the full [documentation][link-docs] for further details.

## Overview of Algorithm:

![figure](https://github.com/zunderlab/eschr/raw/main/figure.png)

## Quick Start

This [example jupyter notebook on Google Colab](https://github.com/zunderlab/eschr/raw/main/docs/notebooks/paul15_mouse_hematopoiesis.ipynb) provides a walkthrough of ESCHR analysis using an example scRNA-seq dataset. If you launch the notebook in Google Colab, you will not need to install ESCHR locally.

## Installation

### Quick install
`pip install eschr`

### Detailed installation instructions
You need to have a Python version between 3.8 and 3.10 (inclusive) installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).
(_If you are installing on a Windows OS, you will need to use python version 3.8 only due to issues with installing the nmslib dependency in other versions_)

1. Make sure you have Anaconda installed and functional. [Conda FAQ](https://docs.anaconda.com/anaconda/user-guide/faq/) is a great resource for troubleshooting and verifying that everything is working properly.
2. Open terminal or equivalent command line interface and run `conda create --name <env_name> python=<version>` (e.g. `conda create --name eschrEnv python=3.8`)
3. Activate the environment by running `conda activate <env_name>`
4. Once environment is activated, run `conda install pip`
5. If you do not have Git installed, run `conda install git`
6. To install the most up to date version of ESCHR into your conda environment, run the following line:
   `pip install git+https://github.com/zunderlab/eschr.git`
   or to install teh latest release, run:
   `pip install eschr`
8. Verify that the Conda environment was created successfully by running `conda list` and verifying that expected packages are installed for this environment. Then either close the environment by running `conda deactivate` or proceed to subsequent optional setup and/or running the method within the environment.


## Basic example run script:

If you want to try ESCHR on your own data, you can start with this basic example script. This assumes that you have a preprocessed .csv file with features as columns and cells or other data points as rows.

```
import eschr as es
import pandas as pd
import anndata

# Read in data from a csv file.
# The method expects features as columns
# Use commented out ".T" if you have features as rows
# Remove "index_col = 0" if your csv does not have row indices included.
# Also ensure that data has already been preprocessed/scaled/normalized
# as appropriate for your data type.
data_filepath = "/path/to/your/data.csv"
data = pd.read_csv(data_filepath, index_col=0)#.T

# Make AnnData object with your data
adata = anndata.AnnData(X=data)

# Optionally specify the path for creating the zarr store that
# will be used for interacting with your data. Otherwise it will
# be created in the working directory.
zarr_loc = "/path/to/your/data.zarr"

# Now you can run the method with your prepped data!
# (add any optional hyperparameter specifications,
# but bear in mind the method was designed to work for
# diverse datasets with the default settings.)
adata = es.tl.consensus_cluster(
            adata=adata,
            zarr_loc=zarr_loc
        )

# Plot soft membership matrix heatmap visualization
es.pl.smm_heatmap(adata, output_path="/where/to/save/figure.png")

# Plot umap visualization
es.pl.umap_heatmap(adata, output_path="/where/to/save/figure.png")
```

## Setting up to run via command line:

```
conda activate <env_name>
python3
import eschr
```

Now you can run code adapted from the example run scripts above or copy and paste lines of code from the tutorial jupyter notebook.

## Setting up to run a Jupyter Notebook on your PC:

(note it is likely only small datasets will be able to run on a PC, but feel free to try large ones!)

1. Open terminal or equivalent command line interface, activate the environment you create above by running `conda activate <env_name>`
2. While in the activated environment, run `conda install -c anaconda ipykernel`
3. Next run `python -m ipykernel install --user --name=<env_name>`.
4. You can then then close the environment by running `conda deactivate`
5. Open Anaconda Navigator and click on the icon for Jupyter Notebooks (this should open an instance of a command line interface, and then open a tab in your default browser to a page containing your PC's file structure)
6. Navigate to where you saved the downloaded [tutorial notebook](https://github.com/zunderlab/eschr/raw/main/docs/notebooks/paul15_mouse_hematopoiesis.ipynb) and click to open that notebook, or start a fresh notebook if you prefer to work off of the `Quick Start` instructions with your own data.
7. Upon opening the notebook, you may be prompted to select a kernel, or if not you can click on the `Kernel` menu from the top navigation bar, and then `Change kernel`. The name of the environment you created should show up as an option for the kernel - select that as the kernel for your notebook.
8. You should now be ready to run! Just click your way through the notebook. You can change output paths for the visualizations when you get to those cells.

## Setting up to run a Jupyter Notebook on HPC OpenOnDemand:

1. Navigate to [UVA OpenOnDemand](https://rivanna-portal.hpc.virginia.edu/pun/sys/dashboard/) if you are a member of the University fo Virginia, otherwise navigate to the equivalent for your institution.
2. Enter the dropdown menu for "Interactive Apps" from the top menu bar and select "JupyterLab"
3. Submit a job - for optimal performance, select 4+ cores and 100GB memory (many cases won't end up needing that much memory, but it is very unlikely anything would ever exceed that - highest I've seen for peak mem is pushing 50 GB)
4. Once your job starts, click `Connect to Jupyter`
5. First, start a terminal window. You can either port in your conda env from your computer to the location of conda envs in your Rivanna storage (home/<your compute id>/.conda/envs - you will have to click `show dot files` to find this). Alternatively, you can upload the requirements.txt file and follow the installation instructions above to create the environment on Rivanna.
6. Either way, you want to end up in this terminal window with your environment activated (`conda activate <env_name>`). While in the activated environment, run `conda install -c anaconda ipykernel`
7. Next run `python -m ipykernel install --user --name=<env_name>`.
8. You can then then close the environment by running `conda deactivate`.
9. Navigate to where you uploaded the tutorial notebook in your rivanna files, or upload it to your current working directory in the file navigation pane to the left of the JupyterLab GUI. (instructions for downloading the tutorial notebook from GitHub can be found in `Tutorial` section below) and click to open that notebook, or start a fresh notebook if you prefer to work off of the `Quick Start` instructions.
10. Once you open the notebook, you can set the kernel to be the environment you created by clicking on the current kernel name (upper right corner, to the left of a gray circle).
11. You should now be ready to run through the notebook!

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out to Sarah Goggin by [email](mailto:sg4dm@virginia.edu).
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

    Goggin, S.M., Zunder, E.R. ESCHR: a hyperparameter-randomized ensemble approach for robust clustering across diverse datasets. Genome Biol 25, 242 (2024). https://doi.org/10.1186/s13059-024-03386-5

[issue-tracker]: https://github.com/zunderlab/eschr/issues
[changelog]: https://eschr.readthedocs.io/latest/changelog.html
[link-docs]: https://eschr.readthedocs.io
[link-api]: https://eschr.readthedocs.io/latest/api.html
