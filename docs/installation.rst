Installation
============

You need to have a Python version between 3.8 and 3.10 (inclusive) installed on your system. If you don't have
Python installed, we recommend installing `Mambaforge <https://github.com/conda-forge/miniforge#mambaforge>`_.
(*If you are installing on a Windows OS, you will need to use python version 3.8 only due to issues with installing the nmslib dependency in other versions*)

1. Make sure you have Anaconda installed and functional. `Conda FAQ <https://docs.anaconda.com/anaconda/user-guide/faq/>`_ is a great resource for troubleshooting and verifying that everything is working properly.
2. Open terminal or equivalent command line interface and run ``conda create --name <env_name> python=<version>`` (e.g. ``conda create --name eschrEnv python=3.8``)
3. Activate the environment by running ``conda activate <env_name>``
4. Once environment is activated, run ``conda install pip``
5. If you do not have Git installed, run ``conda install git``
6. To install ESCHR into your conda environment, run the following line:
   ``pip install git+https://github.com/zunderlab/eschr.git``
7. Verify that the Conda environment was created successfully by running ``conda list`` and verifying that expected packages are installed for this environment. Then either close the environment by running ``conda deactivate`` or proceed to subsequent optional setup and/or running the method within the environment.
