**[< Table Of Contents](https://github.com/lehner-lab/MoCHI#table-of-contents)**
<p align="left">
  <img src="../Mochi.png" width="100">
</p>

# Installation Instructions

## System requirements

DiMSum is expected to work on all operating systems.

## Installing MoCHI and its dependencies using Conda (recommended)

The easiest way to install MoCHI is by using the [bioconda package](http://bioconda.github.io/recipes/pymochi/README.html).

Firstly, install the [Conda](https://docs.conda.io/) package/environment management system (if you don't already have it).

On MacOS, run:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
sh Miniconda3-latest-MacOSX-x86_64.sh
```
On Linux, run:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

**IMPORTANT:** If in doubt, respond with "yes" to the following question during installation: "Do you wish the installer to initialize Miniconda3 by running conda init?". In this case Conda will modify your shell scripts (*~/.bashrc* or *~/.bash_profile*) to initialize Miniconda3 on startup. Ensure that any future modifications to your *$PATH* variable in your shell scripts occur **before** this code to initialize Miniconda3.

After installing Conda you will need to add the bioconda channel as well as the other channels bioconda depends on. Start a new console session (e.g. by closing the current window and opening a new one) and run the following:
```
conda config --add channels defaults
conda config --add channels bioconda
```

Next, create a dedicated Conda environment to install the [MoCHI bioconda package](http://bioconda.github.io/recipes/pymochi/README.html) and it's dependencies:
```
conda create -n pymochi pymochi
conda activate pymochi
```
**TIP:** See [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more information about managing conda environments.

To check that you have a working installation of MoCHI, run the [Demo](#demo-mochi)

## Installing MoCHI from source (GitHub)

Installing MoCHI from source is not recommended. The easiest way to install MoCHI (and its dependencies) is by using the [MoCHI bioconda package](http://bioconda.github.io/recipes/pymochi/README.html). See [Installing MoCHI and its dependencies using Conda](#installing-mochi-and-its-dependencies-using-conda-recommended).

This [this yaml file](../pymochi.yaml) can be used to create a dedicated Conda environment with all necessary dependencies (as explained below).

1. Install the [Conda](https://docs.conda.io/) package/environment management system (if you already have Conda skip to step 2):

   On MacOS, run:
   ```
   $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
   $ sh Miniconda3-latest-MacOSX-x86_64.sh
   ```
   On Linux, run:
   ```
   $ curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   $ sh Miniconda3-latest-Linux-x86_64.sh
   ```

   **IMPORTANT:** If in doubt, respond with "yes" to the following question during installation: "Do you wish the installer to initialize Miniconda3 by running conda init?". In this case Conda will modify your shell scripts (*~/.bashrc* or *~/.bash_profile*) to initialize Miniconda3 on startup. Ensure that any future modifications to your *$PATH* variable in your shell scripts occur **before** this code to initialize Miniconda3.

2. Clone the MoCHI GitHub repository:
   ```
   $ git clone https://github.com/lehner-lab/MoCHI.git
   ```

3. Create the pymochi Conda environment:
   ```
   $ conda env create -f MoCHI/pymochi.yaml
   ```

4. Install MoCHI:
   ```
   $ conda activate pymochi
   $ cd MoCHI
   $ pip install -e ./
   ```

## Demo MoCHI

Run the demo to ensure that you have a working MoCHI installation (expected run time <10min):
   ```
   $ demo_mochi.py
   ```
