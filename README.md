<p align="left">
  <img src="./Mochi.png" width="100">
</p>

# MoCHI

Welcome to the GitHub repository for MoCHI: A tool to fit mechanistic models to deep mutational scanning data.

# Installation

MoCHI is currently only available through GitHub, but we recommend using [this yaml file](pymochi.yaml) to create a dedicated Conda environment with all necessary dependencies (as explained below).

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

5. Run the demo to ensure that you have a working installation:
   ```
   $ cd ..
   $ demo_mochi.py
   ```





### Copyright

Copyright (c) 2022, Andre J Faure


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
