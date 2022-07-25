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

# Usage

You can run a standard MoCHI workflow using the command line tool or a custom analysis by taking advantage of the "pymochi" package in your own python script.

MoCHI requires a table describing the measured phenotypes and how they relate to the underlying additive (biophysical) traits. The table should have the following 4 columns (see example file [here](pymochi/data/model_design_example.txt)):
 - *trait*: One or more additive trait names 
 - *transformation*: The shape of the global epistatic trend (Linear/TwoStateFractionFolded/ThreeStateFractionBound/SumOfSigmoids)
 - *phenotype*: A unique phenotype name e.g. Abundance, Binding or Kinase Activity
 - *file*: Path to DiMSum output (.RData) with variant fitness and error estimates for the corresponding phenotype (support for other input formats coming soon)

## Option A: MoCHI command line tool
   ```
   $ conda activate pymochi
   $ run_mochi.py model_design.txt
   ```

Get help with additional command line parameters:
   ```
   $ run_mochi.py -h
   ```

## Option B: Custom python script

Below is an example of a custom MoCHI workflow to infer the underlying free energies of folding and binding from [doubledeepPCA](https://www.nature.com/articles/s41586-022-04586-4) data.

1. Create a *MochiProject* object with one-hot encoded variant sequences, interaction terms and 10 cross-validation groups:
   ```
   import pymochi
   from pymochi.data import MochiData
   from pymochi.models import MochiProject
   from pymochi.report import MochiReport
   import pandas as pd

   k_folds = 10

   my_model_design = pd.DataFrame({
      'phenotype': ['Abundance', 'Binding'],
      'transformation': ['TwoStateFractionFolded', 'ThreeStateFractionBound'],
      'trait': [['Folding'], ['Folding', 'Binding']],
      'file': ["dimsum_abundance_fitness.RData", "dimsum_binding_fitness.RData"]})

   mochi_project = MochiProject(
      directory = 'my_project',
      data = MochiData(
         model_design = my_model_design,
         max_interaction_order = 1,
         k_folds = k_folds))
   ```

2. Hyperparameter tuning and model fitting:
   ```
   mochi_project.grid_search() 

   for i in range(k_folds):
      mochi_project.fit_best(fold = i+1)
   ``` 

3. Generate *MochiReport*, phenotype predictions, inferred additive trait summaries and save project:
   ```
   temperature = 30

   mochi_report = MochiReport(
      project = mochi_project,
      RT = (273+temperature)*0.001987)

   energies = mochi_project.get_additive_trait_weights(
      RT = (273+temperature)*0.001987)
    
   mochi_project.save()
   ```
   Load previously saved project:
   ```
   mochi_project = MochiProject(directory = 'my_project')
   ```
Report plots, predictions and additive trait summaries will be saved to the "my_project/report", "my_project/predictions" and "my_project/weights" subfolders.

# Manual

Comprehensive documentation is coming soon, but in the meantime get more information about specific classes/methods in python e.g.
   ```
   help(MochiData)
   ```


# Bugs and feedback

You may submit a bug report here on GitHub as an issue or you could send an email to ajfaure@gmail.com.


### Copyright

Copyright (c) 2022, Andre J Faure


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

(Vector illustration credit: <a href="https://www.vecteezy.com">Vecteezy!</a>)

