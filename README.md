![example workflow](https://github.com/lehner-lab/MoCHI/actions/workflows/CI.yaml/badge.svg)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/pymochi/README.html)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/pymochi/badges/version.svg?branch=master&kill_cache=1)](https://anaconda.org/bioconda/pymochi)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/pymochi/badges/latest_release_relative_date.svg?branch=master&kill_cache=1)](https://anaconda.org/bioconda/pymochi)
[![Anaconda-Server Badge](https://anaconda.org/bioconda/pymochi/badges/downloads.svg?branch=master&kill_cache=1)](https://anaconda.org/bioconda/pymochi)

<p align="left">
  <img src="./Mochi.png" width="100">
</p>

# MoCHI

Welcome to the GitHub repository for MoCHI: Neural networks to fit interpretable models and quantify energies, energetic couplings, epistasis and allostery from deep mutational scanning data.

# Table Of Contents

1. **[Installation](#installation)**
1. **[Usage](#usage)**
   1. **[Option A: MoCHI command line tool](#option-a-mochi-command-line-tool)**
   1. **[Option B: Custom Python script](#option-b-custom-python-script)**
   1. **[Demo](#demo-mochi)**
1. **[Manual](#manual)**
1. **[Bugs and feedback](#bugs-and-feedback)**
1. **[Citing MoCHI](#citing-mochi)**

# Installation

The easiest way to install MoCHI is by using the [bioconda package](http://bioconda.github.io/recipes/pymochi/README.html):
```
conda install -c bioconda pymochi
```

See the full [Installation Instructions](docs/INSTALLATION.md) for further details and alternative installation options.

# Usage

You can run a standard MoCHI workflow using the command line tool or a custom analysis by taking advantage of the "pymochi" package in your own python script.

MoCHI requires a table describing the measured phenotypes and how they relate to the underlying additive (biophysical) traits. The table should have the following 4 columns (see example file [here](pymochi/data/model_design_example.txt)):
 - *trait*: One or more additive trait names 
 - *transformation*: The shape of the global epistatic trend (Linear/ReLU/SiLU/Sigmoid/SumOfSigmoids/TwoStateFractionFolded/ThreeStateFractionBound)
 - *phenotype*: A unique phenotype name e.g. Abundance, Binding or Kinase Activity
 - *file*: Path to DiMSum output (.RData) or plain text file with variant fitness and error estimates for the corresponding phenotype

## Option A: MoCHI command line tool
```
conda activate pymochi
run_mochi.py --model_design model_design.txt
```

Get help with additional command line parameters:
```
run_mochi.py -h
```

## Option B: Custom Python script

Below is an example of a custom MoCHI workflow (written in Python) to infer the underlying free energies of folding and binding from [doubledeepPCA](https://www.nature.com/articles/s41586-022-04586-4) data.

```
#Imports
import pymochi
from pymochi.data import MochiData
from pymochi.models import MochiTask
from pymochi.report import MochiReport
import pandas as pd
from pathlib import Path

#####################
# Step 1: Create a *MochiTask* object with one-hot encoded variant sequences, interaction terms and 10 cross-validation groups
#####################

#Globals
k_folds = 10
abundance_path = str(Path(pymochi.__file__).parent / "data/fitness_abundance.txt") #MoCHI demo data
binding_path = str(Path(pymochi.__file__).parent / "data/fitness_binding.txt") #MoCHI demo data

#Define model
my_model_design = pd.DataFrame({
   'phenotype': ['Abundance', 'Binding'],
   'transformation': ['TwoStateFractionFolded', 'ThreeStateFractionBound'],
   'trait': [['Folding'], ['Folding', 'Binding']],
   'file': [abundance_path, binding_path]})

#Create Task
mochi_task = MochiTask(
   directory = 'my_task',
   data = MochiData(
      model_design = my_model_design,
      k_folds = k_folds))

#####################
# Step 2: Hyperparameter tuning and model fitting
#####################

#Perform grid search overy hyperparameters
mochi_task.grid_search() 

#Fit model using optimal hyperparameters
for i in range(k_folds):
   mochi_task.fit_best(fold = i+1)

#####################
# Step 3: Generate report, phenotype predictions, inferred additive trait summaries and save task
#####################

temperature_celcius = 30

mochi_report = MochiReport(
   task = mochi_task,
   RT = (273+temperature_celcius)*0.001987)

energies = mochi_task.get_additive_trait_weights(
   RT = (273+temperature_celcius)*0.001987)
 
mochi_task.save()
```
Report plots, predictions and additive trait summaries will be saved to the "my_task/report", "my_task/predictions" and "my_task/weights" subfolders.

## Demo MoCHI

Run the demo to ensure that you have a working MoCHI installation (expected run time <10min):
```
demo_mochi.py
```

# Manual

Comprehensive documentation is coming soon, but in the meantime get more information about specific classes/methods in python e.g.
```
help(MochiData)
```

# Bugs and feedback

You may submit a bug report here on GitHub as an issue or you could send an email to ajfaure@gmail.com.

# Citing MoCHI

Please cite the following publication if you use MoCHI:

Faure, A. J. & Lehner, B. MoCHI: neural networks to fit interpretable models and quantify energies, energetic couplings, epistasis and allostery from deep mutational scanning data. BioRxiv (2024). [10.1101/2024.01.21.575681](https://www.biorxiv.org/content/10.1101/2024.01.21.575681)

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

(Vector illustration credit: <a href="https://www.vecteezy.com">Vecteezy!</a>)
