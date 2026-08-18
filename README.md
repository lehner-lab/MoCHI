![example workflow](https://github.com/lehner-lab/MoCHI/actions/workflows/CI.yaml/badge.svg)

<p align="left">
  <img src="./Mochi.png" width="100">
</p>

# MoCHI

Welcome to the GitHub repository for MoCHI: Neural networks to fit interpretable models and quantify energies, energetic couplings, epistasis, and allostery from deep mutational scanning data.

# Table Of Contents

1. **[Installation](#installation)**
1. **[Usage](#usage)**
   1. **[Default: Nextflow](#default-nextflow)**
   1. **[Direct command line tool](#direct-command-line-tool)**
   1. **[Custom Python script](#custom-python-script)**
   1. **[Demo](#demo-mochi)**
1. **[Manual](#manual)**
1. **[Bugs and feedback](#bugs-and-feedback)**
1. **[Citing MoCHI](#citing-mochi)**

# Installation

MoCHI uses [uv](https://docs.astral.sh/uv/) to create its Python environment and install the locked dependencies. From a clone of this repository, run:

```bash
cd MoCHI
bash bootstrap_mochi_uv.sh
```

This installs uv if necessary, creates `.venv`, and synchronizes the environment from `pyproject.toml` and `uv.lock`.

The [Bioconda package](http://bioconda.github.io/recipes/pymochi/README.html) remains available for the legacy MoCHI 1.1 release on Python 3.9:

```bash
conda create -n pymochi -c conda-forge -c bioconda pymochi
```

It does not install the current source version or its locked environment, so uv is recommended for current development and Nextflow runs.

# Usage

The default way to run a standard MoCHI workflow is through Nextflow. The command line tool remains available for direct runs, and the `pymochi` package can be used for custom analyses.

MoCHI requires a plain text model design file containing a table describing the measured phenotypes and how they relate to the underlying additive (biophysical) traits. The table should have the following 4 tab-separated columns (see example [here](pymochi/data/model_design_example.txt)):
 - `trait`: One or more additive trait names 
 - `transformation`: The shape of the global epistatic trend (Linear/ReLU/SiLU/Sigmoid/SumOfSigmoids/TwoStateFractionFolded/ThreeStateFractionBound)
 - `phenotype`: A unique phenotype name e.g. Abundance, Binding or Kinase Activity
 - `file`: Path to DiMSum output (.RData) or plain text file with variant fitness and error estimates for the corresponding phenotype(s) (nucleotide sequence example [here](https://github.com/lehner-lab/MoCHI/blob/master/pymochi/data/fitness_example_nt.txt), amino acid sequence example [here](https://github.com/lehner-lab/MoCHI/blob/master/pymochi/data/fitness_example_aa.txt))

## Default: Nextflow

After [installing with uv](#installation), install [Nextflow](https://www.nextflow.io/) and Java, then run the portable local profile:

```bash
cd MoCHI
bash nextflow/scripts/run_mochi_nextflow.sh \
    --run_name my-mochi-run \
    --model_design /path/to/model_design.tsv
```

The default `local` profile runs tasks on the current host. For LSF, set `NEXTFLOW_PROFILE=lsf` and use the LSF master launcher. See [the Nextflow run guide](nextflow/RUN.md) for scheduler configuration, resume instructions, and additional options.

## Direct command line tool

Replace `MY_MODEL` with the path to your model design file (see example [here](pymochi/data/model_design_example.txt)).

```bash
uv run run_mochi.py --model_design MY_MODEL
```

Get help with additional command line parameters:

```bash
uv run run_mochi.py -h
```

## Custom Python script

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
Report plots, predictions and additive trait summaries will be saved to the `my_task/report`, `my_task/predictions` and `my_task/weights` subfolders.

## Demo MoCHI

Run the demo to ensure that you have a working MoCHI installation (expected run time <10min):

```bash
uv run demo_mochi.py
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

Faure, A. J. & Lehner, B. MoCHI: neural networks to fit interpretable models and quantify energies, energetic couplings, epistasis, and allostery from deep mutational scanning data. Genome Biol 25, 303 (2024). [10.1186/s13059-024-03444-y](https://doi.org/10.1186/s13059-024-03444-y)

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.

(Vector illustration credit: <a href="https://www.vecteezy.com">Vecteezy!</a>)
