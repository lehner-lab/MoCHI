<p align="left">
  <img src="./Mochi.png" width="100">
</p>

# Overview

Welcome to the GitHub repository for MoCHI: A module for calculating higher-order genetic interactions.

# Required Software

To run the DiMSum pipeline you will need the following software and associated packages:

* **[R](https://www.r-project.org/) >=v3.5.2** (data.table, ggplot2, optparse, parallel, plyr, reshape2, stringr)

# Installation and loading

Open R and enter:

```
# Install
if(!require(devtools)) install.packages("devtools")
devtools::install_github("lehner-lab/MoCHI")

# Load
library(MoCHI)

# Help
?mochi
```

# MoCHI command-line tool

Clone the MoCHI repository and install the R package locally. The * must be replaced by what is actually downloaded and built.

```
git clone https://github.com/lehner-lab/MoCHI.git
R CMD build MoCHI
R CMD INSTALL MoCHI_*.tar.gz
```
Add the cloned MoCHI repository base directory to your path. You can do this by adding the following line at the bottom of your ~/.bashrc file:
```
export PATH=CLONED_MOCHI_REPOSITORY:$PATH
```
Get a description of MoCHI command-line arguments with the following:
```
MoCHI -h
```




(Vector illustration credit: <a href="https://www.vecteezy.com">Vecteezy!</a>)
