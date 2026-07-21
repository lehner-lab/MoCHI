# Development, testing, and deployment tools

This directory contains a collection of tools for running Continuous Integration (CI) tests, 
conda installation, and other development tools not directly related to the coding process.


## Manifest

### Continuous Integration

You should test your code, but do not feel compelled to use these specific programs. You also may not need Unix and 
Windows testing if you only plan to deploy on specific platforms. These are just to help you get started.

The items in this directory have been left for legacy purposes since the change to GitHub Actions, 
They will likely be removed in a future version.

* `legacy-miniconda-setup`: A preserved copy of a helper directory which made Linux and OSX based testing through [Travis-CI](https://about.travis-ci.com/) simpler
  * `before_install.sh`: Pip/Miniconda pre-package installation script for Travis. No longer needed thanks to 
    [GitHub Actions](https://docs.github.com/en/free-pro-team@latest/actions) and the [conda-incubator/setup-miniconda Action](https://github.com/conda-incubator/setup-miniconda)

### Conda Environment:

This directory contains the files to setup the Conda environment for testing purposes

* `conda-envs`: directory containing the YAML file(s) which fully describe Conda Environments, their dependencies, and those dependency provenance's
  * `test_env.yaml`: Simple test environment file with base dependencies. Channels are not specified here and therefore respect global Conda configuration
  
### Additional Scripts:

This directory contains OS agnostic helper scripts which don't fall in any of the previous categories
* `scripts`
  * `create_conda_env.py`: Helper program for spinning up new conda environments based on a starter file with Python Version and Env. Name command-line options


## How to contribute changes
- Clone the repository if you have write access to the main repo, fork the repository if you are a collaborator.
- Make a new branch with `git checkout -b {your branch name}`
- Make changes and test your code
- Ensure that the test environment dependencies (`conda-envs`) line up with the build and deploy dependencies (`conda-recipe/meta.yaml`)
- Push the branch to the repo (either the main or your fork) with `git push -u origin {your branch name}`
  * Note that `origin` is the default name assigned to the remote, yours may be different
- Make a PR on GitHub with your changes
- We'll review the changes and get your code into the repo after lively discussion!


## Checklist for updates
- [ ] Make sure there is an/are issue(s) opened for your specific update
- [ ] Create the PR, referencing the issue
- [ ] Debug the PR as needed until tests pass
- [ ] Tag the final, debugged version 
   *  `git tag -a X.Y.Z [latest pushed commit] && git push --follow-tags`
- [ ] Get the PR merged in

## Git-tag Auto-version
[setuptools-scm](https://github.com/pypa/setuptools-scm) automatically infers the installed version
from the repository's `git` tags and how many commits ahead the working tree is. The format follows
[PEP 440](https://peps.python.org/pep-0440/).

If the current commit matches a release tag, the installed version matches that tag, for example
`pymochi-1.1`. Otherwise the installed version becomes a development version derived from the most
recent tag and the current commit, for example `pymochi-1.2.dev24+gbc15a0cb7.d20260407`.
