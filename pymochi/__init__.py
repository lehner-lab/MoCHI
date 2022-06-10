"""A tool to fit mechanistic models to deep mutational scanning data"""

# Add imports here
from .pymochi import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
