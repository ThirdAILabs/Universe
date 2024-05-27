"""
models.py used to contain Mach and the Model interface. Mach was moved to mach.py 
to resolve a circular import issue since Mach needed to import FinetunableRetriever
which in turn needed to import the Model interface class. However this caused issues
when loading models becuase pickle was looking for the import models.models.Mach, 
however it was changed to either models.mach.Mach or models.Mach. This adds back 
the classes that were originally in models.py so that any older imports still work.
"""

from .mach import Mach
from .model_interface import *
