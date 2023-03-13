import thirdai._thirdai.telemetry
from telemetry_modifications import modified_telemetry_start
from thirdai._thirdai.telemetry import *

modified_telemetry_start()

__all__ = []
__all__.extend(dir(thirdai._thirdai.telemetry))
