"""
======
railFE
======
Rail Vehicle Dynamics simulation software.

"""

from ._version import __version__
from . import MatrixAssemblyOperations
from . import UnitConversions
from . import Newmark_NL
from . import TimoshenkoBeamModel
from . import TrackModelAssembly
from . import VehicleModelAssembly
from . import SystemAssembly

__name__ = 'railFE'