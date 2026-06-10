"""
Backward-compatibility shim: re-exports everything from the submodules.
New code should import directly from the relevant submodule.
"""

from .signal import *          # noqa: F401, F403
from .simulation import *      # noqa: F401, F403
from .graph import *           # noqa: F401, F403
from .io import *              # noqa: F401, F403
from .plotting import *        # noqa: F401, F403
from .uncertainty import *     # noqa: F401, F403
from .fitting import *         # noqa: F401, F403
from .recon import *           # noqa: F401, F403
