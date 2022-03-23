"""
This file exposes all concrete classes defined in the data.__core__ directory
through a `data.core` module (this file).

We do this so that concrete classes can be written in separate files (so as to 
not bloat a single file) while keeping import paths fairly concise for the user.
For example, the user can do:

```
from data.core import Schema
```

but the implementation of Schema can be written in a schema.py file instead of 
being crammed into a core.py file with Dataset, which has a much longer 
implementation.
"""

from .__core__.dataset import *
from .__core__.schema import *
