"""
This file exposes all concrete classes defined in the data.__sources__ directory
through a `data.sources` module (this file).

We do this so that concrete classes can be written in separate files (so as to 
not bloat a single file) while keeping import paths fairly concise for the user.
For example, the user can do:

```
from data.sources import InMemoryCollection
```

but the implementation of InMemoryCollection can be written in a 
python_sources.py file instead of being crammed into a sources.py file with 
every other source location or source format implementation.
"""

from .__sources__.in_memory import *
from .__sources__.file_system import *
from .__sources__.csv_iterable import *
