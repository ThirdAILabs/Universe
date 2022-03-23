"""
This file exposes all concrete classes defined in the data.__blocks__ directory
through a `data.blocks` module (this file).

We do this so that concrete classes can be written in separate files (so as to 
not bloat a single file) while keeping import paths fairly concise for the user.
For example, the user can do:

```
from data.blocks import TextBlock
```

but the implementation of TextBlock can be written in a text_block.py file 
instead of being crammed into a blocks.py file with every other block 
implementation.
"""

from .__blocks__.text_block import *
