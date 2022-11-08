thirdai.logging
===============

The thirdai internal logging library uses
`spdlog <https://github.com/gabime/spdlog>`_ to do logging within the C++
library. This requires a setup at the start of your program. Further capability
to write to these configured logs are exposed in Python via functions so that
C++ library and Python may have sychronized single place to store logs of the
program.

Setup
-----

The following function can be used to setup logging at the start of the program.

.. autofunction:: thirdai.logging.setup

Logging
-------

The following are python bindings to the `log::<level>` functions available in
the thirdai C++ library.

.. autofunction:: thirdai.logging.trace
.. autofunction:: thirdai.logging.debug
.. autofunction:: thirdai.logging.info
.. autofunction:: thirdai.logging.warn
.. autofunction:: thirdai.logging.error
.. autofunction:: thirdai.logging.critical
