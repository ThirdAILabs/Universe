// Code to create thirdai modules
#include <bolt/python_bindings/BoltPython.h>
#include <hashing/python_bindings/HashingPython.h>
#include <auto_classifiers/python_bindings/AutoClassifiersPython.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <search/python_bindings/DocSearchPython.h>
#include <utils/Logging.h>
#include <utils/Version.h>

// Pybind11 library
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Licensing wrapper
#include <wrappers/src/LicenseWrapper.h>

#ifndef __clang__
#include <omp.h>
#endif

// TODO(all): Figure out naming convention for python exposed classes and
// methods
// TODO(any): Add docstrings to methods
// TODO(any): Can we remove redudancy in the bindings?
PYBIND11_MODULE(_thirdai, m) {  // NOLINT

#ifndef __clang__
  m.def("set_global_num_threads", &omp_set_num_threads,
        py::arg("max_num_threads"),
        "Set the maximum number of threads to use to any future calls to the "
        "thirdai library.");
#endif

#if THIRDAI_CHECK_LICENSE
  m.def("set_thirdai_license_path",
        &thirdai::licensing::LicenseWrapper::setLicensePath,
        py::arg("license_path"),
        "Set a license filepath for any future calls to the thirdai library.");
#endif

  m.def("setup_logging", &thirdai::logging::setupLogging,
        py::arg("log_to_stderr") = thirdai::logging::DEFAULT_LOG_TO_STDERR,
        py::arg("path") = thirdai::logging::DEFAULT_LOG_PATH,
        py::arg("level") = thirdai::logging::DEFAULT_LOG_LEVEL,
        py::arg("pattern") = thirdai::logging::DEFAULT_LOG_PATTERN,
        "Set up logging for thirdai C++ library.\n"
        "  log_to_stderr: bool - Print logs to standard error. Turned off "
        "(false) by default.\n"
        "  path: str - Path to file to write logs to. Empty (default) implies "
        "no file logging.\n"
        "  level: str - Print logs upto this level. Choices are "
        "trace,debug,info,warn,critical,error,off. Default is info.\n"
        "  pattern: str - Pattern string to customize logging from client. See "
        "https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for using "
        "format-strings.");

  m.attr("__version__") = thirdai::version();

  // Per pybind11 docs breaking up the construction of bindings in this way
  // could speed up build times. See below for more info:
  // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time
  thirdai::dataset::python::createDatasetSubmodule(m);

  thirdai::hashing::python::createHashingSubmodule(m);

  auto bolt_submodule = thirdai::bolt::python::createBoltSubmodule(m);

  thirdai::bolt::python::defineAutoClassifeirsInModule(bolt_submodule);

  thirdai::search::python::createSearchSubmodule(m);
}
