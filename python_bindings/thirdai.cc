// Code to create thirdai modules
#include <bolt/python_bindings/BoltPython.h>
#include <hashing/python_bindings/HashingPython.h>
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

void createLoggingSubmodule(py::module_& module_) {
  // Logging submodule
  auto logging_submodule = module_.def_submodule("logging");

  logging_submodule.def(
      "setup", &thirdai::log::setupLogging,
      py::arg("log_to_stderr") = thirdai::log::DEFAULT_LOG_TO_STDERR,
      py::arg("path") = thirdai::log::DEFAULT_LOG_PATH,
      py::arg("level") = thirdai::log::DEFAULT_LOG_LEVEL,
      py::arg("pattern") = thirdai::log::DEFAULT_LOG_PATTERN,
      R"pbdoc(
        Set up logging for thirdai C++ library.

        :param log_to_stderr: Print logs to standard error. Turned off 
               (false) by default.
        :type log_to_stderr: bool
        :param path: Path to file to write logs to. Empty (default) implies 
               no file logging.
        :type path: str
        :param level: Print logs upto this level. Choices are
              trace,debug,info,warn,critical,error,off. Default is info.
        :type level: str
        :param pattern: Pattern string to customize logging from client. See 
               https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for using 
               format-strings.
        :type pattern: str
        )pbdoc");

  logging_submodule.def(
      "critical",
      [](const std::string& logline) { thirdai::log::critical(logline); },
      R"pbdoc(Write to logs with level critical.)pbdoc");

  logging_submodule.def(
      "error", [](const std::string& logline) { thirdai::log::error(logline); },
      R"pbdoc(Write to logs with level error.)pbdoc");

  logging_submodule.def(
      "warn", [](const std::string& logline) { thirdai::log::warn(logline); },
      R"pbdoc(Write to logs with level warn.)pbdoc");

  logging_submodule.def(
      "info", [](const std::string& logline) { thirdai::log::info(logline); },
      R"pbdoc(Write to logs with level info.)pbdoc");

  logging_submodule.def(
      "debug", [](const std::string& logline) { thirdai::log::debug(logline); },
      R"pbdoc(Write to logs with level debug.)pbdoc");

  logging_submodule.def(
      "trace", [](const std::string& logline) { thirdai::log::trace(logline); },
      R"pbdoc(Write to logs with level trace.)pbdoc");
}

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

  m.attr("__version__") = thirdai::version();

  createLoggingSubmodule(m);

  // Per pybind11 docs breaking up the construction of bindings in this way
  // could speed up build times. See below for more info:
  // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time
  thirdai::dataset::python::createDatasetSubmodule(m);

  thirdai::hashing::python::createHashingSubmodule(m);

  thirdai::bolt::python::createBoltSubmodule(m);

  thirdai::search::python::createSearchSubmodule(m);
}
