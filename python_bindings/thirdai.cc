// Code to create thirdai modules
#include <bolt/python_bindings/BoltNNPython.h>
#include <bolt/python_bindings/BoltPython.h>
#include <bolt/python_bindings/BoltV2NNPython.h>
#include <bolt/python_bindings/BoltV2TrainPython.h>
#include <bolt/python_bindings/CallbacksPython.h>
#include <hashing/python_bindings/HashingPython.h>
#include <auto_ml/python_bindings/AutomlPython.h>
#include <dataset/python_bindings/DatasetPython.h>
#include <licensing/python_bindings/LicensingPython.h>
#include <new_dataset/python_bindings/DatasetPython.h>
#include <new_dataset/python_bindings/FeaturizationPython.h>
#include <search/python_bindings/DocSearchPython.h>
#include <telemetry/python_bindings/TelemetryPython.h>
#include <utils/Logging.h>
#include <utils/Version.h>

// Pybind11 library
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifndef __clang__
#include <omp.h>
#endif

void createLoggingSubmodule(py::module_& module_) {
  // Logging submodule
  auto logging_submodule = module_.def_submodule("logging");

  logging_submodule.def(
      "setup", &thirdai::logging::setup,
      py::arg("log_to_stderr") = thirdai::logging::DEFAULT_LOG_TO_STDERR,
      py::arg("path") = thirdai::logging::DEFAULT_LOG_PATH,
      py::arg("level") = thirdai::logging::DEFAULT_LOG_LEVEL,
      py::arg("pattern") = thirdai::logging::DEFAULT_LOG_PATTERN,
      py::arg("flush_interval") = thirdai::logging::DEFAULT_LOG_FLUSH_INTERVAL,
      R"pbdoc(
        Set up log for thirdai C++ library.

        :param log_to_stderr: Print logs to standard error. Turned off 
               (false) by default.
        :type log_to_stderr: bool
        :param path: Path to file to write logs to. Empty (default) implies 
               no file log.
        :type path: str
        :param level: Print logs upto this level. Choices are
              trace,debug,info,warn,critical,error,off. Default is info.
        :type level: str
        :param pattern: Pattern string to customize log from client. See 
               https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for using 
               format-strings.
        :type pattern: str
        :param flush_interval: Interval in seconds at which logs will be flushed while in operation.
        :type flush_interval: int
        )pbdoc");

  logging_submodule.def(
      "critical",
      [](const std::string& logline) { thirdai::logging::critical(logline); },
      R"pbdoc(Write to logs with level critical.)pbdoc");

  logging_submodule.def(
      "error",
      [](const std::string& logline) { thirdai::logging::error(logline); },
      R"pbdoc(Write to logs with level error.)pbdoc");

  logging_submodule.def(
      "warn",
      [](const std::string& logline) { thirdai::logging::warn(logline); },
      R"pbdoc(Write to logs with level warn.)pbdoc");

  logging_submodule.def(
      "info",
      [](const std::string& logline) { thirdai::logging::info(logline); },
      R"pbdoc(Write to logs with level info.)pbdoc");

  logging_submodule.def(
      "debug",
      [](const std::string& logline) { thirdai::logging::debug(logline); },
      R"pbdoc(Write to logs with level debug.)pbdoc");

  logging_submodule.def(
      "trace",
      [](const std::string& logline) { thirdai::logging::trace(logline); },
      R"pbdoc(Write to logs with level trace.)pbdoc");

  logging_submodule.def(
      "flush", []() { thirdai::logging::flush(); },
      R"pbdoc(Force a flush on the logger.)pbdoc");
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

  m.attr("__version__") = thirdai::version();

  createLoggingSubmodule(m);

  // Per pybind11 docs breaking up the construction of bindings in this way
  // could speed up build times. See below for more info:
  // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-reduce-the-build-time
  // TODO(Josh/Nick): Deprecate this call and change NewDataset/new_dataset to
  // Dataset/dataset everyone in the codebase.
  thirdai::dataset::python::createDatasetSubmodule(m);

  // Licensing Submodule
#if THIRDAI_CHECK_LICENSE
  thirdai::licensing::python::createLicensingSubmodule(m);
#endif

  // Telemetry submodule
  thirdai::telemetry::python::createTelemetrySubmodule(m);

  // Data Submodule
  auto data_submodule = m.def_submodule("data");
  thirdai::dataset::python::createDataSubmodule(data_submodule);
  thirdai::data::python::createFeaturizationSubmodule(data_submodule);

  // Hashing Submodule
  thirdai::hashing::python::createHashingSubmodule(m);

  // Bolt Submodule
  auto bolt_submodule = m.def_submodule("bolt");
  thirdai::bolt::python::createBoltSubmodule(bolt_submodule);
  thirdai::bolt::python::createBoltNNSubmodule(bolt_submodule);
  thirdai::bolt::python::createCallbacksSubmodule(bolt_submodule);

  // Automl in Bolt
  thirdai::automl::python::defineAutomlInModule(bolt_submodule);
  thirdai::automl::python::createModelsSubmodule(bolt_submodule);
  thirdai::automl::python::createUDTTypesSubmodule(bolt_submodule);
  thirdai::automl::python::createUDTTemporalSubmodule(bolt_submodule);

  // Search Submodule
  thirdai::search::python::createSearchSubmodule(m);

  // Deployment submodule
  thirdai::automl::python::createDeploymentSubmodule(m);


  // Bolt V2
  auto bolt_v2_submodule = m.def_submodule("bolt_v2");
#if THIRDAI_EXPOSE_ALL
  thirdai::bolt::nn::python::createBoltV2NNSubmodule(bolt_v2_submodule);
  thirdai::bolt::train::python::createBoltV2TrainSubmodule(bolt_v2_submodule);
#endif
}
