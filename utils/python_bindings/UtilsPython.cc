#include "UtilsPython.h"
#include <utils/src/Logging.h>
#include <utils/src/Version.h>

namespace thirdai::utils::python {
void createUtilsSubmodule(py::module_& module_) {
  auto utils_submodule = module_.def_submodule("utils");

  // Logging submodule
  auto logging_submodule = utils_submodule.def_submodule("logging");
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
}  // namespace thirdai::utils::python
