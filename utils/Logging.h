#pragma once
#include <spdlog/spdlog.h>
#include <string>

// Include stdout, file sinks we use.
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace thirdai::log {

// Provide of a few sensible defaults for a logger. The following is also used
// at pybindings, editing here provides consistency in the C++ and Python API
// being the single-source of truth.
constexpr auto NAME = "thirdai";
constexpr auto DEFAULT_LOG_TO_STDERR = true;
constexpr auto DEFAULT_LOG_PATH = "";
constexpr auto DEFAULT_LOG_LEVEL = "info";
constexpr auto DEFAULT_LOG_PATTERN = "[%Y-%m-%d %T] %v";

// This configures a logger provided a string path. Client is
// expected to configure logging at the beginning.
inline void setupLogging(bool log_to_stderr = DEFAULT_LOG_TO_STDERR,
                         const std::string& path = DEFAULT_LOG_PATH,
                         const std::string& level = DEFAULT_LOG_LEVEL,
                         const std::string& pattern = DEFAULT_LOG_PATTERN) {
  try {
    using FileSink = spdlog::sinks::basic_file_sink_mt;
    using StderrSink = spdlog::sinks::stderr_color_sink_mt;

    std::vector<spdlog::sink_ptr> sinks;

    if (log_to_stderr) {
      auto stderr_sink = std::make_shared<StderrSink>();
      sinks.push_back(stderr_sink);
    }

    if (!path.empty()) {
      auto file_sink = std::make_shared<FileSink>(path, true);
      sinks.push_back(file_sink);
    }

    auto logger =
        std::make_shared<spdlog::logger>(NAME, sinks.begin(), sinks.end());

    spdlog::register_logger(logger);
    logger->set_pattern(pattern);

    // Convert a supplied string level into the corresponding
    // spdlog level and configures the logger accordingly.
    if (level == "trace")
      logger->set_level(spdlog::level::trace);
    else if (level == "debug")
      logger->set_level(spdlog::level::debug);
    else if (level == "info")
      logger->set_level(spdlog::level::info);
    else if (level == "warn")
      logger->set_level(spdlog::level::warn);
    else if (level == "err" || level == "error")
      logger->set_level(spdlog::level::err);
    else if (level == "critical")
      logger->set_level(spdlog::level::critical);
    else if (level == "off")
      logger->set_level(spdlog::level::off);
    else {
      logger->warn("Unknown log level '{}' for logger '{}'", level,
                   logger->name());
    }

  } catch (const spdlog::spdlog_ex& exception) {
    std::cerr << "Failed to initialize logger: " << exception.what()
              << std::endl;
  }
}

// Macro to prevent repetition. The desire is to achieve a syntax:
// thirdai::log::{trace,debug,info,warn,error,critical}
#define _DEFINE_RELAY_FN(level)      \
  template <class... Args>           \
  void level(Args... args) {         \
    auto logger = spdlog::get(NAME); \
    if (!logger) {                   \
      return;                        \
    }                                \
    logger->level(args...);          \
  }

// Function definitions via macros
_DEFINE_RELAY_FN(trace)
_DEFINE_RELAY_FN(debug)
_DEFINE_RELAY_FN(info)
_DEFINE_RELAY_FN(warn)
_DEFINE_RELAY_FN(error)
_DEFINE_RELAY_FN(critical)

#undef _DEFINE_RELAY_FN

}  // namespace thirdai::log
