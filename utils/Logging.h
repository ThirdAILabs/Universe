#pragma once
#include <spdlog/spdlog.h>
#include <string>

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
void setupLogging(bool log_to_stderr = DEFAULT_LOG_TO_STDERR,
                  const std::string& path = DEFAULT_LOG_PATH,
                  const std::string& level = DEFAULT_LOG_LEVEL,
                  const std::string& pattern = DEFAULT_LOG_PATTERN);

// Macro to prevent repetition. The desire is to achieve a syntax:
// thirdai::log::{trace,debug,info,warn,error,critical}
//
// With no modifications this would be spdlog::info or spdlog::warn. We make the
// syntax at call sites thirdai::log::info or thirdai::log::warn. When within
// the thirdai namespace, one may simply use log::info or log::warn, omitting
// thirdai.
#define DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(level) \
  template <class... Args>                             \
  void level(Args... args) {                           \
    auto logger = spdlog::get(NAME);                   \
    if (!logger) {                                     \
      return;                                          \
    }                                                  \
    logger->level(args...);                            \
  }

// Function definitions via macros
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(trace)
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(debug)
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(info)
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(warn)
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(error)
DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION(critical)

#undef DEFINE_THIRDAI_TO_SPDLOG_RELAY_FUNCTION

}  // namespace thirdai::log
