#include <utils/Logging.h>
#include <utils/Version.h>
#include <iostream>
#include <string>

// Include stdout, file sinks we use.
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace thirdai::log {
// This configures a logger provided a string path. Client is
// expected to configure logging at the beginning.
void setupLogging(bool log_to_stderr /*= DEFAULT_LOG_TO_STDERR*/,
                  const std::string& path /*= DEFAULT_LOG_PATH*/,
                  const std::string& level /*= DEFAULT_LOG_LEVEL*/,
                  const std::string& pattern /*= DEFAULT_LOG_PATTERN*/) {
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
    if (level == "trace") {
      logger->set_level(spdlog::level::trace);
    } else if (level == "debug") {
      logger->set_level(spdlog::level::debug);
    } else if (level == "info") {
      logger->set_level(spdlog::level::info);
    } else if (level == "warn") {
      logger->set_level(spdlog::level::warn);
    } else if (level == "err" || level == "error") {
      logger->set_level(spdlog::level::err);
    } else if (level == "critical") {
      logger->set_level(spdlog::level::critical);
    } else if (level == "off") {
      logger->set_level(spdlog::level::off);
    } else {
      logger->warn("Unknown log level '{}' for logger '{}'", level,
                   logger->name());
    }
    // At this point the client has requested logging.
    if (logger) {
      logger->info("thirdai {}", version());
    }

  } catch (const spdlog::spdlog_ex& exception) {
    std::cerr << "Failed to initialize logger: " << exception.what()
              << std::endl;
  }
}

}  // namespace thirdai::log
