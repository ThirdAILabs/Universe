#include <utils/Logging.h>
#include <utils/Version.h>
#include <chrono>
#include <iostream>
#include <string>

// Include stdout, file sinks we use.
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace thirdai::logging {

void setup(bool log_to_stderr /*= DEFAULT_LOG_TO_STDERR*/,
           const std::string& path /*= DEFAULT_LOG_PATH*/,
           const std::string& level /*= DEFAULT_LOG_LEVEL*/,
           const std::string& pattern /* = DEFAULT_LOG_PATTERN*/,
           uint32_t flush_interval /*= DEFAULT_LOG_FLUSH_INTERVAL*/) {
  try {
    // Flush any existing loggers. Useful when setup is called multiple times.
    flush();

    // Shutdown existing loggers, so we can start clean-slate
    spdlog::shutdown();

    // Construct new loggers.
    using FileSink = spdlog::sinks::basic_file_sink_mt;
    using StderrSink = spdlog::sinks::stderr_color_sink_mt;

    std::vector<spdlog::sink_ptr> sinks;

    if (log_to_stderr) {
      auto stderr_sink = std::make_shared<StderrSink>();
      sinks.push_back(stderr_sink);
    }

    if (!path.empty()) {
      auto file_sink = std::make_shared<FileSink>(path, /*truncate=*/false);
      sinks.push_back(file_sink);
    }

    auto logger =
        std::make_shared<spdlog::logger>(NAME, sinks.begin(), sinks.end());

    spdlog::register_logger(logger);
    logger->set_pattern(pattern);

    // This ensures that logs are periodically flushed.
    spdlog::flush_every(std::chrono::seconds(flush_interval));
/*
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
*/
    std::cerr << level << std::endl; 
  } catch (const spdlog::spdlog_ex& exception) {
    std::cerr << "Failed to initialize logger: " << exception.what()
              << std::endl;
  }
}

}  // namespace thirdai::logging
