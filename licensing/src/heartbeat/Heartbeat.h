#pragma once

#include <atomic>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
namespace thirdai::licensing {

// Every 1 second we do a heartbeat with the server and check to see if we
// should stop
constexpr uint32_t HEARTBEAT_PERIOD_SECONDS = 1;

// This is the maximum allowable value for _heartbeat_lag_timeout. The user can
// set the time limit to be lower than this, e.g. for testing or if they want
// their system to fail fast if the licensing server goes down, but they cannot
// set it to be more than this.
constexpr uint32_t MAX_NO_HEARTBEAT_TIME_LIMIT = 10000;

class HeartbeatThread {
 public:
  explicit HeartbeatThread(const std::string& url,
                           const std::optional<uint32_t>& heartbeat_timeout);

  ~HeartbeatThread();

  void verify();

  void terminate();

 private:
  void heartbeatThread(const std::string& url);

  bool doSingleHeartbeat(const std::string& url);

  std::string _machine_id;
  std::atomic_bool _verified;
  std::atomic_bool _should_terminate;
  std::thread _heartbeat_thread;
  int32_t _no_heartbeat_time_limit;
};

}  // namespace thirdai::licensing