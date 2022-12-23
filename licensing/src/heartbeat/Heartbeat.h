#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
namespace thirdai::licensing {

// Every 1 second we check to see if we should stop
// Every 100 seconds we do a heartbeat with the server
// If 10000 seconds go by without a success, verify will throw an exception
// TODO(Josh): Make customizable
constexpr uint32_t VALIDATION_FAIL_TIMEOUT_SECONDS = 10000;
constexpr uint32_t HEARTBEAT_PERIOD_SECONDS = 100;
constexpr uint32_t THREAD_SLEEP_PERIOD_SECONDS = 1;

class HeartbeatThread {
 public:
  explicit HeartbeatThread(const std::string& url);

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
};

}  // namespace thirdai::licensing