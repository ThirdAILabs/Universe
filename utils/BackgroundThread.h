

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>

namespace thirdai::threads {

class BackgroundThread;
using BackgroundThreadPtr = std::unique_ptr<BackgroundThread>;

/*
 * A class that will call the wrapped function once every function_run_period_ms
 * milliseconds in a std::thread that starts in the constructor. When this class
 * is destructed, the std::thread is killed and joined gracefully, preventing
 * undefined behavior.
 */
class BackgroundThread {
 public:
  // How long to sleep between checks to _should_terminate (destructing the
  // object may take up to this many milliseconds, since the thread may have
  // just started a sleep of this length)
  constexpr static uint64_t TERMINATE_CHECK_PERIOD_MS = 100;

  static BackgroundThreadPtr make(const std::function<void()>& func,
                                  uint64_t function_run_period_ms) {
    return std::unique_ptr<BackgroundThread>(
        new BackgroundThread(func, function_run_period_ms));
  }

  ~BackgroundThread() {
    _should_terminate = true;
    _thread.join();
  }

 private:
  std::atomic_bool _should_terminate = false;
  std::thread _thread;

  explicit BackgroundThread(const std::function<void()>& func,
                            uint64_t function_run_period_ms);
};

}  // namespace thirdai::threads