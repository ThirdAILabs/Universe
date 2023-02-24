

#include <atomic>
#include <iostream>
#include <memory>
#include <thread>

namespace thirdai::threads {

class BackgroundThread;
using BackgroundThreadPtr = std::unique_ptr<BackgroundThread>;

class BackgroundThread {
 public:
  static BackgroundThreadPtr make(std::function<void()> func,
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

  explicit BackgroundThread(std::function<void()> func,
                            uint64_t function_run_period_ms);
};

}  // namespace thirdai::threads