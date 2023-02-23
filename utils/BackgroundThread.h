

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

namespace thirdai::threads {

class BackgroundThread;
using BackgroundThreadPtr = std::unique_ptr<BackgroundThread>;

class BackgroundThread : public std::thread {
 public:
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
                            uint64_t function_run_period_ms) {
    auto repeating_func = [func, function_run_period_ms, this]() {
      while (true) {
        func();
        if (_should_terminate) {
          break;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(function_run_period_ms));
      }
    };

    _thread = std::thread(repeating_func);
  }
};

}  // namespace thirdai::threads