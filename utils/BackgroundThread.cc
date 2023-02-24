#include "BackgroundThread.h"
#include <chrono>
#include <iostream>

namespace thirdai::threads {

BackgroundThread::BackgroundThread(const std::function<void()>& func,
                                   uint64_t function_run_period_ms) {
  auto repeating_func = [func, function_run_period_ms, this]() {
    while (true) {
      func();
      if (_should_terminate) {
        return;
      }

      uint64_t time_slept_so_far = 0;
      while (time_slept_so_far < function_run_period_ms) {
        uint64_t time_to_sleep =
            std::min(function_run_period_ms - time_slept_so_far,
                     TERMINATE_CHECK_PERIOD_MS);
        std::this_thread::sleep_for(std::chrono::milliseconds(time_to_sleep));
        time_slept_so_far += time_to_sleep;

        if (_should_terminate) {
          func();
          return;
        }
      }
    }
  };

  _thread = std::thread(repeating_func);
}

}  // namespace thirdai::threads
