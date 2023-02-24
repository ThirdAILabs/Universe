#include "BackgroundThread.h"
#include <chrono>
#include <iostream>

namespace thirdai::threads {

BackgroundThread::BackgroundThread(std::function<void()> func,
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

}  // namespace thirdai::threads
