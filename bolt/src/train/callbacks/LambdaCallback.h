#pragma once

#include "Callback.h"
#include <functional>
#include <utility>

namespace thirdai::bolt::callbacks {

class LambdaOnStoppedCallback : public Callback {
 public:
  explicit LambdaOnStoppedCallback(std::function<void()> lambda)
      : lambda(std::move(lambda)) {}

  void onEpochEnd() final {
    std::cout << "ON EPOCH END" << std::endl;
    if (getTrainState()->isTrainingStopped()) {
      std::cout << "IN THE THING" << std::endl;
      lambda();
    }
  }

 private:
  std::function<void()> lambda;
};

}  // namespace thirdai::bolt::callbacks