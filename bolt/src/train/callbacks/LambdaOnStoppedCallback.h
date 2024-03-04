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
    if (getTrainState()->isTrainingStopped()) {
      lambda();
    }
  }

 private:
  std::function<void()> lambda;
};

}  // namespace thirdai::bolt::callbacks