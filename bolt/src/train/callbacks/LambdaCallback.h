#pragma once

#include "Callback.h"
#include <functional>
#include <utility>

namespace thirdai::bolt::callbacks {

// Calls a given lambda function at specified intervals throughout training.
// Defaults to calling on epoch end.
class LambdaCallback : public Callback {
 public:
  explicit LambdaCallback(std::function<void()> lambda,
                          bool on_train_begin = false,
                          bool on_train_end = false,
                          bool on_epoch_begin = false, bool on_epoch_end = true,
                          bool on_batch_begin = false,
                          bool on_batch_end = false, bool before_update = false)
      : lambda(std::move(lambda)),
        on_train_begin(on_train_begin),
        on_train_end(on_train_end),
        on_epoch_begin(on_epoch_begin),
        on_epoch_end(on_epoch_end),
        on_batch_begin(on_batch_begin),
        on_batch_end(on_batch_end),
        before_update(before_update) {}

  void onTrainBegin() final {
    if (on_train_begin) {
      lambda();
    }
  }

  void onTrainEnd() final {
    if (on_train_end) {
      lambda();
    }
  }

  void onEpochBegin() final {
    if (on_epoch_begin) {
      lambda();
    }
  }

  void onEpochEnd() final {
    if (on_epoch_end) {
      lambda();
    }
  }

  void onBatchBegin() final {
    if (on_batch_begin) {
      lambda();
    }
  }

  void onBatchEnd() final {
    if (on_batch_end) {
      lambda();
    }
  }

  void beforeUpdate() final {
    if (before_update) {
      lambda();
    }
  }

 private:
  std::function<void()> lambda;
  bool on_train_begin, on_train_end, on_epoch_begin, on_epoch_end,
      on_batch_begin, on_batch_end, before_update;
};

}  // namespace thirdai::bolt::callbacks