#pragma once

#include <bolt/src/graph/callbacks/Callback.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt {
// For explanation of PYBIND11_OVERRIDE_NAME, see
//  https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
//  Basically this allows us to define data loaders in python
class PyCallback : public Callback {
 public:
  /* Inherit the constructor */
  using Callback::Callback;

  void onTrainBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_train_begin", /* Name of Python function */
                           onTrainBegin,     /* Name of C++ function */
                           model,            /* Argument(s) */
                           train_state);
  }

  void onTrainEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_train_end", /* Name of Python function */
                           onTrainEnd,     /* Name of C++ function */
                           model,          /* Argument(s) */
                           train_state);
  }

  void onEpochBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_epoch_begin", /* Name of Python function */
                           onEpochBegin,     /* Name of C++ function */
                           model,            /* Argument(s) */
                           train_state);
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_epoch_end", /* Name of Python function */
                           onEpochEnd,     /* Name of C++ function */
                           model,          /* Argument(s) */
                           train_state);
  }

  void onBatchBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_batch_begin", /* Name of Python function */
                           onBatchBegin,     /* Name of C++ function */
                           model,            /* Argument(s) */
                           train_state);
  }

  void onBatchEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_batch_end", /* Name of Python function */
                           onBatchEnd,     /* Name of C++ function */
                           model,          /* Argument(s) */
                           train_state);
  }
};

}  // namespace thirdai::bolt