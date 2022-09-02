#pragma once

#include <bolt/src/graph/callbacks/Callback.h>

namespace thirdai::bolt {
// For explanation of PYBIND11_OVERRIDE_PURE_NAME, see
//  https://pybind11.readthedocs.io/en/stable/advanced/classes.html#overriding-virtual-functions-in-python
//  Basically this allows us to define data loaders in python
class PyCallback : public Callback {
 public:
  /* Inherit the constructor */
  using Callback::Callback;

  void onTrainBegin(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,             /* Return type */
        Callback,         /* Parent class */
        "on_train_begin", /* Name of function in C++ (must match Python name) */
        model             /* Argument(s) */
    );
  }

  void onTrainEnd(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,           /* Return type */
        Callback,       /* Parent class */
        "on_train_end", /* Name of function in C++ (must match Python name) */
        model           /* Argument(s) */
    );
  }

  void onEpochBegin(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,             /* Return type */
        Callback,         /* Parent class */
        "on_epoch_begin", /* Name of function in C++ (must match Python name) */
        model             /* Argument(s) */
    );
  }

  void onEpochEnd(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,           /* Return type */
        Callback,       /* Parent class */
        "on_epoch_end", /* Name of function in C++ (must match Python name) */
        model           /* Argument(s) */
    );
  }

  void onBatchBegin(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,             /* Return type */
        Callback,         /* Parent class */
        "on_batch_begin", /* Name of function in C++ (must match Python name) */
        model             /* Argument(s) */
    );
  }

  void onBatchEnd(BoltGraph& model) override {
    PYBIND11_OVERRIDE_PURE(
        void,           /* Return type */
        Callback,       /* Parent class */
        "on_batch_end", /* Name of function in C++ (must match Python name) */
        model           /* Argument(s) */
    );
  }

  bool shouldStopTraining() override {
    PYBIND11_OVERRIDE_PURE(bool,                   /* Return type */
                           Callback,               /* Parent class */
                           "should_stop_training", /* Name of function in C++
                                                  (must match Python name) */
                                                   /* Empty list of arguments */
    );
  }
}
}  // namespace thirdai::bolt