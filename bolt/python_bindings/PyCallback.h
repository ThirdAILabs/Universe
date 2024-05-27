#pragma once

#include <bolt/src/train/callbacks/Callback.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::python {

class PyCallback : public callbacks::Callback {
 public:
  /* Inherit the constructor */
  using callbacks::Callback::Callback;

  void onTrainBegin() override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_train_begin", /* Name of Python function */
                           onTrainBegin,     /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void onTrainEnd() override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_train_end", /* Name of Python function */
                           onTrainBegin,   /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void onEpochBegin() override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_epoch_begin", /* Name of Python function */
                           onEpochBegin,     /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void onEpochEnd() override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_epoch_end", /* Name of Python function */
                           onEpochEnd,     /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void onBatchBegin() override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_batch_begin", /* Name of Python function */
                           onBatchBegin,     /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void onBatchEnd() override {
    PYBIND11_OVERRIDE_NAME(void,           /* Return type */
                           Callback,       /* Parent class */
                           "on_batch_end", /* Name of Python function */
                           onBatchEnd,     /* Name of C++ function */
                           /* Empty list of arguments */);
  }

  void beforeUpdate() override {
    PYBIND11_OVERRIDE_NAME(void,            /* Return type */
                           Callback,        /* Parent class */
                           "before_update", /* Name of Python function */
                           beforeUpdate,    /* Name of C++ function */
                           /* Empty list of arguments */);
  }
};

}  // namespace thirdai::bolt::python