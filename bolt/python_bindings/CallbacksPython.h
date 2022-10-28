#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createCallbacksSubmodule(py::module_& module);

class PyCallback : public Callback {
 public:
  /* Inherit the constructor */
  using Callback::Callback;

  void onTrainBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_train_begin", /* Name of Python function */
                           onTrainBegin,     /* Name of C++ function */
                           std::ref(model),  /* Argument(s) */
                           std::ref(train_state));
  }

  void onTrainEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,            /* Return type */
                           Callback,        /* Parent class */
                           "on_train_end",  /* Name of Python function */
                           onTrainEnd,      /* Name of C++ function */
                           std::ref(model), /* Argument(s) */
                           std::ref(train_state));
  }

  void onEpochBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_epoch_begin", /* Name of Python function */
                           onEpochBegin,     /* Name of C++ function */
                           std::ref(model),  /* Argument(s) */
                           std::ref(train_state));
  }

  void onEpochEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,            /* Return type */
                           Callback,        /* Parent class */
                           "on_epoch_end",  /* Name of Python function */
                           onEpochEnd,      /* Name of C++ function */
                           std::ref(model), /* Argument(s) */
                           std::ref(train_state));
  }

  void onBatchBegin(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,             /* Return type */
                           Callback,         /* Parent class */
                           "on_batch_begin", /* Name of Python function */
                           onBatchBegin,     /* Name of C++ function */
                           std::ref(model),  /* Argument(s) */
                           std::ref(train_state));
  }

  void onBatchEnd(BoltGraph& model, TrainState& train_state) override {
    PYBIND11_OVERRIDE_NAME(void,            /* Return type */
                           Callback,        /* Parent class */
                           "on_batch_end",  /* Name of Python function */
                           onBatchEnd,      /* Name of C++ function */
                           std::ref(model), /* Argument(s) */
                           std::ref(train_state));
  }
};

// The following callback uses py:: and PyErr symbols, which come from Python.
// Doing this in an alternate way would require these symbols to be visible in
// Graph.cc, which kind-of violates the existing structuring.
//
// Per pybind11 docs, no Ctrl-C is a Python artifact, which means standalone
// library Ctrl-C is functional:
//
//    Ctrl-C is received by the Python interpreter, and holds it until the GIL
//    is released, so a long-running function won’t be interrupted.
class KeyboardInterrupt : public Callback {
 public:
  // Check whether Ctrl-C has been called on each batch begin. This is at a
  // granularity where the users can't tell the difference, and probably does
  // not hurt speed.
  void onBatchBegin(BoltGraph& model, TrainState& train_state) final {
    Callback::onBatchBegin(model, train_state);
    if (PyErr_CheckSignals() != 0) {
      throw py::error_already_set();
    }
  }
};

using KeyboardInterruptPtr = std::shared_ptr<KeyboardInterrupt>;

}  // namespace thirdai::bolt::python