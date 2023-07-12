#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedCommInterface.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::train::python {

class PyDistributedComm : public DistributedCommInterface {
 public:
  PyDistributedComm() {}

  void communicate(const bolt::nn::model::ModelPtr& model) override {
    PYBIND11_OVERRIDE_PURE_NAME(void,                     /* Return type */
                                DistributedCommInterface, /* Parent class */
                                "communicate", /* Name of Python function */
                                communicate,   /* Name of C++ function */
                                model          /* Argument(s) */
    );
  }

  uint64_t min_num_batches(uint64_t num_batches) override {
    PYBIND11_OVERRIDE_PURE_NAME(uint64_t,                 /* Return type */
                                DistributedCommInterface, /* Parent class */
                                "min_num_batches", /* Name of Python function */
                                min_num_batches,   /* Name of C++ function */
                                num_batches        /* Argument(s) */
    );
  }
};

}  // namespace thirdai::bolt::train::python