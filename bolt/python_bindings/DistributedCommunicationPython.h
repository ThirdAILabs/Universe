#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <pybind11/pybind11.h>

namespace thirdai::bolt::train::python {

class PyDistributedComm : public DistributedComm {
 public:
  PyDistributedComm() {}

  void communicate(const bolt::nn::model::ModelPtr& model) override {
    PYBIND11_OVERRIDE_PURE_NAME(void,                     /* Return type */
                                DistributedComm, /* Parent class */
                                "communicate", /* Name of Python function */
                                communicate,   /* Name of C++ function */
                                model          /* Argument(s) */
    );
  }

  uint64_t minNumBatches(uint64_t num_batches) override {
    PYBIND11_OVERRIDE_PURE_NAME(uint64_t,                 /* Return type */
                                DistributedComm, /* Parent class */
                                "min_num_batches", /* Name of Python function */
                                minNumBatches,   /* Name of C++ function */
                                num_batches        /* Argument(s) */
    );
  }
};

}  // namespace thirdai::bolt::train::python