#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace thirdai::bolt::python {

using MetricNameValue = std::pair<std::string, float>;

class PyDistributedComm : public DistributedComm {
 public:
  PyDistributedComm() {}

  void communicate(const ModelPtr& model) override {
    PYBIND11_OVERRIDE_PURE_NAME(void,            /* Return type */
                                DistributedComm, /* Parent class */
                                "communicate",   /* Name of Python function */
                                communicate,     /* Name of C++ function */
                                model            /* Argument(s) */
    );
  }

  uint64_t minNumBatches(uint64_t num_batches) override {
    PYBIND11_OVERRIDE_PURE_NAME(uint64_t,          /* Return type */
                                DistributedComm,   /* Parent class */
                                "min_num_batches", /* Name of Python function */
                                minNumBatches,     /* Name of C++ function */
                                num_batches        /* Argument(s) */
    );
  }

  std::vector<MetricNameValue> broadcastMetrics(
      std::vector<MetricNameValue> train_metrics) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        std::vector<MetricNameValue>, /* Return type */
        DistributedComm,              /* Parent class */
        "broadcast_metrics",          /* Name of Python function */
        broadcastMetrics,             /* Name of C++ function */
        train_metrics                 /* Argument(s) */
    );
  }
};

}  // namespace thirdai::bolt::python