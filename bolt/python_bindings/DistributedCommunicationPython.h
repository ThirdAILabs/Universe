#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedCommInterface.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

class DistributedCommPython final
    : public DistributedCommInterface,
      public std::enable_shared_from_this<DistributedCommPython> {
 public:
  explicit DistributedCommPython(py::object& py_instance);

  void communicate(const bolt::nn::model::ModelPtr& model) final;

  uint64_t min_num_batches(uint64_t num_batches) final;

  std::optional<std::shared_ptr<DistributedCommPython>> to_optional();

 private:
  py::object py_instance;
};

}  // namespace thirdai::bolt::train::python