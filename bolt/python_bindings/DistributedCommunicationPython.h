#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedCommInterface.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::train::python {

class DistributedCommPython : DistributedCommInterface {
 public:
  explicit DistributedCommPython(py::object& py_instance);

  void communicate(const bolt::nn::model::ModelPtr& model) override;

  uint64_t min_num_batches(uint64_t num_batches) override;

  std::optional<std::shared_ptr<thirdai::bolt::train::DistributedCommInterface>>
  to_optional();

 private:
  py::object py_instance;
};

}  // namespace thirdai::bolt::train::python