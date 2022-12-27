#pragma once

#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>

namespace thirdai::bolt::nn::graph {

class ComputationGraph {
 public:
  ComputationGraph(std::vector<tensor::InputTensorPtr> inputs,
                   std::vector<tensor::ActivationTensorPtr> outputs);

 private:
  std::unordered_map<ops::OpPtr, uint32_t> getInDegrees() const;

  void createOpSchedule();

  std::vector<tensor::InputTensorPtr> _inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;

  std::vector<ops::OpPtr> _op_schedule;
};

}  // namespace thirdai::bolt::nn::graph