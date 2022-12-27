#pragma once

#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <vector>

namespace thirdai::bolt::nn::graph {

class ComputationGraph {
 public:
  ComputationGraph(std::vector<tensor::InputTensorPtr> inputs,
                   std::vector<tensor::ActivationTensorPtr> outputs,
                   std::vector<loss::LossPtr> losses);

 private:
  void forward(uint32_t index_in_batch);

  void backpropagate(uint32_t index_in_batch,
                     const std::vector<BoltBatch>& labels);

  void createOpSchedule();

  std::unordered_map<ops::OpPtr, uint32_t> getInDegrees() const;

  void checkNoOutputsHaveDependentOps() const;

  void checkOnlyOutputsHaveNoDependentOps() const;

  void checkAllOutputsAreUsedInLosses() const;

  std::vector<tensor::InputTensorPtr> _inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;
  std::vector<loss::LossPtr> _losses;

  std::vector<tensor::ActivationTensorPtr> _activations;
  std::vector<ops::OpPtr> _op_schedule;
};

}  // namespace thirdai::bolt::nn::graph