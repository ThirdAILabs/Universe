#pragma once

#include "ActivationsManager.h"
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <vector>

namespace thirdai::bolt::nn::computation_graph {

class ComputationGraph {
 public:
  ComputationGraph(std::vector<tensor::InputTensorPtr> inputs,
                   std::vector<tensor::ActivationTensorPtr> outputs,
                   std::vector<loss::LossPtr> losses);

  void forward(const std::vector<BoltBatch>& inputs, bool use_sparsity);

  void backpropagate(const std::vector<BoltBatch>& labels);

  void trainOnBatch(const std::vector<BoltBatch>& inputs,
                    const std::vector<BoltBatch>& labels);

 private:
  void forward(uint32_t index_in_batch);

  void backpropagate(uint32_t index_in_batch);

  uint32_t setInputs(const std::vector<BoltBatch>& inputs);

  uint32_t setLabels(const std::vector<BoltBatch>& labels);

  void createOpSchedule();

  std::unordered_map<ops::OpPtr, uint32_t> getInDegrees() const;

  void checkNoOutputsHaveDependentOps() const;

  void checkOnlyOutputsHaveNoDependentOps() const;

  void checkAllOutputsAreUsedInLosses() const;

  std::vector<tensor::InputTensorPtr> _inputs;
  std::vector<tensor::InputTensorPtr> _label_inputs;
  std::vector<tensor::ActivationTensorPtr> _outputs;

  std::vector<loss::LossPtr> _losses;

  std::vector<ops::OpPtr> _op_schedule;
  ActivationsManager _activations;
};

}  // namespace thirdai::bolt::nn::computation_graph