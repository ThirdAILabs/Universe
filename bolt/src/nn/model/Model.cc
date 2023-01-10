#include "Model.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::model {

Model::Model(std::vector<tensor::InputTensorPtr> inputs,
             std::vector<tensor::ActivationTensorPtr> outputs,
             std::vector<loss::LossPtr> losses)
    : _inputs(std::move(inputs)),
      _outputs(std::move(outputs)),
      _losses(std::move(losses)),
      _allocation_manager({}),
      _train_steps(0) {
  for (const auto& loss : _losses) {
    _label_inputs.push_back(loss->labels());
  }

  createComputationSchedule();

  checkNoOutputsHaveDependentOps();
  checkAllOutputsAreUsedInLosses();

  matchOutputFullyConnectedLayersWithLabels();
}

std::shared_ptr<Model> Model::make(
    std::vector<tensor::InputTensorPtr> inputs,
    std::vector<tensor::ActivationTensorPtr> outputs,
    std::vector<loss::LossPtr> losses) {
  return std::make_shared<Model>(std::move(inputs), std::move(outputs),
                                 std::move(losses));
}

void Model::forward(const std::vector<BoltBatch>& inputs, bool use_sparsity) {
  uint32_t input_batch_size = setInputs(inputs);

  forwardImpl(input_batch_size, use_sparsity);
}

void Model::forwardSingleInput(const BoltBatch& inputs, bool use_sparsity) {
  setSingleInput(inputs);

  forwardImpl(inputs.getBatchSize(), use_sparsity);
}

void Model::backpropagate(const std::vector<BoltBatch>& labels) {
  uint32_t label_batch_size = setLabels(labels);

  backpropagateImpl(label_batch_size);
}

void Model::backpropagateSingleInput(const BoltBatch& labels) {
  setSingleLabel(labels);

  backpropagateImpl(labels.getBatchSize());
}

void Model::trainOnBatch(const std::vector<BoltBatch>& inputs,
                         const std::vector<BoltBatch>& labels) {
  uint32_t input_batch_size = setInputs(inputs);
  uint32_t label_batch_size = setLabels(labels);

  trainOnBatchImpl(input_batch_size, label_batch_size);
}

void Model::trainOnBatchSingleInput(const BoltBatch& inputs,
                                    const BoltBatch& labels) {
  setSingleInput(inputs);
  setSingleLabel(labels);

  trainOnBatchImpl(inputs.getBatchSize(), labels.getBatchSize());
}

void Model::updateParameters(float learning_rate) {
  ++_train_steps;
  for (auto& op : _ops) {
    op->updateParameters(learning_rate, _train_steps);
  }
}

std::vector<ops::OpPtr> Model::opComputationOrder() const {
  std::vector<ops::OpPtr> ops;
  for (const auto& tensor : _activation_tensor_computation_order) {
    ops.push_back(tensor->source());
  }
  return ops;
}

const std::vector<tensor::ActivationTensorPtr>& Model::tensorComputationOrder()
    const {
  return _activation_tensor_computation_order;
}

ops::OpPtr Model::getOp(const std::string& name) const {
  for (const auto& op : _ops) {
    if (op->name() == name) {
      return op;
    }
  }
  throw std::invalid_argument("Could not find op with name '" + name + "'.");
}

tensor::ActivationTensorPtr Model::getTensor(const std::string& name) const {
  for (const auto& tensor : _activation_tensor_computation_order) {
    if (tensor->name() == name) {
      return tensor;
    }
  }

  throw std::invalid_argument("Could not find tensor with name '" + name +
                              "'.");
}

tensor::InputTensorPtr Model::getLabelsForOutput(
    const std::string& output_name) {
  for (const auto& loss : _losses) {
    auto outputs_used = loss->outputsUsed();
    if (outputs_used.size() == 1) {
      if (outputs_used.at(0)->name() == output_name) {
        return loss->labels();
      }
    }
  }

  return nullptr;
}

const std::vector<tensor::ActivationTensorPtr>& Model::outputs() const {
  return _outputs;
}

std::string Model::summary(bool print) const {
  std::stringstream summary;

  summary << "===================== Model =====================\n";
  for (uint32_t i = 0; i < _activation_tensor_computation_order.size(); i++) {
    const auto& tensor = _activation_tensor_computation_order[i];
    tensor->source()->summary(summary, tensor->inputs(), tensor.get());
    summary << "\n";
    if (i < _activation_tensor_computation_order.size() - 1) {
      summary << "-------------------------------------------------\n";
    }
  }
  summary << "=================================================\n";

  if (print) {
    std::cout << summary.str() << std::endl;
  }

  return summary.str();
}

uint32_t Model::trainSteps() const { return _train_steps; }

void Model::forwardImpl(uint32_t input_batch_size, bool use_sparsity) {
  _allocation_manager.reallocateForBatch(input_batch_size, use_sparsity);

#pragma omp parallel for default(none) shared(input_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ false);
  }
}

void Model::backpropagateImpl(uint32_t label_batch_size) {
  if (label_batch_size != _allocation_manager.currentBatchSize()) {
    throw std::invalid_argument(
        "Label batch size does not match input batch size.");
  }

#pragma omp parallel for default(none) shared(label_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < label_batch_size;
       index_in_batch++) {
    backpropagateVector(index_in_batch);
  }
}

void Model::trainOnBatchImpl(uint32_t input_batch_size,
                             uint32_t label_batch_size) {
  if (input_batch_size != label_batch_size) {
    throw std::invalid_argument(
        "Input batch size and label batch size do not match.");
  }
  _allocation_manager.reallocateForBatch(input_batch_size,
                                         /* use_sparsity= */ true);

#pragma omp parallel for default(none) shared(input_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ true);
    backpropagateVector(index_in_batch);
  }
}

void Model::forwardVector(uint32_t index_in_batch, bool training) {
  for (auto& tensor : _activation_tensor_computation_order) {
    tensor->forward(index_in_batch, training);
  }
}

void Model::backpropagateVector(uint32_t index_in_batch) {
  _allocation_manager.resetOutputGradients(index_in_batch);

  for (auto& loss : _losses) {
    loss->gradients(index_in_batch, _allocation_manager.currentBatchSize());
  }

  for (auto tensor = _activation_tensor_computation_order.rbegin();
       tensor != _activation_tensor_computation_order.rend(); ++tensor) {
    (*tensor)->backpropagate(index_in_batch);
  }
}

inline uint32_t setBatchHelper(
    std::vector<tensor::InputTensorPtr>& input_tensors,
    const std::vector<BoltBatch>& batches, const std::string& type) {
  if (batches.size() != input_tensors.size()) {
    std::stringstream error;
    error << "Expected " << input_tensors.size() << " " << type
          << " but received " << batches.size() << ".";
    throw std::invalid_argument(error.str());
  }

  std::optional<uint32_t> batch_size = std::nullopt;
  for (uint32_t i = 0; i < input_tensors.size(); i++) {
    if (batch_size && batches[i].getBatchSize() != *batch_size) {
      std::stringstream error;
      error << "Expected all " << type
            << " to have same batch size but received inputs with batch "
               "size "
            << *batch_size << " and " << batches[i].getBatchSize() << ".";
      throw std::invalid_argument(error.str());
    }
    if (!batch_size) {
      batch_size = batches[i].getBatchSize();
    }
    input_tensors[i]->setInputs(batches[i]);
  }

  return batch_size.value();
}

uint32_t Model::setInputs(const std::vector<BoltBatch>& input_batches) {
  return setBatchHelper(_inputs, input_batches, "inputs");
}

void Model::setSingleInput(const BoltBatch& inputs) {
  if (_inputs.size() != 1) {
    throw std::invalid_argument("Expected " + std::to_string(_inputs.size()) +
                                " input batches but received 1.");
  }
  _inputs[0]->setInputs(inputs);
}

uint32_t Model::setLabels(const std::vector<BoltBatch>& label_batches) {
  return setBatchHelper(_label_inputs, label_batches, "labels");
}

void Model::setSingleLabel(const BoltBatch& labels) {
  if (_label_inputs.size() != 1) {
    throw std::invalid_argument("Expected " +
                                std::to_string(_label_inputs.size()) +
                                " label batches but received 1.");
  }
  _label_inputs[0]->setInputs(labels);
}

void Model::createComputationSchedule() {
  std::unordered_map<tensor::ActivationTensorPtr, uint32_t> out_degrees =
      getOutDegrees();

  std::queue<tensor::ActivationTensorPtr> queue;

  for (const auto& output : _outputs) {
    queue.push(output);
  }

  while (!queue.empty()) {
    auto next_tensor = queue.front();
    queue.pop();
    _activation_tensor_computation_order.push_back(next_tensor);

    for (const auto& input : next_tensor->inputs()) {
      auto act_input = tensor::asActivationTensor(input);
      if (!act_input) {
        continue;
      }

      out_degrees.at(act_input)--;
      if (out_degrees.at(act_input) == 0) {
        queue.push(act_input);
        out_degrees.erase(act_input);
      }
    }
  }

  std::reverse(_activation_tensor_computation_order.begin(),
               _activation_tensor_computation_order.end());

  // We need to handle the case where ops are use multiple times in the model.
  std::unordered_set<ops::OpPtr> unique_ops;
  for (auto& tensor : _activation_tensor_computation_order) {
    unique_ops.insert(tensor->source());
  }
  _ops = {unique_ops.begin(), unique_ops.end()};

  _allocation_manager = AllocationManager(_activation_tensor_computation_order);
}

std::unordered_map<tensor::ActivationTensorPtr, uint32_t> Model::getOutDegrees()
    const {
  std::unordered_map<tensor::ActivationTensorPtr, uint32_t> out_degrees;

  std::unordered_set<tensor::ActivationTensorPtr> visited;

  std::function<void(const tensor::ActivationTensorPtr&)> recurse;

  recurse = [&visited, &out_degrees,
             &recurse](const tensor::ActivationTensorPtr& tensor) {
    if (visited.count(tensor)) {
      return;
    }

    visited.insert(tensor);

    for (const auto& input : tensor->inputs()) {
      auto act_input = tensor::asActivationTensor(input);
      // If it's an input tensor its execution order doesn't matter.
      if (act_input) {
        out_degrees[act_input]++;
        recurse(act_input);
      }
    }
  };

  for (const auto& output : _outputs) {
    recurse(output);
  }

  return out_degrees;
}

void Model::checkNoOutputsHaveDependentOps() const {
  auto out_degrees = getOutDegrees();

  for (const auto& output : _outputs) {
    if (out_degrees.count(output)) {
      throw std::invalid_argument(
          "Outputs must not be inputs to any ops. Found output '" +
          output->name() + "' with a dependent op.");
    }
  }
}

void Model::checkAllOutputsAreUsedInLosses() const {
  std::unordered_set<tensor::ActivationTensorPtr> outputs_set(_outputs.begin(),
                                                              _outputs.end());

  for (const auto& loss : _losses) {
    for (const auto& output : loss->outputsUsed()) {
      if (!outputs_set.count(output)) {
        throw std::invalid_argument(
            "Only outputs can be used in losses and outputs cannot be reused "
            "in multiple losses. Found tensor '" +
            output->name() +
            "' which is either not an output or or has already been used in a "
            "loss function.");
      }

      outputs_set.erase(output);
    }
  }

  if (!outputs_set.empty()) {
    throw std::invalid_argument(
        "All outputs must be used by a loss. Found an output '" +
        (*outputs_set.begin())->name() +
        "' which is not used by any loss function.");
  }
}

void Model::matchOutputFullyConnectedLayersWithLabels() {
  for (const auto& loss : _losses) {
    auto outputs_used = loss->outputsUsed();
    if (outputs_used.size() == 1) {
      auto fully_connected = std::dynamic_pointer_cast<ops::FullyConnected>(
          outputs_used.at(0)->source());

      if (fully_connected) {
        outputs_used.at(0)->addInput(loss->labels());
      }
    }
  }
}

}  // namespace thirdai::bolt::nn::model