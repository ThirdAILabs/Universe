#include "Model.h"
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/nn/tensor/Tensor.h>
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
      _activations({}),
      _train_steps(0) {
  for (const auto& loss : _losses) {
    _label_inputs.push_back(loss->labels());
  }

  createOpSchedule();

  checkNoOutputsHaveDependentOps();
  checkOnlyOutputsHaveNoDependentOps();
  checkAllOutputsAreUsedInLosses();
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
  for (auto& op : _op_schedule) {
    op->updateParameters(learning_rate, ++_train_steps);
  }
}

const std::vector<ops::OpPtr>& Model::ops() const { return _op_schedule; }

void Model::forwardImpl(uint32_t input_batch_size, bool use_sparsity) {
  _activations.reallocateForBatch(input_batch_size, use_sparsity);

  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch);
  }
}

void Model::backpropagateImpl(uint32_t label_batch_size) {
  if (label_batch_size != _activations.currentBatchSize()) {
    throw std::invalid_argument(
        "Label batch size does not match input batch size.");
  }

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
  _activations.reallocateForBatch(input_batch_size, /* use_sparsity= */ true);

  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch);
    backpropagateVector(index_in_batch);
  }
}

void Model::forwardVector(uint32_t index_in_batch) {
  for (auto& op : _op_schedule) {
    op->forward(index_in_batch);
  }
}

void Model::backpropagateVector(uint32_t index_in_batch) {
  _activations.resetOutputGradients(index_in_batch);

  for (auto& loss : _losses) {
    loss->computeGradients(index_in_batch);
  }

  for (auto op = _op_schedule.rbegin(); op != _op_schedule.rend(); ++op) {
    (*op)->backpropagate(index_in_batch);
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

void Model::createOpSchedule() {
  std::unordered_map<ops::OpPtr, uint32_t> in_degrees = getInDegrees();

  std::queue<ops::OpPtr> queue;

  for (const auto& input : _inputs) {
    for (const auto& op : input->dependantOps()) {
      in_degrees[op]--;
      if (in_degrees[op] == 0) {
        queue.push(op);
        in_degrees.erase(op);
      }
    }
  }

  std::vector<tensor::ActivationTensorPtr> activations;

  while (!queue.empty()) {
    auto next_op = queue.front();
    queue.pop();
    _op_schedule.push_back(next_op);

    for (const auto& output : next_op->outputs()) {
      activations.push_back(output);
      for (const auto& op : output->dependantOps()) {
        in_degrees[op]--;
        if (in_degrees[op] == 0) {
          queue.push(op);
          in_degrees.erase(op);
        }
      }
    }
  }

  _activations = ActivationsManager(activations);
}

std::unordered_map<ops::OpPtr, uint32_t> Model::getInDegrees() const {
  std::unordered_map<ops::OpPtr, uint32_t> in_degrees;

  std::vector<tensor::TensorPtr> unexplored(_inputs.begin(), _inputs.end());

  while (!unexplored.empty()) {
    std::vector<tensor::TensorPtr> next_unexplored;

    for (const auto& tensor : unexplored) {
      for (const auto& op : tensor->dependantOps()) {
        if (!in_degrees.count(op)) {
          in_degrees[op] = op->inputs().size();

          auto op_outputs = op->outputs();

          next_unexplored.insert(next_unexplored.end(), op_outputs.begin(),
                                 op_outputs.end());
        }
      }
    }

    unexplored = next_unexplored;
  }

  return in_degrees;
}

void Model::checkNoOutputsHaveDependentOps() const {
  for (const auto& output : _outputs) {
    if (!output->dependantOps().empty()) {
      throw std::invalid_argument("Outputs must not be inputs to any ops.");
    }
  }
}

void Model::checkOnlyOutputsHaveNoDependentOps() const {
  std::unordered_set<tensor::ActivationTensorPtr> outputs_set(_outputs.begin(),
                                                              _outputs.end());

  for (const auto& activation : _activations.activationTensors()) {
    if (activation->dependantOps().empty() && !outputs_set.count(activation)) {
      throw std::invalid_argument(
          "All non outputs must be used in at least one op.");
    }
  }
}

void Model::checkAllOutputsAreUsedInLosses() const {
  std::unordered_set<tensor::ActivationTensorPtr> outputs_set(_outputs.begin(),
                                                              _outputs.end());

  for (const auto& loss : _losses) {
    for (const auto& output : loss->outputsUsed()) {
      if (!outputs_set.count(output)) {
        throw std::invalid_argument("Only outputs can be used in losses.");
      }

      outputs_set.erase(output);
    }
  }

  if (!outputs_set.empty()) {
    throw std::invalid_argument("All outputs must be used by a loss.");
  }
}

}  // namespace thirdai::bolt::nn::model