#include "Model.h"
#include <bolt/src/nn/autograd/ComputationGraph.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::model {

Model::Model(autograd::ComputationList inputs,
             autograd::ComputationList outputs,
             std::vector<loss::LossPtr> losses)
    : _inputs(std::move(inputs)),
      _outputs(std::move(outputs)),
      _losses(std::move(losses)),
      _allocation_manager({}),
      _train_steps(0) {
  for (const auto& loss : _losses) {
    _labels.push_back(loss->labels());
  }

  checkNoOutputsHaveDependentOps();
  checkAllOutputsAreUsedInLosses();

  _computation_order = autograd::getComputationOrder(_inputs, _outputs);
  _allocation_manager = AllocationManager(_computation_order);

  std::unordered_set<ops::OpPtr> ops;
  for (const auto& comp : _computation_order) {
    ops.insert(comp->op());
  }
  _ops.assign(ops.begin(), ops.end());

  matchOutputFullyConnectedLayersWithLabels();
}

std::shared_ptr<Model> Model::make(autograd::ComputationList inputs,
                                   autograd::ComputationList outputs,
                                   std::vector<loss::LossPtr> losses) {
  return std::make_shared<Model>(std::move(inputs), std::move(outputs),
                                 std::move(losses));
}

void Model::forward(const tensor::TensorList& inputs, bool use_sparsity) {
  uint32_t input_batch_size = setInputs(inputs);

  forward(input_batch_size, use_sparsity);
}

void Model::forward(const tensor::TensorPtr& inputs, bool use_sparsity) {
  setSingleInput(inputs);

  forward(inputs->batchSize(), use_sparsity);
}

void Model::trainOnBatch(const tensor::TensorList& inputs,
                         const tensor::TensorList& labels) {
  uint32_t input_batch_size = setInputs(inputs);
  uint32_t label_batch_size = setLabels(labels);

  trainOnBatch(input_batch_size, label_batch_size);
}

void Model::trainOnBatch(const tensor::TensorPtr& inputs,
                         const tensor::TensorPtr& labels) {
  setSingleInput(inputs);
  setSingleLabel(labels);

  trainOnBatch(inputs->batchSize(), labels->batchSize());
}

void Model::updateParameters(float learning_rate) {
  ++_train_steps;
  for (auto& op : _ops) {
    op->updateParameters(learning_rate, _train_steps);
  }
}

std::vector<ops::OpPtr> Model::opComputationOrder() const {
  std::vector<ops::OpPtr> ops;
  for (const auto& comp : _computation_order) {
    ops.push_back(comp->op());
  }
  return ops;
}

autograd::ComputationList Model::computationOrder() const {
  autograd::ComputationList all_comps;
  all_comps.insert(all_comps.end(), _inputs.begin(), _inputs.end());
  all_comps.insert(all_comps.end(), _computation_order.begin(),
                   _computation_order.end());
  return all_comps;
}

const autograd::ComputationList& Model::outputs() const { return _outputs; }

ops::OpPtr Model::getOp(const std::string& name) const {
  for (const auto& op : _ops) {
    if (op->name() == name) {
      return op;
    }
  }
  throw std::invalid_argument("Could not find op with name '" + name + "'.");
}

autograd::ComputationPtr Model::getComputation(const std::string& name) const {
  for (const auto& comp : _computation_order) {
    if (comp->name() == name) {
      return comp;
    }
  }

  throw std::invalid_argument("Could not find computation with name '" + name +
                              "'.");
}

autograd::ComputationPtr Model::getLabelsForOutput(
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

std::string Model::summary(bool print) const {
  std::stringstream summary;

  summary << "===================== Model =====================\n";
  for (uint32_t i = 0; i < _computation_order.size(); i++) {
    _computation_order[i]->summary(summary);
    summary << "\n";
    if (i < _computation_order.size() - 1) {
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

void Model::forward(uint32_t input_batch_size, bool use_sparsity) {
  _allocation_manager.reallocateForBatch(input_batch_size, use_sparsity);

#pragma omp parallel for default(none) shared(input_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ false);
  }
}

void Model::trainOnBatch(uint32_t input_batch_size, uint32_t label_batch_size) {
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
  for (auto& tensor : _computation_order) {
    tensor->forward(index_in_batch, training);
  }
}

void Model::backpropagateVector(uint32_t index_in_batch) {
  _allocation_manager.resetOutputGradients(index_in_batch);

  for (auto& loss : _losses) {
    loss->gradients(index_in_batch, _allocation_manager.currentBatchSize());
  }

  for (auto tensor = _computation_order.rbegin();
       tensor != _computation_order.rend(); ++tensor) {
    (*tensor)->backpropagate(index_in_batch);
  }
}

inline uint32_t setBatchHelper(autograd::ComputationList& inputs,
                               const tensor::TensorList& batches,
                               const std::string& type) {
  if (batches.size() != inputs.size()) {
    std::stringstream error;
    error << "Expected " << inputs.size() << " " << type << " but received "
          << batches.size() << ".";
    throw std::invalid_argument(error.str());
  }

  std::optional<uint32_t> batch_size = std::nullopt;
  for (uint32_t i = 0; i < inputs.size(); i++) {
    if (batch_size && batches[i]->batchSize() != *batch_size) {
      std::stringstream error;
      error << "Expected all " << type
            << " to have same batch size but received inputs with batch "
               "size "
            << *batch_size << " and " << batches[i]->batchSize() << ".";
      throw std::invalid_argument(error.str());
    }
    if (!batch_size) {
      batch_size = batches[i]->batchSize();
    }
    inputs[i]->setTensor(batches[i]);
  }

  return batch_size.value();
}

uint32_t Model::setInputs(const tensor::TensorList& input_batches) {
  return setBatchHelper(_inputs, input_batches, "inputs");
}

void Model::setSingleInput(const tensor::TensorPtr& input) {
  if (_inputs.size() != 1) {
    throw std::invalid_argument("Expected " + std::to_string(_inputs.size()) +
                                " input batches but received 1.");
  }
  _inputs[0]->setTensor(input);
}

uint32_t Model::setLabels(const tensor::TensorList& label_batches) {
  return setBatchHelper(_labels, label_batches, "labels");
}

void Model::setSingleLabel(const tensor::TensorPtr& labels) {
  if (_labels.size() != 1) {
    throw std::invalid_argument("Expected " + std::to_string(_labels.size()) +
                                " label batches but received 1.");
  }
  _labels[0]->setTensor(labels);
}

void Model::checkNoOutputsHaveDependentOps() const {
  auto out_degrees = autograd::countDependentComputations(_outputs);

  for (const auto& output : _outputs) {
    if (out_degrees.count(output)) {
      throw std::invalid_argument(
          "Outputs must not be inputs to any ops. Found output '" +
          output->name() + "' with a dependent op.");
    }
  }
}

void Model::checkAllOutputsAreUsedInLosses() const {
  std::unordered_set<autograd::ComputationPtr> outputs_set(_outputs.begin(),
                                                           _outputs.end());

  for (const auto& loss : _losses) {
    for (const auto& output : loss->outputsUsed()) {
      if (!outputs_set.count(output)) {
        throw std::invalid_argument(
            "Only outputs can be used in losses and outputs cannot be reused "
            "in multiple losses. Found output '" +
            output->name() +
            "' which is either not an output or has already been used in a "
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
          outputs_used.at(0)->op());

      if (fully_connected) {
        outputs_used.at(0)->addInput(loss->labels());
      }
    }
  }
}

}  // namespace thirdai::bolt::nn::model