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
    auto labels = loss->labels();
    _labels.insert(_labels.end(), labels.begin(), labels.end());
  }

  _computation_order =
      autograd::getComputationOrder(_inputs, _outputs, _losses);

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

tensor::TensorList Model::forward(const tensor::TensorList& inputs,
                                  bool use_sparsity) {
  uint32_t input_batch_size = setInput(inputs);

  return forward(input_batch_size, use_sparsity);
}

tensor::TensorList Model::forward(const tensor::TensorPtr& inputs,
                                  bool use_sparsity) {
  setInput(inputs);

  return forward(inputs->batchSize(), use_sparsity);
}

void Model::trainOnBatch(const tensor::TensorList& inputs,
                         const tensor::TensorList& labels) {
  uint32_t input_batch_size = setInput(inputs);
  uint32_t label_batch_size = setLabels(labels);

  trainOnBatch(input_batch_size, label_batch_size);
}

void Model::trainOnBatch(const tensor::TensorPtr& inputs,
                         const tensor::TensorPtr& labels) {
  setInput(inputs);
  setLabels(labels);

  trainOnBatch(inputs->batchSize(), labels->batchSize());
}

tensor::TensorList Model::forward(const tensor::TensorList& inputs,
                                  const tensor::TensorList& labels,
                                  bool use_sparsity) {
  setLabels(labels);
  return forward(inputs, use_sparsity);
}

tensor::TensorList Model::forward(const tensor::TensorPtr& inputs,
                                  const tensor::TensorPtr& labels,
                                  bool use_sparsity) {
  setLabels(labels);
  return forward(inputs, use_sparsity);
}

void Model::updateParameters(float learning_rate) {
  ++_train_steps;
  for (auto& op : _ops) {
    op->updateParameters(learning_rate, _train_steps);
  }
}

std::vector<ops::OpPtr> Model::opExecutionOrder() const {
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

tensor::TensorList Model::forward(uint32_t input_batch_size,
                                  bool use_sparsity) {
  _allocation_manager.reallocateIfNeeded(input_batch_size, use_sparsity);

#pragma omp parallel for default(none) shared(input_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ false);
  }

  tensor::TensorList outputs;
  for (auto& output : _outputs) {
    outputs.push_back(output->tensor());
  }
  return outputs;
}

void Model::trainOnBatch(uint32_t input_batch_size, uint32_t label_batch_size) {
  if (input_batch_size != label_batch_size) {
    throw std::invalid_argument(
        "Input batch size and label batch size do not match.");
  }
  _allocation_manager.reallocateIfNeeded(input_batch_size,
                                         /* use_sparsity= */ true);

#pragma omp parallel for default(none) shared(input_batch_size)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ true);
    backpropagateVector(index_in_batch, input_batch_size);
  }
}

void Model::forwardVector(uint32_t index_in_batch, bool training) {
  for (auto& tensor : _computation_order) {
    tensor->forward(index_in_batch, training);
  }
}

void Model::backpropagateVector(uint32_t index_in_batch, uint32_t batch_size) {
  _allocation_manager.resetOutputGradients(index_in_batch);

  for (auto& loss : _losses) {
    loss->gradients(index_in_batch, batch_size);
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
    if (!batch_size) {
      batch_size = batches[i]->batchSize();
    }
    if (batches[i]->batchSize() != *batch_size) {
      std::stringstream error;
      error << "Expected all " << type
            << " to have same batch size but received inputs with batch "
               "size "
            << *batch_size << " and " << batches[i]->batchSize() << ".";
      throw std::invalid_argument(error.str());
    }
    inputs[i]->setTensor(batches[i]);
  }

  return batch_size.value();
}

uint32_t Model::setInput(const tensor::TensorList& input_batches) {
  return setBatchHelper(_inputs, input_batches, "inputs");
}

void Model::setInput(const tensor::TensorPtr& input) {
  if (_inputs.size() != 1) {
    throw std::invalid_argument("Expected " + std::to_string(_inputs.size()) +
                                " input batches but received 1.");
  }
  _inputs[0]->setTensor(input);
}

uint32_t Model::setLabels(const tensor::TensorList& label_batches) {
  return setBatchHelper(_labels, label_batches, "labels");
}

void Model::setLabels(const tensor::TensorPtr& labels) {
  if (_labels.size() != 1) {
    throw std::invalid_argument("Expected " + std::to_string(_labels.size()) +
                                " label batches but received 1.");
  }
  _labels[0]->setTensor(labels);
}

void Model::matchOutputFullyConnectedLayersWithLabels() {
  for (const auto& loss : _losses) {
    auto outputs_used = loss->outputsUsed();
    auto loss_labels = loss->labels();
    if (outputs_used.size() == 1 && loss_labels.size() == 1) {
      auto fully_connected = std::dynamic_pointer_cast<ops::FullyConnected>(
          outputs_used.at(0)->op());

      if (fully_connected) {
        outputs_used.at(0)->addInput(loss_labels.at(0));
      }
    }
  }
}

}  // namespace thirdai::bolt::nn::model