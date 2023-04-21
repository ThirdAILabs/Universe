#include "Model.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/autograd/ComputationGraph.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Versions.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <licensing/src/CheckLicense.h>
#include <utils/UUID.h>
#include <utils/Version.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <memory>
#include <numeric>
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
      _train_steps(0),
      _model_uuid(
          utils::uuid::getRandomHexString(/* num_bytes_randomness= */ 16)) {
  licensing::checkLicense();

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

  verifyAllowedOutputDim();
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

  _allocation_manager.reallocateIfNeeded(input_batch_size, use_sparsity);

#pragma omp parallel for default(none) \
    shared(input_batch_size) if (input_batch_size > 1)
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

void Model::trainOnBatch(const tensor::TensorList& inputs,
                         const tensor::TensorList& labels) {
  uint32_t input_batch_size = setInput(inputs);
  uint32_t label_batch_size = setLabels(labels);

  _total_training_samples += input_batch_size;
  licensing::entitlements().verifyAllowedNumberOfTrainingSamples(
      _total_training_samples);

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

tensor::TensorList Model::forward(const tensor::TensorList& inputs,
                                  const tensor::TensorList& labels,
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

const autograd::ComputationList& Model::labels() const { return _labels; }

const std::vector<loss::LossPtr>& Model::losses() const { return _losses; }

const std::vector<ops::OpPtr>& Model::ops() const { return _ops; }

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

  summary << "\n===================== Model =====================\n";
  for (const auto& input : _inputs) {
    input->summary(summary);
    summary << "\n";
  }
  for (const auto& comp : _computation_order) {
    comp->summary(summary);
    summary << "\n";
  }
  summary << "=================================================\n";

  if (print) {
    std::cout << summary.str() << std::endl;
  }

  return summary.str();
}

uint32_t Model::trainSteps() const { return _train_steps; }

std::vector<uint32_t> Model::inputDims() const {
  std::vector<uint32_t> dims;
  for (const auto& input : _inputs) {
    dims.push_back(input->dim());
  }
  return dims;
}

std::vector<uint32_t> Model::labelDims() const {
  std::vector<uint32_t> dims;
  for (const auto& label : _labels) {
    dims.push_back(label->dim());
  }
  return dims;
}

std::vector<std::vector<float>*> Model::gradients() const {
  std::vector<std::vector<float>*> grads;

  for (const auto& op : _ops) {
    auto op_grads = op->gradients();
    grads.insert(grads.end(), op_grads.begin(), op_grads.end());
  }

  return grads;
}

void Model::freezeHashTables(bool insert_labels_if_not_found) {
  for (auto& op : _ops) {
    if (auto fc = std::dynamic_pointer_cast<ops::FullyConnected>(op)) {
      // insert_labels_if_not_found will have no effect on non output layers
      // because they will not have access to labels.
      fc->freezeHashTables(insert_labels_if_not_found);
    }
  }
}

std::vector<std::pair<autograd::ComputationPtr, autograd::ComputationPtr>>
Model::outputLabelPairs() const {
  std::vector<std::pair<autograd::ComputationPtr, autograd::ComputationPtr>>
      output_label_pairs;

  for (const auto& loss : _losses) {
    auto outputs_used = loss->outputsUsed();
    auto loss_labels = loss->labels();
    // A label and output match if they are both used in a loss function with no
    // other labels or outputs, hence we can iterate over the loss functions and
    // see which act on a single output and label.
    if (outputs_used.size() == 1 && loss_labels.size() == 1) {
      output_label_pairs.emplace_back(outputs_used.at(0), loss_labels.at(0));
    }
  }
  return output_label_pairs;
}

void Model::save(const std::string& filename, bool save_metadata) {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);

  setSerializeOptimizer(false);

  save_stream(output_stream);

  if (save_metadata) {
    saveMetadata(filename);
  }
}

void Model::checkpoint(const std::string& filename, bool save_metadata) {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);

  setSerializeOptimizer(true);

  save_stream(output_stream);

  if (save_metadata) {
    saveMetadata(filename);
  }
}

void Model::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

void Model::setSerializeOptimizer(bool should_save_optimizer) {
  for (auto& op : _ops) {
    op->setSerializeOptimizer(should_save_optimizer);
  }
}

std::shared_ptr<Model> Model::load(const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<Model> Model::load_stream(std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<Model> deserialize_into(new Model());
  iarchive(*deserialize_into);

  return deserialize_into;
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
    error << "When preparing the model for the next batch, expected "
          << inputs.size() << " " << type << " but received " << batches.size()
          << ".";
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
  return setBatchHelper(_inputs, input_batches, "input batches");
}

uint32_t Model::setLabels(const tensor::TensorList& label_batches) {
  return setBatchHelper(_labels, label_batches, "label batches");
}

void Model::matchOutputFullyConnectedLayersWithLabels() const {
  for (const auto& [output, label] : outputLabelPairs()) {
    auto fully_connected = ops::FullyConnected::cast(output->op());

    if (fully_connected) {
      output->addInput(label);
    }
  }
}

void Model::saveMetadata(const std::string& save_path) const {
  auto file = dataset::SafeFileIO::ofstream(save_path + ".metadata");

  file << "thirdai_version=" << version() << std::endl;

  file << "model_uuid=" << _model_uuid << std::endl;

  auto time = std::chrono::system_clock::now();
  auto c_time = std::chrono::system_clock::to_time_t(time);
  file << "date_saved=" << std::ctime(&c_time);

  file << "train_steps_before_save=" << trainSteps() << std::endl;

#if THIRDAI_EXPOSE_ALL
  file << "model_summary=";
  file << summary(/* print= */ false);
#endif
}

void Model::verifyAllowedOutputDim() const {
  uint64_t total_output_dim = std::transform_reduce(
      _outputs.begin(), _outputs.end(), 0UL, std::plus(),
      [](const auto& output) { return output->op()->dim(); });

  licensing::entitlements().verifyAllowedOutputDim(total_output_dim);
}

template void Model::serialize(cereal::BinaryInputArchive&, uint32_t version);
template void Model::serialize(cereal::BinaryOutputArchive&, uint32_t version);

template <class Archive>
void Model::serialize(Archive& archive, const uint32_t version) {
  licensing::entitlements().verifySaveLoad();

  std::string class_name = "BOLT_MODEL";
  versions::checkVersion(version, versions::BOLT_MODEL_VERSION, class_name);

  archive(_inputs, _outputs, _labels, _losses, _ops, _computation_order,
          _allocation_manager, _train_steps, _model_uuid,
          _total_training_samples);

  verifyAllowedOutputDim();
}

}  // namespace thirdai::bolt::nn::model

CEREAL_CLASS_VERSION(thirdai::bolt::nn::model::Model,
                     thirdai::bolt::nn::model::versions::BOLT_MODEL_VERSION)