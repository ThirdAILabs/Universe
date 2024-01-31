#include "Model.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/autograd/ComputationGraph.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/Switch.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <licensing/src/CheckLicense.h>
#include <utils/UUID.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt {

Model::Model(ComputationList inputs, ComputationList outputs,
             std::vector<LossPtr> losses, ComputationList additional_labels)
    : _inputs(std::move(inputs)),
      _outputs(std::move(outputs)),
      _losses(std::move(losses)),
      _epochs(0),
      _train_steps(0),
      _model_uuid(
          utils::uuid::getRandomHexString(/* num_bytes_randomness= */ 16)),
      _thirdai_version(thirdai::version()) {
  licensing::checkLicense();

  for (const auto& loss : _losses) {
    auto labels = loss->labels();
    _labels.insert(_labels.end(), labels.begin(), labels.end());
  }
  _labels.insert(_labels.end(), additional_labels.begin(),
                 additional_labels.end());

  _computation_order = getComputationOrder(_inputs, _outputs, _losses);

  nameComputations(_inputs, _computation_order, _labels);

  _allocation_manager = AllocationManager(_computation_order);

  std::unordered_set<std::string> op_names;
  std::unordered_set<OpPtr> ops;
  for (const auto& comp : _computation_order) {
    std::string name = comp->op()->name();

    // Check if we have found a new op with the same name.
    if (op_names.count(name) && !ops.count(comp->op())) {
      throw std::invalid_argument(
          "Found multiple Ops in model with the name '" + name +
          "'. All ops in a model must have unique names. The name of the op "
          "can be updated with `op.name = 'new_name'`.");
    }
    ops.insert(comp->op());
    op_names.insert(comp->op()->name());
  }
  _ops.assign(ops.begin(), ops.end());

  matchOutputFullyConnectedLayersWithLabels();

  verifyAllowedOutputDim();
}

std::shared_ptr<Model> Model::make(ComputationList inputs,
                                   ComputationList outputs,
                                   std::vector<LossPtr> losses,
                                   ComputationList additional_labels) {
  auto model = std::shared_ptr<Model>(
      new Model(std::move(inputs), std::move(outputs), std::move(losses),
                std::move(additional_labels)));

  // This has to be done here because we need the model to be allocated using a
  // shared_ptr in order to use shared_from_this() to get a valid reference.
  model->registerWithOps();
  return model;
}

TensorList Model::forward(const TensorList& inputs, bool use_sparsity) {
  uint32_t input_batch_size = setInput(inputs);

  _allocation_manager.reallocateIfNeeded(input_batch_size, use_sparsity);

#pragma omp parallel for default(none) \
    shared(input_batch_size) if (input_batch_size > 1)
  for (uint32_t index_in_batch = 0; index_in_batch < input_batch_size;
       index_in_batch++) {
    forwardVector(index_in_batch, /* training= */ false);
  }

  TensorList outputs;
  for (auto& output : _outputs) {
    outputs.push_back(output->tensor());
  }
  return outputs;
}

void Model::trainOnBatch(const TensorList& inputs, const TensorList& labels) {
  requireOptimizer();

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

void Model::backpropagate(const TensorList& labels) {
  requireOptimizer();

  uint32_t batch_size = setLabels(labels);

  _total_training_samples += batch_size;
  licensing::entitlements().verifyAllowedNumberOfTrainingSamples(
      _total_training_samples);

  if (!_inputs.empty() && _inputs.at(0)->tensor()->batchSize() != batch_size) {
    throw std::invalid_argument(
        "Labels provided to backpropagate do not match batch size of last "
        "batch.");
  }

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(batch_size, error)
  for (uint32_t index_in_batch = 0; index_in_batch < batch_size;
       index_in_batch++) {
    try {
      backpropagateVector(index_in_batch, batch_size);
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }
}

TensorList Model::forward(const TensorList& inputs, const TensorList& labels,
                          bool use_sparsity) {
  setLabels(labels);
  return forward(inputs, use_sparsity);
}

void Model::updateParameters(float learning_rate) {
  requireOptimizer();

  ++_train_steps;
  for (auto& op : _ops) {
    op->updateTrainableParameters(learning_rate, _train_steps);
  }
}

void Model::forceStateReallocation() {
  _allocation_manager.forceReallocation();
}

std::vector<OpPtr> Model::opExecutionOrder() const {
  std::vector<OpPtr> ops;
  for (const auto& comp : _computation_order) {
    ops.push_back(comp->op());
  }
  return ops;
}

ComputationList Model::computationOrder() const {
  ComputationList all_comps;
  all_comps.insert(all_comps.end(), _inputs.begin(), _inputs.end());
  all_comps.insert(all_comps.end(), _computation_order.begin(),
                   _computation_order.end());
  return all_comps;
}

ComputationList Model::computationOrderWithoutInputs() const {
  return _computation_order;
}

const ComputationList& Model::inputs() const { return _inputs; }

const ComputationList& Model::outputs() const { return _outputs; }

const ComputationList& Model::labels() const { return _labels; }

const std::vector<LossPtr>& Model::losses() const { return _losses; }

const std::vector<OpPtr>& Model::ops() const { return _ops; }

OpPtr Model::getOp(const std::string& name) const {
  for (const auto& op : _ops) {
    if (op->name() == name) {
      return op;
    }
  }
  throw std::invalid_argument("Could not find op with name '" + name + "'.");
}

ComputationPtr Model::getComputation(const std::string& name) const {
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
  summary << "Total Paramters: " << numParams() << "\n";
  summary << "thirdai Version: " << thirdaiVersion() << "\n";
  summary << "=================================================\n";

  if (print) {
    std::cout << summary.str() << std::endl;
  }

  return summary.str();
}

std::string Model::thirdaiVersion() const { return _thirdai_version; }

size_t Model::numParams() const {
  size_t total_params = 0;
  for (const auto* param : parameters()) {
    total_params += param->size();
  }
  return total_params;
}

uint32_t Model::trainSteps() const { return _train_steps; }

uint32_t Model::epochs() const { return _epochs; }

void Model::incrementEpochs() { _epochs++; }

void Model::overrideTrainSteps(uint32_t train_steps) {
  _train_steps = train_steps;
}

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

std::vector<std::vector<float>*> Model::parameters() const {
  std::vector<std::vector<float>*> params;

  for (const auto& op : _ops) {
    auto op_params = op->parameters();
    params.insert(params.end(), op_params.begin(), op_params.end());
  }

  return params;
}

uint64_t sumFlattenedDims(const std::vector<std::vector<float>*>& values) {
  uint64_t total_dim = 0;
  for (const auto* value : values) {
    total_dim += value->size();
  }
  return total_dim;
}

std::pair<const float*, uint64_t> concatenateValues(
    const std::vector<std::vector<float>*>& values) {
  uint64_t total_dim = sumFlattenedDims(values);

  float* combined_values = new float[total_dim];
  uint64_t offset = 0;
  for (const auto* value : values) {
    std::copy(value->data(), value->data() + value->size(),
              combined_values + offset);
    offset += value->size();
  }

  return {combined_values, total_dim};
}

void setValues(const std::vector<std::vector<float>*>& values,
               const float* concatenated_values, uint64_t flattened_dim) {
  uint64_t total_dim = sumFlattenedDims(values);

  if (total_dim != flattened_dim) {
    std::stringstream error;
    error << "Expected " << total_dim
          << " parameters in setValues, but received " << flattened_dim
          << " parameters.";
    throw std::invalid_argument(error.str());
  }

  uint64_t offset = 0;
  for (auto* value : values) {
    std::copy(concatenated_values + offset,
              concatenated_values + offset + value->size(), value->data());
    offset += value->size();
  }
}

std::pair<const float*, uint64_t> Model::getFlattenedGradients() {
  requireOptimizer();

  return concatenateValues(gradients());
}

std::pair<const float*, uint64_t> Model::getFlattenedParameters() const {
  return concatenateValues(parameters());
}

void Model::setFlattenedGradients(const float* concatenated_values,
                                  uint64_t flattened_dim) {
  requireOptimizer();

  setValues(gradients(), concatenated_values, flattened_dim);
}

void Model::setFlattenedParameters(const float* concatenated_values,
                                   uint64_t flattened_dim) const {
  setValues(parameters(), concatenated_values, flattened_dim);
  /*
   * Here, we are re-building the hash tables again, as the older weights
   * seems to be redundant, when we all-reduce the weights while using
   * distributed.
   */
  for (const auto& op : _ops) {
    if (auto fc = std::dynamic_pointer_cast<FullyConnected>(op)) {
      fc->reBuildHashFunction();
    }
  }
}

void Model::disableSparseParameterUpdates() {
  for (const auto& op : _ops) {
    op->disableSparseParameterUpdates();
  }
}

void Model::enableSparseParameterUpdates() {
  for (const auto& op : _ops) {
    op->enableSparseParameterUpdates();
  }
}

void Model::switchToSgd() {
  for (const auto& op : _ops) {
    op->switchToSgd();
  }
}

void Model::freezeHashTables(bool insert_labels_if_not_found) {
  for (auto& op : _ops) {
    if (auto fc = std::dynamic_pointer_cast<FullyConnected>(op)) {
      // insert_labels_if_not_found will have no effect on non output layers
      // because they will not have access to labels.
      fc->freezeHashTables(insert_labels_if_not_found);
    }
  }
}

void Model::unfreezeHashTables() {
  for (auto& op : _ops) {
    if (auto fc = FullyConnected::cast(op)) {
      // insert_labels_if_not_found will have no effect on non output layers
      // because they will not have access to labels.
      fc->unfreezeHashTables();
    }
  }
}

std::vector<std::pair<ComputationPtr, ComputationPtr>> Model::outputLabelPairs()
    const {
  std::vector<std::pair<ComputationPtr, ComputationPtr>> output_label_pairs;

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

std::unordered_map<std::string, double> Model::getNorms() const {
  std::unordered_map<std::string, double> norms;

  for (const auto& op : _ops) {
    auto op_norms = op->parameterAndGradNorms();
    for (const auto& [name, norm] : op_norms) {
      norms[op->name() + "_" + name] = norm;
    }
  }

  return norms;
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

void Model::requireOptimizer() {
  if (!_optimizer_initialized) {
    for (auto& op : _ops) {
      op->initOptimizer();
    }
    _optimizer_initialized = true;
  }
}

inline uint32_t setBatchHelper(ComputationList& inputs,
                               const TensorList& batches,
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

uint32_t Model::setInput(const TensorList& input_batches) {
  return setBatchHelper(_inputs, input_batches, "input batches");
}

uint32_t Model::setLabels(const TensorList& label_batches) {
  return setBatchHelper(_labels, label_batches, "label batches");
}

void Model::matchOutputFullyConnectedLayersWithLabels() const {
  for (const auto& [output, label] : outputLabelPairs()) {
    auto fully_connected = FullyConnected::cast(output->op());
    auto switch_op = Switch::cast(output->op());

    if (fully_connected || switch_op) {
      output->addInput(label);
    }
  }
}

void Model::registerWithOps() {
  for (auto& op : _ops) {
    op->registerModel(weak_from_this());
  }
}

void Model::nameComputations(ComputationList& inputs, ComputationList& comps,
                             ComputationList& labels) {
  uint32_t comp_count = 0;
  auto next_name = [&comp_count]() {
    return "tensor_" + std::to_string(++comp_count);
  };
  std::unordered_set<ComputationPtr> visited;
  for (auto& input : inputs) {
    // The same computation might be referenced multiple times in the inputs.
    if (!visited.count(input)) {
      input->setName(next_name());
      visited.insert(input);
    }
  }
  for (auto& comp : comps) {
    if (visited.count(comp)) {
      throw std::invalid_argument(
          "A computation must not be used multiple times in the computation "
          "graph.");
    }
    comp->setName(next_name());
    visited.insert(comp);
  }
  // The same computation might be referenced multiple times in the labels.
  for (auto& label : labels) {
    if (!visited.count(label)) {
      label->setName(next_name());
      visited.insert(label);
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

template void Model::serialize(cereal::BinaryInputArchive&,
                               const uint32_t version);
template void Model::serialize(cereal::BinaryOutputArchive&,
                               const uint32_t version);

template <class Archive>
void Model::serialize(Archive& archive, const uint32_t version) {
  licensing::entitlements().verifySaveLoad();

  _thirdai_version = thirdai::version();
  archive(_thirdai_version);

  std::string class_name = "BOLT_MODEL";
  versions::checkVersion(version, versions::BOLT_MODEL_VERSION,
                         _thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::BOLT_MODEL_VERSION after serialization changes
  archive(_inputs, _outputs, _labels, _losses, _ops, _computation_order,
          _allocation_manager, _train_steps, _model_uuid,
          _total_training_samples);

  verifyAllowedOutputDim();

  registerWithOps();
}

}  // namespace thirdai::bolt

CEREAL_CLASS_VERSION(thirdai::bolt::Model,
                     thirdai::versions::BOLT_MODEL_VERSION)