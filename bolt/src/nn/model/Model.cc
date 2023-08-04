#include "Model.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <bolt/src/nn/autograd/ComputationGraph.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/loss/protobuf_utils/FromProto.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/ops/protobuf_utils/FromProto.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <licensing/src/CheckLicense.h>
#include <utils/UUID.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt::nn::model {

Model::Model(autograd::ComputationList inputs,
             autograd::ComputationList outputs,
             std::vector<loss::LossPtr> losses,
             autograd::ComputationList additional_labels)
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
  _labels.insert(_labels.end(), additional_labels.begin(),
                 additional_labels.end());

  _computation_order =
      autograd::getComputationOrder(_inputs, _outputs, _losses);

  nameComputations(_inputs, _computation_order, _labels);

  _allocation_manager = AllocationManager(_computation_order);

  std::unordered_set<std::string> op_names;
  std::unordered_set<ops::OpPtr> ops;
  for (const auto& comp : _computation_order) {
    ops.insert(comp->op());
    std::string name = comp->op()->name();

    // Check if we have found a new op with the same name.
    if (op_names.count(name) && !ops.count(comp->op())) {
      throw std::invalid_argument(
          "Found multiple Ops in model with the name '" + name +
          "'. All ops in a model must have unique names. The name of the op "
          "can be updated with `op.name = 'op_name'`.");
    }
    op_names.insert(comp->op()->name());
  }
  _ops.assign(ops.begin(), ops.end());

  matchOutputFullyConnectedLayersWithLabels();

  verifyAllowedOutputDim();
}

std::shared_ptr<Model> Model::make(
    autograd::ComputationList inputs, autograd::ComputationList outputs,
    std::vector<loss::LossPtr> losses,
    autograd::ComputationList additional_labels) {
  auto model = std::shared_ptr<Model>(
      new Model(std::move(inputs), std::move(outputs), std::move(losses),
                std::move(additional_labels)));

  // This has to be done here because we need the model to be allocated using a
  // shared_ptr in order to use shared_from_this() to get a valid reference.
  model->registerWithOps();
  return model;
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

void Model::forceStateReallocation() {
  _allocation_manager.forceReallocation();
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

autograd::ComputationList Model::computationOrderWithoutInputs() const {
  return _computation_order;
}

const autograd::ComputationList& Model::inputs() const { return _inputs; }

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

std::pair<const float*, uint64_t> Model::getFlattenedGradients() const {
  return concatenateValues(gradients());
}

std::pair<const float*, uint64_t> Model::getFlattenedParameters() const {
  return concatenateValues(parameters());
}

void Model::setFlattenedGradients(const float* concatenated_values,
                                  uint64_t flattened_dim) const {
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
    if (auto fc = std::dynamic_pointer_cast<ops::FullyConnected>(op)) {
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

proto::bolt::Model Model::toProto(bool with_optimizer) const {
  proto::bolt::Model model;

  for (const auto& op : ops()) {
    model.mutable_ops()->AddAllocated(op->toProto(with_optimizer));
  }

  for (const auto& input : _inputs) {
    auto* placeholder = model.add_inputs();
    placeholder->set_name(input->name());
    placeholder->set_dim(input->dim());
  }

  for (const auto& label : _labels) {
    auto* placeholder = model.add_labels();
    placeholder->set_name(label->name());
    placeholder->set_dim(label->dim());
  }

  for (const auto& comp : _computation_order) {
    auto* comp_proto = model.mutable_computation_graph()->Add();
    comp_proto->set_name(comp->name());
    comp_proto->set_op(comp->op()->name());
    for (const auto& input : comp->inputs()) {
      comp_proto->add_inputs(input->name());
    }
  }

  for (const auto& loss : _losses) {
    model.mutable_losses()->AddAllocated(loss->toProto());
  }

  for (const auto& output : _outputs) {
    model.add_outputs(output->name());
  }

  auto* meta = model.mutable_metadata();
  meta->set_train_steps(_train_steps);
  meta->set_total_training_samples(_total_training_samples);
  meta->set_uuid(_model_uuid);

  return model;
}

std::shared_ptr<Model> Model::fromProto(const proto::bolt::Model& model_proto) {
  std::unordered_map<std::string, ops::OpPtr> ops;

  for (const auto& op_proto : model_proto.ops()) {
    ops[op_proto.name()] = ops::fromProto(op_proto);
  }

  autograd::ComputationList inputs;
  autograd::ComputationList labels;
  std::unordered_map<std::string, autograd::ComputationPtr> computations;

  for (const auto& input_proto : model_proto.inputs()) {
    auto input = ops::Input::make(input_proto.dim());
    computations[input_proto.name()] = input;
    inputs.push_back(input);
  }

  for (const auto& label_proto : model_proto.labels()) {
    auto label = ops::Input::make(label_proto.dim());
    computations[label_proto.name()] = label;
    labels.push_back(label);
  }

  for (const auto& comp_proto : model_proto.computation_graph()) {
    autograd::ComputationList op_inputs;
    for (const auto& input : comp_proto.inputs()) {
      op_inputs.push_back(computations.at(input));
    }

    auto op = ops.at(comp_proto.op());
    computations[comp_proto.name()] = op->apply(op_inputs);
  }

  std::vector<loss::LossPtr> losses;
  for (const auto& loss : model_proto.losses()) {
    losses.push_back(loss::fromProto(loss, computations));
  }

  autograd::ComputationList outputs;
  for (const auto& output : model_proto.outputs()) {
    outputs.push_back(computations.at(output));
  }

  auto model = Model::make(inputs, outputs, losses);

  model->_model_uuid = model_proto.metadata().uuid();
  model->_train_steps = model_proto.metadata().train_steps();
  model->_total_training_samples =
      model_proto.metadata().total_training_samples();

  return model;
}

void Model::saveProto(const std::string& filename, bool with_optimizer) const {
  auto proto = toProto(with_optimizer);

  std::ofstream output = dataset::SafeFileIO::ofstream(filename);
  proto.SerializeToOstream(&output);
}

std::shared_ptr<Model> Model::loadProto(const std::string& filename) {
  proto::bolt::Model proto;

  std::ifstream input = dataset::SafeFileIO::ifstream(filename);
  if (!proto.ParseFromIstream(&input)) {
    throw std::invalid_argument("Error parsing protobuf archive.");
  }

  return Model::fromProto(proto);
}

std::string Model::serializeProto(bool with_optimizer) const {
  auto proto = toProto(with_optimizer);
  return proto.SerializeAsString();
}

std::shared_ptr<Model> Model::deserializeProto(const std::string& binary) {
  proto::bolt::Model proto;

  if (!proto.ParseFromString(binary)) {
    throw std::invalid_argument("Error parsing protobuf archive.");
  }

  return Model::fromProto(proto);
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

void Model::unfreezeHashTables() {
  for (auto& op : _ops) {
    if (auto fc = ops::FullyConnected::cast(op)) {
      // insert_labels_if_not_found will have no effect on non output layers
      // because they will not have access to labels.
      fc->unfreezeHashTables();
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

void Model::registerWithOps() {
  for (auto& op : _ops) {
    op->registerModel(weak_from_this());
  }
}

void Model::nameComputations(autograd::ComputationList& inputs,
                             autograd::ComputationList& comps,
                             autograd::ComputationList& labels) {
  uint32_t comp_count = 0;
  auto next_name = [&comp_count]() {
    return "tensor_" + std::to_string(++comp_count);
  };
  std::unordered_set<autograd::ComputationPtr> visited;
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

  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);

  std::string class_name = "BOLT_MODEL";
  versions::checkVersion(version, versions::BOLT_MODEL_VERSION, thirdai_version,
                         thirdai::version(), class_name);

  // Increment thirdai::versions::BOLT_MODEL_VERSION after serialization changes
  archive(_inputs, _outputs, _labels, _losses, _ops, _computation_order,
          _allocation_manager, _train_steps, _model_uuid,
          _total_training_samples);

  verifyAllowedOutputDim();

  registerWithOps();
}

}  // namespace thirdai::bolt::nn::model

CEREAL_CLASS_VERSION(thirdai::bolt::nn::model::Model,
                     thirdai::versions::BOLT_MODEL_VERSION)