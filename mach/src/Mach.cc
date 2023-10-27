#include "Mach.h"
#include "Model.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <data/src/transformations/AddMachRlhfSamples.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <random>
#include <utility>

namespace thirdai::mach {

Mach::Mach(const bolt::Model& model, uint32_t num_hashes,
           float mach_sampling_threshold, bool freeze_hash_tables,
           std::string input_indices_column, std::string input_values_column,
           std::string label_column, std::string bucket_column, bool use_rlhf,
           uint32_t num_balancing_docs, uint32_t num_balancing_samples_per_doc)

    : _model(modifyForMach(model)),
      _emb(getEmbeddingComputation(*_model)),
      _mach_sampling_threshold(mach_sampling_threshold),
      _freeze_hash_tables(freeze_hash_tables),
      _state(data::State::make(dataset::mach::MachIndex::make(
          /* num_buckets= */ model.outputs().front()->dim(),
          /* num_hashes=*/num_hashes))),
      _label_to_buckets(data::MachLabel::make(label_column, bucket_column)),
      _add_balancing_samples(data::AddMachRlhfSamples::make(
          input_indices_column, input_values_column, label_column,
          bucket_column)),
      _bolt_input_columns(
          {data::OutputColumns(input_indices_column, input_values_column)}),
      _bolt_label_columns(
          {data::OutputColumns(bucket_column, inferLabelValueFill(model)),
           data::OutputColumns(label_column)}),
      _all_bolt_columns({std::move(input_indices_column),
                         std::move(input_values_column),
                         std::move(label_column), std::move(bucket_column)}) {
  if (use_rlhf) {
    enableRlhf(num_balancing_docs, num_balancing_samples_per_doc);
  }

  updateSamplingStrategy();
}

void Mach::randomlyIntroduceEntities(const data::ColumnMap& columns) {
  uint32_t num_hashes = index()->numBuckets();
  const auto& labels = columns.getArrayColumn<uint32_t>(labelColumn());

  for (size_t i = 0; i < columns.numRows(); i++) {
    if (labels->row(i).size() < 1) {
      continue;
    }

    std::vector<uint32_t> hashes(num_hashes);
    for (uint32_t i = 0; i < num_hashes; i++) {
      std::uniform_int_distribution<uint32_t> dist(0,
                                                   index()->numBuckets() - 1);
      auto hash = dist(_mt);
      while (std::find(hashes.begin(), hashes.end(), hash) != hashes.end()) {
        hash = dist(_mt);
      }
      hashes[i] = hash;
    }

    index()->insert(labels->row(i)[0], std::move(hashes));
  }
}

void Mach::eraseEntity(uint32_t entity) {
  index()->erase(entity);
  _state->rlhfSampler().removeDoc(entity);

  if (index()->numEntities() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }

  updateSamplingStrategy();
}

void Mach::eraseAllEntities() {
  index()->clear();
  _state->rlhfSampler().clear();
  updateSamplingStrategy();
}

bolt::metrics::History Mach::train(
    data::ColumnMapIteratorPtr train_iter, data::ColumnMapIteratorPtr val_iter,
    float learning_rate, uint32_t epochs,
    const bolt::metrics::InputMetrics& train_metrics,
    const bolt::metrics::InputMetrics& val_metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    TrainOptions options, const bolt::DistributedCommPtr& comm) {
  std::optional<uint32_t> freeze_hash_tables_epoch;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        bolt::python::CtrlCCheck{});

  return trainer.train_with_data_loader(
      /* train_data_loader= */
      loader(std::move(train_iter), /* train= */ true, options),
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */ train_metrics,
      /* validation_data_loader= */
      val_iter ? loader(std::move(val_iter), /* train= */ false, options)
               : nullptr,
      /* validation_metrics= */ val_metrics,
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);
}

void Mach::train(data::ColumnMap columns, float learning_rate) {
  addMachLabels(columns);
  addRlhfSamplesIfNeeded(columns);
  _model->trainOnBatch(inputTensors(columns), labelTensors(columns));
  _model->updateParameters(learning_rate);
}

void Mach::trainBuckets(data::ColumnMap columns, float learning_rate) {
  addDummyLabels(columns);
  addRlhfSamplesIfNeeded(columns);
  _model->trainOnBatch(inputTensors(columns), labelTensors(columns));
  _model->updateParameters(learning_rate);
}

bolt::metrics::History Mach::evaluate(
    data::ColumnMapIteratorPtr eval_iter,
    const bolt::metrics::InputMetrics& metrics, bool sparse_inference,
    bool verbose) {
  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});
  return trainer.validate_with_data_loader(
      loader(std::move(eval_iter), /* train= */ false, TrainOptions()), metrics,
      sparse_inference, verbose);
}

std::vector<std::vector<std::pair<uint32_t, double>>> Mach::predict(
    const data::ColumnMap& columns, bool sparse_inference, uint32_t top_k,
    uint32_t num_scanned_buckets) {
  auto outputs = _model->forward(inputTensors(columns), sparse_inference).at(0);

  uint32_t num_classes = index()->numEntities();
  if (top_k && top_k > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  uint32_t batch_size = outputs->batchSize();

  std::vector<std::vector<std::pair<uint32_t, double>>> predicted_entities(
      batch_size);
#pragma omp parallel for default(none)                     \
    shared(outputs, predicted_entities, top_k, batch_size, \
           num_scanned_buckets) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    auto predictions = index()->decode(
        /* output = */ vector,
        /* top_k = */ top_k,
        /* num_buckets_to_eval = */ num_scanned_buckets);
    predicted_entities[i] = predictions;
  }

  return predicted_entities;
}

std::vector<std::vector<uint32_t>> Mach::predictBuckets(
    const data::ColumnMap& columns, bool sparse_inference,
    std::optional<uint32_t> top_k, bool force_non_empty) {
  auto outputs = _model->forward(inputTensors(columns), sparse_inference).at(0);

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, top_k, force_non_empty)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    uint32_t k = top_k.value_or(_state->machIndex()->numHashes());
    if (force_non_empty) {
      heap = _state->machIndex()->topKNonEmptyBuckets(output, k);
    } else {
      heap = output.findKLargestActivations(k);
    }

    std::vector<uint32_t> hashes;
    while (!heap.empty()) {
      auto [_, active_neuron] = heap.top();
      hashes.push_back(active_neuron);
      heap.pop();
    }

    std::reverse(hashes.begin(), hashes.end());

    all_hashes[i] = hashes;
  }

  return all_hashes;
}

auto repeatRows(data::ColumnMap&& columns, uint32_t repetitions) {
  std::vector<size_t> permutation(columns.numRows() * repetitions);
  for (uint32_t rep = 0; rep < repetitions; rep++) {
    auto begin = permutation.begin() + rep * columns.numRows();
    std::iota(begin, begin + columns.numRows(), 0);
  }
  return columns.permute(permutation);
}

void Mach::upvote(data::ColumnMap upvotes, float learning_rate,
                  uint32_t repeats, uint32_t num_balancers, uint32_t epochs,
                  size_t batch_size) {
  addMachLabels(upvotes);
  teach(std::move(upvotes), learning_rate, repeats, num_balancers, epochs,
        batch_size);
}

void Mach::associate(data::ColumnMap from_table,
                     const data::ColumnMap& to_table, float learning_rate,
                     uint32_t repeats, uint32_t num_balancers,
                     uint32_t num_buckets, uint32_t epochs, size_t batch_size) {
  return teach(associateSamples(std::move(from_table), to_table, num_buckets),
               learning_rate, repeats, num_balancers, epochs, batch_size);
}

bolt::metrics::History Mach::associateTrain(
    data::ColumnMap from_table, const data::ColumnMap& to_table,
    data::ColumnMap train_data, float learning_rate, uint32_t repeats,
    uint32_t num_buckets, uint32_t epochs, size_t batch_size,
    const bolt::metrics::InputMetrics& metrics, TrainOptions options) {
  auto associations = repeatRows(
      associateSamples(std::move(from_table), to_table, num_buckets), repeats);

  addMachLabels(train_data);
  addRlhfSamplesIfNeeded(train_data);

  train_data = train_data.keepColumns(_all_bolt_columns)
                   .concat(associations.keepColumns(_all_bolt_columns));

  bolt::Trainer trainer(_model);
  return trainer.train(
      /* train_data= */ data::toLabeledDataset(train_data, _bolt_input_columns,
                                               _bolt_label_columns, batch_size),
      /* learning_rate= */ learning_rate,
      /* epochs= */ epochs,
      /* train_metrics= */ metrics,
      /* validation_data= */ {},
      /* validation_metrics= */ {},
      /* steps_per_validation= */ {},
      /* use_sparsity_in_validation= */ false,
      /* callbacks= */ {},
      /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval);
}

std::vector<uint32_t> Mach::outputCorrectness(
    const data::ColumnMap& columns, const std::vector<uint32_t>& labels,
    std::optional<uint32_t> num_hashes, bool sparse_inference) {
  auto top_buckets = predictBuckets(columns, sparse_inference, num_hashes,
                                    /* force_non_empty= */ true);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes = index()->getHashes(labels[i]);
      uint32_t count = 0;
      for (auto hash : hashes) {
        if (std::count(top_buckets[i].begin(), top_buckets[i].end(), hash) >
            0) {
          count++;
        }
      }
      matching_buckets[i] = count;
    } catch (const std::exception& e) {
#pragma omp critical
      hashes_err = std::current_exception();
    }
  }

  if (hashes_err) {
    std::rethrow_exception(hashes_err);
  }

  return matching_buckets;
}

bolt::TensorPtr Mach::embedding(const data::ColumnMap& columns) {
  // TODO(Nicholas): Sparsity could speed this up, and wouldn't affect the
  // embeddings if the sparsity is in the output layer and the embeddings are
  // from the hidden layer.
  _model->forward(inputTensors(columns), /* use_sparsity= */ false);
  return _emb->tensor();
}

std::vector<float> Mach::entityEmbedding(uint32_t entity) const {
  std::vector<uint32_t> buckets = index()->getHashes(entity);

  auto outputs = _model->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto fc_layer = fc->kernel();

  std::vector<float> averaged_embedding(fc_layer->getInputDim());
  for (uint32_t neuron_id : buckets) {
    auto weights = fc_layer->getWeightsByNeuron(neuron_id);
    if (weights.size() != averaged_embedding.size()) {
      throw std::invalid_argument("Output dim mismatch.");
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
      averaged_embedding[i] += weights[i];
    }
  }

  // TODO(david) try averaging and summing
  for (float& weight : averaged_embedding) {
    weight /= averaged_embedding.size();
  }

  return averaged_embedding;
}

std::optional<data::ColumnMap> Mach::balancingColumnMap(
    uint32_t num_balancers) {
  auto balancers = _state->rlhfSampler().balancingSamples(num_balancers);
  if (balancers.empty()) {
    return {};
  }

  std::vector<std::vector<uint32_t>> indices;
  std::vector<std::vector<float>> values;
  std::vector<std::vector<uint32_t>> buckets;
  indices.reserve(balancers.size());
  values.reserve(balancers.size());
  buckets.reserve(balancers.size());

  for (auto& sample : balancers) {
    indices.push_back(std::move(sample.input_indices));
    values.push_back(std::move(sample.input_values));
    buckets.push_back(std::move(sample.mach_buckets));
  }

  data::ColumnMap columns({});

  columns.setColumn(inputIndicesColumn(), data::ArrayColumn<uint32_t>::make(
                                              std::move(indices), inputDim()));
  columns.setColumn(inputValuesColumn(),
                    data::ArrayColumn<float>::make(std::move(values)));
  columns.setColumn(bucketColumn(), data::ArrayColumn<uint32_t>::make(
                                        std::move(buckets), numBuckets()));

  return columns;
}

void Mach::teach(data::ColumnMap feedback, float learning_rate,
                 uint32_t feedback_repetitions, uint32_t num_balancers,
                 uint32_t epochs, size_t batch_size) {
  feedback = repeatRows(std::move(feedback), feedback_repetitions);
  feedback = feedback.keepColumns(_all_bolt_columns);

  auto balancers = balancingColumnMap(num_balancers);
  if (balancers) {
    balancers = balancers->keepColumns(_all_bolt_columns);
    feedback = feedback.concat(*balancers);
  }

  auto train_data = thirdai::data::toLabeledDataset(
      feedback, _bolt_input_columns, _bolt_label_columns, batch_size);

  for (uint32_t epoch = 0; epoch < epochs; epoch++) {
    for (uint32_t batch = 0; batch < train_data.first.size(); batch++) {
      _model->trainOnBatch(train_data.first.at(batch),
                           train_data.second.at(batch));
      _model->updateParameters(learning_rate);
    }
  }
}

data::ColumnMap Mach::associateSamples(data::ColumnMap from_columns,
                                       const data::ColumnMap& to_columns,
                                       uint32_t num_buckets) {
  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      predictBuckets(to_columns, /* sparse_inference= */ false, num_buckets,
                     /* force_non_empty= */ true),
      index()->numBuckets());
  from_columns.setColumn(bucketColumn(), mach_labels);
  addDummyLabels(from_columns);
  return from_columns;
}

template void Mach::serialize(cereal::BinaryInputArchive&);
template void Mach::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Mach::serialize(Archive& archive) {
  archive(_model, _emb, _mach_sampling_threshold, _freeze_hash_tables, _state,
          _label_to_buckets, _add_balancing_samples, _bolt_input_columns,
          _bolt_label_columns, _all_bolt_columns);
}

}  // namespace thirdai::mach