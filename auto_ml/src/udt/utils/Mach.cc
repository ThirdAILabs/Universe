#include "Mach.h"
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/columns/Column.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/State.h>
#include <dataset/src/mach/MachIndex.h>
#include <algorithm>
#include <iostream>
#include <ostream>
#include <random>
#include <strings.h>
#include <unordered_map>
#include <utility>

namespace thirdai::automl::udt::utils {

bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
}

Mach::Mach(uint32_t input_dim, uint32_t num_buckets,
           const config::ArgumentMap& args,
           const std::optional<std::string>& model_config, bool use_sigmoid_bce,
           uint32_t num_hashes, float mach_sampling_threshold,
           bool freeze_hash_tables, std::string input_indices_column,
           std::string input_values_column, std::string label_column,
           std::string bucket_column)

    : _model(buildModel(input_dim, num_buckets, args, model_config,
                        use_sigmoid_bce, /* mach= */ true)),
      _emb(getEmbeddingComputation(*_model)),
      _mach_sampling_threshold(mach_sampling_threshold),
      _freeze_hash_tables(freeze_hash_tables),
      _state(data::State::make(dataset::mach::MachIndex::make(
          /* num_buckets= */ _model->outputs().front()->dim(),
          /* num_hashes=*/num_hashes))),
      _label_to_buckets(data::MachLabel::make(label_column, bucket_column)),
      _bolt_input_columns(
          {data::OutputColumns(input_indices_column, input_values_column)}),
      _bolt_label_columns(
          {data::OutputColumns(bucket_column,
                               use_sigmoid_bce ? data::ValueFillType::Ones
                                               : data::ValueFillType::SumToOne),
           data::OutputColumns(label_column)}),
      _all_bolt_columns({std::move(input_indices_column),
                         std::move(input_values_column),
                         std::move(label_column), std::move(bucket_column)}) {
  updateSamplingStrategy();
}

void Mach::randomlyIntroduceEntities(const data::ColumnMap& columns) {
  const auto& labels = columns.getArrayColumn<uint32_t>(labelColumn());

  for (size_t row = 0; row < columns.numRows(); row++) {
    if (labels->row(row).size() < 1) {
      continue;
    }

    std::vector<uint32_t> hashes(index()->numHashes());
    for (uint32_t h = 0; h < index()->numHashes(); h++) {
      std::uniform_int_distribution<uint32_t> dist(0,
                                                   index()->numBuckets() - 1);
      auto hash = dist(_mt);
      while (std::find(hashes.begin(), hashes.end(), hash) != hashes.end()) {
        hash = dist(_mt);
      }
      hashes[h] = hash;
    }

    index()->insert(labels->row(row)[0], std::move(hashes));
  }
}

void Mach::introduceEntities(const data::ColumnMap& columns,
                             std::optional<uint32_t> num_buckets_to_sample_opt,
                             uint32_t num_random_hashes) {
  uint32_t num_buckets_to_sample =
      num_buckets_to_sample_opt.value_or(index()->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  bolt::python::CtrlCCheck ctrl_c_check;

  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto scores = _model->forward(inputTensors(columns)).at(0);

  ctrl_c_check();

  auto doc_ids = columns.getArrayColumn<uint32_t>(labelColumn());
  for (uint32_t row = 0; row < scores->batchSize(); row++) {
    uint32_t label = doc_ids->row(row)[0];
    top_k_per_doc[label].push_back(
        scores->getVector(row).findKLargestActivations(num_buckets_to_sample));
    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), num_buckets_to_sample,
                                  num_random_hashes);
    index()->insert(doc, hashes);

    ctrl_c_check();
  }

  updateSamplingStrategy();
}

void Mach::eraseEntity(uint32_t entity) {
  index()->erase(entity);
  if (_state->hasRlhfSampler()) {
    _state->rlhfSampler().removeDoc(entity);
  }

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
  if (_state->hasRlhfSampler()) {
    _state->rlhfSampler().clear();
  }
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

  if (top_k > size()) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. Model currently can predict one of " +
        std::to_string(size()) + " classes.");
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
  assertRlhfEnabled();

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

std::vector<std::vector<std::pair<uint32_t, double>>> Mach::score(
    const data::ColumnMap& columns,
    std::vector<std::unordered_set<uint32_t>>& entities,
    std::optional<uint32_t> top_k) {
  if (columns.numRows() != entities.size()) {
    throw std::invalid_argument(
        "Length of entities list must be equal to the number of rows in the "
        "column.");
  }

  // sparse inference could become an issue here because maybe the entities
  // we score wouldn't otherwise be in the top results, thus their buckets
  // have
  // lower similarity and don't get selected by LSH
  auto outputs =
      _model->forward(inputTensors(columns), /* use_sparsity= */ false).at(0);

  size_t batch_size = columns.numRows();
  std::vector<std::vector<std::pair<uint32_t, double>>> scores(batch_size);

#pragma omp parallel for default(none) \
    shared(entities, outputs, scores, top_k, batch_size) if (batch_size > 1)
  for (uint32_t i = 0; i < batch_size; i++) {
    const BoltVector& vector = outputs->getVector(i);
    scores[i] = index()->scoreEntities(vector, entities[i], top_k);
  }

  return scores;
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

void Mach::enableRlhf(uint32_t num_balancing_docs,
                      uint32_t num_balancing_samples_per_doc) {
  _add_balancing_samples = data::AddMachRlhfSamples::make(
      inputIndicesColumn(), inputValuesColumn(), labelColumn(), bucketColumn());
  _state->setRlhfSampler(RLHFSampler(
      /* max_docs= */ num_balancing_docs,
      /* max_samples_per_doc= */ num_balancing_samples_per_doc));
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

  std::unordered_map<std::string, data::ColumnPtr> columns(
      {{inputIndicesColumn(),
        data::ArrayColumn<uint32_t>::make(std::move(indices), inputDim())},
       {inputValuesColumn(), data::ArrayColumn<float>::make(std::move(values))},
       {bucketColumn(),
        data::ArrayColumn<uint32_t>::make(std::move(buckets), numBuckets())}});

  data::ColumnMap map(std::move(columns));
  addDummyLabels(map);

  return map;
}

void Mach::teach(data::ColumnMap feedback, float learning_rate,
                 uint32_t feedback_repetitions, uint32_t num_balancers,
                 uint32_t epochs, size_t batch_size) {
  assertRlhfEnabled();
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

void Mach::updateSamplingStrategy() {
  auto output_layer =
      bolt::FullyConnected::cast(_model->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = index()->sparsity();
  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::MachNeuronIndex::make(index());
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::MachNeuronIndex>(neuron_index)) {
      float sparsity = autotuneSparsity(index()->numBuckets());

      auto sampling_config = bolt::DWTASamplingConfig::autotune(
          index()->numBuckets(), sparsity,
          /* experimental_autotune= */ false);

      output_layer->setSparsity(sparsity, false, false);

      if (sampling_config) {
        auto new_index = sampling_config->getNeuronIndex(
            output_layer->dim(), output_layer->inputDim());
        output_layer->kernel()->setNeuronIndex(new_index);
      }
    }
  }
}

struct BucketScore {
  uint32_t frequency = 0;
  float score = 0.0;
};

struct CompareBuckets {
  bool operator()(const std::pair<uint32_t, BucketScore>& lhs,
                  const std::pair<uint32_t, BucketScore>& rhs) {
    if (lhs.second.frequency == rhs.second.frequency) {
      return lhs.second.score > rhs.second.score;
    }
    return lhs.second.frequency > rhs.second.frequency;
  }
};

std::vector<uint32_t> Mach::topHashesForDoc(
    std::vector<TopKActivationsQueue>&& top_k_per_sample,
    uint32_t num_buckets_to_sample, uint32_t num_random_hashes) {
  uint32_t num_hashes = index()->numHashes();

  auto& mach_index = *index();

  if (num_buckets_to_sample < mach_index.numHashes()) {
    throw std::invalid_argument(
        "Sampling from fewer buckets than num_hashes is not supported. If "
        "you'd like to introduce using fewer hashes, please reset the number "
        "of hashes for the index.");
  }

  if (num_buckets_to_sample > mach_index.numBuckets()) {
    throw std::invalid_argument(
        "Cannot sample more buckets than there are in the index.");
  }

  std::unordered_map<uint32_t, BucketScore> hash_freq_and_scores;
  for (auto& top_k : top_k_per_sample) {
    while (!top_k.empty()) {
      auto [activation, active_neuron] = top_k.top();
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, activation};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += activation;
      }
      top_k.pop();
    }
  }

  // We sort the hashes first by number of occurances and tiebreak with
  // the higher aggregated score if necessary. We don't only use the
  // activations since those typically aren't as useful as the
  // frequencies.
  std::vector<std::pair<uint32_t, BucketScore>> sorted_hashes(
      hash_freq_and_scores.begin(), hash_freq_and_scores.end());

  CompareBuckets cmp;
  std::sort(sorted_hashes.begin(), sorted_hashes.end(), cmp);

  if (num_buckets_to_sample > num_hashes) {
    // If we are sampling more buckets then we end up using we rerank the
    // buckets based on size to load balance the index.
    std::sort(sorted_hashes.begin(),
              sorted_hashes.begin() + num_buckets_to_sample,
              [&mach_index, &cmp](const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index.bucketSize(lhs.first);
                size_t rhs_size = mach_index.bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.
                if (lhs_size == rhs_size) {
                  return cmp(lhs, rhs);
                }

                return lhs_size < rhs_size;
              });
  }

  std::vector<uint32_t> new_hashes;

  // We can optionally specify the number of hashes we'd like to be
  // random for a new document. This is to encourage an even distribution
  // among buckets.
  if (num_random_hashes > num_hashes) {
    throw std::invalid_argument(
        "num_random_hashes cannot be greater than num hashes.");
  }

  uint32_t num_informed_hashes = num_hashes - num_random_hashes;
  for (uint32_t i = 0; i < num_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  uint32_t num_buckets = mach_index.numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(thirdai::global_random::nextSeed());

  for (uint32_t i = 0; i < num_random_hashes; i++) {
    new_hashes.push_back(int_dist(rand));
  }

  return new_hashes;
}

template void Mach::serialize(cereal::BinaryInputArchive&);
template void Mach::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Mach::serialize(Archive& archive) {
  archive(_model, _emb, _mach_sampling_threshold, _freeze_hash_tables, _state,
          _label_to_buckets, _add_balancing_samples, _bolt_input_columns,
          _bolt_label_columns, _all_bolt_columns);
}

}  // namespace thirdai::automl::udt::utils