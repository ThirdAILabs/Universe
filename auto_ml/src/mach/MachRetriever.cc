#include "MachRetriever.h"
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/mach/MachConfig.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/Transformation.h>
#include <cassert>
#include <optional>
#include <regex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::mach {

MachRetriever::MachRetriever(const MachConfig& config)
    : _state(config.state()),
      _model(config.model()),
      _text_column(config.getTextCol()),
      _id_column(config.getIdCol()),
      _text_transform(config.textTransformation()),
      _map_to_buckets(config.mapToBucketsTransform()),
      _add_mach_memory_samples(std::make_shared<data::AddMachMemorySamples>()),
      _bolt_input_columns({{input_indices_column, input_values_column}}),
      _bolt_label_columns(
          {data::OutputColumns(bucket_column,
                               config.usesSoftmax()
                                   ? data::ValueFillType::SumToOne
                                   : data::ValueFillType::Ones),
           data::OutputColumns(config.getIdCol())}),
      _all_bolt_columns({input_indices_column, input_values_column,
                         bucket_column, config.getIdCol()}),
      _mach_sampling_threshold(config.getMachSamplingThreshold()),
      _n_buckets_to_eval(config.getNBucketsToEval()),
      _freeze_tables_epoch(config.getFeezeHashTablesEpoch()) {}

using BucketScores = std::vector<ValueIndexPair>;

std::unordered_map<uint32_t, std::vector<BucketScores>> groupScoresByLabel(
    bolt::Tensor& scores, const data::ValueColumnBase<uint32_t>& labels,
    std::optional<uint32_t> only_keep_top_k) {
  assert(scores.batchSize() == labels.numRows());
  std::unordered_map<uint32_t, std::vector<BucketScores>> grouped;
  for (uint32_t i = 0; i < scores.batchSize(); i++) {
    auto& vector = scores.getVector(i);
    auto bucket_scores = only_keep_top_k
                             ? vector.topKNeuronsAsVector(*only_keep_top_k)
                             : vector.valueIndexPairs();
    grouped[labels.value(i)].push_back(bucket_scores);
  }
  return grouped;
}

void MachRetriever::introduce(
    const data::ColumnMapIteratorPtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool text_augmentation,
    std::optional<uint32_t> n_buckets_to_sample_opt, uint32_t n_random_hashes,
    bool load_balancing, bool sort_random_hashes) {
  while (auto columns = data->next()) {
    introduce(std::move(*columns), strong_column_names, weak_column_names,
              text_augmentation, n_buckets_to_sample_opt, n_random_hashes,
              load_balancing, sort_random_hashes);
  }
}

void MachRetriever::introduce(
    data::ColumnMap data, const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool text_augmentation,
    std::optional<uint32_t> n_buckets_to_sample_opt, uint32_t n_random_hashes,
    bool load_balancing, bool sort_random_hashes) {
  assertUniqueIds(data);

  if (text_augmentation) {
    textAugmentation(strong_column_names, weak_column_names, std::nullopt,
                     std::nullopt)
        ->apply(std::move(data), *_state);
  } else {
    textConcat(strong_column_names, weak_column_names)
        ->apply(std::move(data), *_state);
  }
  data = _text_transform->apply(std::move(data), *_state);
  auto input_tensors = inputTensors(data);

  // Perform introduction

  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.

  auto scores = _model->forward(input_tensors).at(0);

  uint32_t n_buckets_to_sample =
      load_balancing ? index()->numBuckets()
                     : n_buckets_to_sample_opt.value_or(index()->numHashes());

  auto bucket_scores_by_doc = groupScoresByLabel(
      /* scores= */ *scores,
      /* labels= */ *data.getValueColumn<uint32_t>(_id_column),
      /* only_keep_top_k= */
      load_balancing ? std::nullopt : std::make_optional(load_balancing));

  uint32_t approx_n_hashes_per_bucket = index()->approxNumHashesPerBucket(
      /* num_new_samples= */ bucket_scores_by_doc.size());

  for (auto& [doc, bucket_scores] : bucket_scores_by_doc) {
    auto hashes = topHashesForDoc(std::move(bucket_scores), n_buckets_to_sample,
                                  approx_n_hashes_per_bucket, n_random_hashes,
                                  load_balancing, sort_random_hashes);
    index()->insert(doc, hashes);
  }

  updateSamplingStrategy();
}

bolt::metrics::History MachRetriever::coldstart(
    const data::ColumnMapIteratorPtr& data,
    const std::vector<std::string>& strong_cols,
    const std::vector<std::string>& weak_cols, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const ColdStartOptions& options) {
  auto augmented_data = data::TransformedIterator::make(
      data,
      textAugmentation(strong_cols, weak_cols, options.variable_length,
                       options.splade_config),
      _state);

  return train(augmented_data, learning_rate, epochs, metrics, callbacks,
               options);
}

bolt::metrics::History MachRetriever::train(
    const data::ColumnMapIteratorPtr& data, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const TrainOptions& options) {
  insertNewIds(data);

  auto train_transform = data::Pipeline::make(
      {_text_transform, _map_to_buckets, _add_mach_memory_samples});

  auto train_data_loader = data::Loader::make(
      data, train_transform, _state, _bolt_input_columns, _bolt_label_columns,
      options.batch_size, /* shuffle= */ true, options.verbose);

  bolt::Trainer trainer(_model, _freeze_tables_epoch,
                        /* gradient_update_interval */ 1,
                        options.interrupt_check);

  return trainer.train_with_data_loader(
      /* train_data_loader= */ train_data_loader,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */ getMetrics(metrics, "train_"),
      /* validation_data_loader= */ nullptr,
      /* validation_metrics= */ {},
      /* steps_per_validation= */ std::nullopt,
      /* use_sparsity_in_validation= */ false,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose);
}

bolt::metrics::History MachRetriever::evaluate(
    const data::ColumnMapIteratorPtr& data,
    const std::vector<std::string>& metrics, bool verbose) {
  bolt::Trainer trainer(_model, _freeze_tables_epoch,
                        /* gradient_update_interval */ 1);

  auto transform = data::Pipeline::make({_text_transform, _map_to_buckets});

  auto eval_data_loader = data::Loader::make(
      data, transform, _state, _bolt_input_columns, _bolt_label_columns, 2048,
      /* shuffle= */ false, verbose);

  return trainer.validate_with_data_loader(
      eval_data_loader, getMetrics(metrics, "val_"),
      /*use_sparsity=*/false, /*verbose=*/verbose);
}

std::vector<IdScores> MachRetriever::search(data::ColumnMap queries,
                                            uint32_t top_k,
                                            bool sparse_inference) {
  uint32_t n_queries = queries.numRows();
  auto in = inputTensors(_text_transform->apply(std::move(queries), *_state));
  auto out = _model->forward(in, sparse_inference).at(0);

  std::vector<IdScores> predictions(n_queries);
#pragma omp parallel for default(none) \
    shared(out, predictions, top_k, n_queries) if (n_queries > 1)
  for (uint32_t i = 0; i < n_queries; i++) {
    const BoltVector& out_vec = out->getVector(i);
    predictions[i] = index()->decode(out_vec, top_k, _n_buckets_to_eval);
  }

  return predictions;
}

std::vector<IdScores> MachRetriever::rank(
    data::ColumnMap queries,
    const std::vector<std::unordered_set<uint32_t>>& candidates,
    std::optional<uint32_t> top_k, bool sparse_inference) {
  assert(queries.numRows() == choices.size());

  auto in = inputTensors(_text_transform->apply(std::move(queries), *_state));
  auto out = _model->forward(in, sparse_inference).at(0);

  uint32_t n_queries = queries.numRows();
  std::vector<IdScores> predictions(n_queries);
#pragma omp parallel for default(none) \
    shared(out, candidates, predictions, top_k, n_queries) if (n_queries > 1)
  for (uint32_t i = 0; i < n_queries; i++) {
    const BoltVector& out_vec = out->getVector(i);
    predictions[i] = index()->scoreEntities(out_vec, candidates[i], top_k);
  }

  return predictions;
}

std::vector<std::vector<uint32_t>> MachRetriever::predictBuckets(
    const data::ColumnMap& columns, bool sparse_inference,
    std::optional<uint32_t> top_k, bool force_non_empty) {
  auto inputs = inputTensors(_text_transform->applyStateless(columns));
  auto outputs = _model->forward(inputs, sparse_inference).at(0);

  uint32_t k = top_k.value_or(_state->machIndex()->numHashes());

  std::vector<std::vector<uint32_t>> all_hashes(outputs->batchSize());
#pragma omp parallel for default(none) \
    shared(outputs, all_hashes, k, force_non_empty)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& output = outputs->getVector(i);

    TopKActivationsQueue heap;
    if (force_non_empty) {
      heap = index()->topKNonEmptyBuckets(output, k);
    } else {
      heap = output.topKNeurons(k);
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
  for (size_t i = 0; i < permutation.size(); i++) {
    permutation[i] = i / repetitions;
  }
  return columns.permute(permutation);
}

void MachRetriever::upvote(data::ColumnMap upvotes, uint32_t n_upvote_samples,
                           uint32_t n_balancing_samples, float learning_rate,
                           uint32_t epochs, size_t batch_size) {
  upvotes = data::Pipeline({_text_transform, _map_to_buckets})
                .apply(std::move(upvotes), *_state);
  teach(upvotes, learning_rate, n_upvote_samples,
        n_balancing_samples * upvotes.numRows(), epochs, batch_size);
}

void MachRetriever::associate(data::ColumnMap sources,
                              const data::ColumnMap& targets,
                              uint32_t n_buckets,
                              uint32_t n_association_samples,
                              uint32_t n_balancing_samples, float learning_rate,
                              uint32_t epochs, bool force_non_empty,
                              size_t batch_size) {
  auto buckets = predictBuckets(targets, /* sparse_inference= */ false,
                                std::max(index()->numBuckets(), n_buckets),
                                force_non_empty);

  auto texts = sources.getValueColumn<std::string>(_text_column);
  std::vector<std::string> source_samples;
  std::vector<std::vector<uint32_t>> mach_labels;
  std::mt19937 rng(global_random::nextSeed());
  for (size_t i = 0; i < buckets.size(); i++) {
    for (size_t j = 0; j < n_association_samples; j++) {
      std::vector<uint32_t> sampled_buckets;
      std::sample(buckets[i].begin(), buckets[i].end(),
                  std::back_inserter(sampled_buckets), n_buckets, rng);
      mach_labels.push_back(sampled_buckets);
      source_samples.push_back(texts->value(i));
    }
  }

  sources = data::ColumnMap(
      {{_text_column,
        data::ValueColumn<std::string>::make(std::move(source_samples))}});

  sources = _text_transform->applyStateless(sources);
  sources.setColumn(bucket_column,
                    data::ArrayColumn<uint32_t>::make(std::move(mach_labels),
                                                      index()->numBuckets()));

  // Add dummy IDs since associations do not have IDs.
  auto doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(sources.numRows(), 0),
      std::numeric_limits<uint32_t>::max());
  sources.setColumn(_id_column, doc_ids);

  teach(sources, learning_rate, 1, n_balancing_samples * targets.numRows(),
        epochs, batch_size);
}

void MachRetriever::teach(data::ColumnMap feedback, float learning_rate,
                          uint32_t feedback_repetitions,
                          uint32_t total_balancers, uint32_t epochs,
                          size_t batch_size) {
  auto balancers = _state->machMemory().getSamples(total_balancers);

  feedback = repeatRows(std::move(feedback), feedback_repetitions);
  feedback = feedback.selectColumns(_all_bolt_columns);

  if (balancers) {
    balancers = balancers->selectColumns(_all_bolt_columns);
    feedback = feedback.concat(*balancers);
  }

  feedback.shuffle();

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

std::vector<uint32_t> MachRetriever::topHashesForDoc(
    std::vector<std::vector<ValueIndexPair>>&& top_k_per_sample,
    uint32_t n_buckets_to_sample, uint32_t approx_n_hashes_per_bucket,
    uint32_t n_random_hashes, bool load_balancing,
    bool sort_random_hashes) const {
  const auto& mach_index = _state->machIndex();

  uint32_t n_hashes = mach_index->numHashes();

  if (n_buckets_to_sample < mach_index->numHashes()) {
    throw std::invalid_argument(
        "Sampling from fewer buckets than n_hashes is not supported. If "
        "you'd like to introduce using fewer hashes, please reset the number "
        "of hashes for the index.");
  }

  if (n_buckets_to_sample > mach_index->numBuckets()) {
    throw std::invalid_argument(
        "Cannot sample more buckets than there are in the index.");
  }

  std::unordered_map<uint32_t, BucketScore> hash_freq_and_scores;
  for (auto& top_k : top_k_per_sample) {
    for (const auto& [activation, active_neuron] : top_k) {
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, activation};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += activation;
      }
    }
  }

  uint32_t n_buckets = mach_index->numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, n_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  if (sort_random_hashes) {
    for (uint32_t i = 0; i < n_random_hashes; i++) {
      uint32_t active_neuron = int_dist(rand);
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = BucketScore{1, 0};
      } else {
        hash_freq_and_scores[active_neuron].frequency += 1;
        hash_freq_and_scores[active_neuron].score += 0;
      }
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

  if (n_buckets_to_sample > n_hashes) {
    // If we are sampling more buckets then we end up using we rerank the
    // buckets based on size to load balance the index.
    std::sort(sorted_hashes.begin(),
              sorted_hashes.begin() + n_buckets_to_sample,
              [&mach_index, &cmp, approx_n_hashes_per_bucket, load_balancing](
                  const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index->bucketSize(lhs.first);
                size_t rhs_size = mach_index->bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.

                if (load_balancing) {
                  if (lhs_size < approx_n_hashes_per_bucket &&
                      rhs_size < approx_n_hashes_per_bucket) {
                    return cmp(lhs, rhs);
                  }
                }
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
  if (n_random_hashes > n_hashes) {
    throw std::invalid_argument(
        "n_random_hashes cannot be greater than num hashes.");
  }

  uint32_t n_informed_hashes =
      sort_random_hashes ? n_hashes : (n_hashes - n_random_hashes);

  for (uint32_t i = 0; i < n_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  if (!sort_random_hashes) {
    for (uint32_t i = 0; i < n_random_hashes; i++) {
      if (load_balancing) {
        uint32_t random_hash;

        do {
          random_hash = int_dist(rand);
        } while (mach_index->bucketSize(random_hash) >=
                 approx_n_hashes_per_bucket);

        new_hashes.push_back(random_hash);

      } else {
        new_hashes.push_back(int_dist(rand));
      }
    }
  }

  return new_hashes;
}

void MachRetriever::updateSamplingStrategy() {
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

bolt::metrics::InputMetrics MachRetriever::getMetrics(
    const std::vector<std::string>& metric_names, const std::string& prefix) {
  if (_model->outputs().size() != 1 || _model->labels().size() != 2 ||
      _model->losses().size() != 1) {
    throw std::invalid_argument(
        "Expected model to have single input, two labels, and one "
        "loss.");
  }

  bolt::ComputationPtr output = _model->outputs().front();
  bolt::ComputationPtr hash_labels = _model->labels().front();
  bolt::ComputationPtr true_class_labels = _model->labels().back();
  bolt::LossPtr loss = _model->losses().front();

  bolt::metrics::InputMetrics metrics;
  for (const auto& name : metric_names) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] = std::make_shared<bolt::metrics::MachPrecision>(
          index(), _n_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<bolt::metrics::MachRecall>(
          index(), _n_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("hash_precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 15, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<bolt::metrics::PrecisionAtK>(output, hash_labels, k);
    } else if (std::regex_match(name, std::regex("hash_recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 12, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<bolt::metrics::RecallAtK>(output, hash_labels, k);
    } else if (name == "loss") {
      metrics[prefix + name] =
          std::make_shared<bolt::metrics::LossMetric>(loss);
    } else {
      throw std::invalid_argument(
          "Invalid metric '" + name +
          "'. Please use precision@k, recall@k, or loss.");
    }
  }

  return metrics;
}

void MachRetriever::insertNewIds(const data::ColumnMapIteratorPtr& data) {
  data->restart();

  while (auto chunk = data->next()) {
    std::unordered_set<uint32_t> all_ids;

    auto ids = chunk->getArrayColumn<uint32_t>(_id_column);
    for (size_t i = 0; i < ids->numRows(); i++) {
      auto row = ids->row(i);
      all_ids.insert(row.begin(), row.end());
    }

    index()->insertNewEntities(all_ids);
  }

  data->restart();
}

ar::ConstArchivePtr MachRetriever::toArchive(bool with_optimizer) const {
  auto map = ar::Map::make();

  map->set("state", _state->toArchive());
  map->set("model", _model->toArchive(with_optimizer));

  map->set("text_column", ar::str(_text_column));
  map->set("id_column", ar::str(_id_column));

  map->set("text_transformation", _text_transform->toArchive());
  map->set("map_to_buckets", _map_to_buckets->toArchive());
  map->set("add_mach_memory_samples", _add_mach_memory_samples->toArchive());

  map->set("bolt_input_columns",
           data::outputColumnsToArchive(_bolt_input_columns));
  map->set("bolt_label_columns",
           data::outputColumnsToArchive(_bolt_label_columns));
  map->set("all_bolt_columns", ar::vecStr(_all_bolt_columns));

  map->set("mach_sampling_threshold", ar::f32(_mach_sampling_threshold));
  map->set("n_buckets_to_eval", ar::u64(_n_buckets_to_eval));

  return map;
}

MachRetriever::MachRetriever(const ar::Archive& archive)
    : _state(data::State::fromArchive(*archive.get("state"))),
      _model(bolt::Model::fromArchive(*archive.get("model"))),
      _text_column(archive.str("text_column")),
      _id_column(archive.str("id_column")),
      _text_transform(
          data::Transformation::fromArchive(*archive.get("text_transform"))),
      _map_to_buckets(
          data::Transformation::fromArchive(*archive.get("map_to_buckets"))),
      _add_mach_memory_samples(data::Transformation::fromArchive(
          *archive.get("add_mach_memory_samples"))),
      _bolt_input_columns(
          data::outputColumnsFromArchive(*archive.get("bolt_input_columns"))),
      _bolt_label_columns(
          data::outputColumnsFromArchive(*archive.get("bolt_label_columns"))),
      _all_bolt_columns(archive.getAs<ar::VecStr>("all_bolt_columns")),
      _mach_sampling_threshold(
          archive.getAs<ar::F32>("mach_sampling_threshold")),
      _n_buckets_to_eval(archive.u64("n_buckets_to_eval")) {}

std::shared_ptr<MachRetriever> MachRetriever::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<MachRetriever>(archive);
}

}  // namespace thirdai::automl::mach