#include "MachRetriever.h"
#include <bolt/python_bindings/CtrlCCheck.h>
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
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Pipeline.h>
#include <cassert>
#include <optional>
#include <regex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::automl::mach {

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
    const data::ColumnMapIteratorPtr& iter,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool phrase_sampling,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool load_balancing, bool sort_random_hashes) {
  while (auto columns = iter->next()) {
    introduce(std::move(*columns), strong_column_names, weak_column_names,
              phrase_sampling, num_buckets_to_sample_opt, num_random_hashes,
              load_balancing, sort_random_hashes);
  }
}

void MachRetriever::introduce(
    data::ColumnMap columns,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, bool phrase_sampling,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool load_balancing, bool sort_random_hashes) {
  assertUniqueIds(columns);

  // Sorry I'm changing the names a bit but I think this will help us
  // disambiguate overloaded terms in the long run.
  if (phrase_sampling) {
    // AKA coldstart transform
    phraseSampling(strong_column_names, weak_column_names, std::nullopt)
        ->apply(std::move(columns), *_state);
  } else {
    textConcat(strong_column_names, weak_column_names)
        ->apply(std::move(columns), *_state);
  }
  columns = _text_transform->apply(std::move(columns), *_state);
  auto input_tensors = inputTensors(columns);

  // Perform introduction

  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.

  bolt::python::CtrlCCheck ctrl_c_check;

  auto scores = _model->forward(input_tensors).at(0);

  ctrl_c_check();

  uint32_t num_buckets_to_sample =
      load_balancing ? index()->numBuckets()
                     : num_buckets_to_sample_opt.value_or(index()->numHashes());

  auto bucket_scores_by_doc = groupScoresByLabel(
      /* scores= */ *scores,
      /* labels= */ *columns.getValueColumn<uint32_t>(_id_column),
      /* only_keep_top_k= */
      load_balancing ? std::nullopt : std::make_optional(load_balancing));

  ctrl_c_check();

  uint32_t approx_num_hashes_per_bucket = index()->approxNumHashesPerBucket(
      /* num_new_samples= */ bucket_scores_by_doc.size());

  for (auto& [doc, bucket_scores] : bucket_scores_by_doc) {
    auto hashes =
        topHashesForDoc(std::move(bucket_scores), num_buckets_to_sample,
                        approx_num_hashes_per_bucket, num_random_hashes,
                        load_balancing, sort_random_hashes);
    index()->insert(doc, hashes);

    ctrl_c_check();
  }

  updateSamplingStrategy();
}

void MachRetriever::coldstart(
    data::ColumnMapIteratorPtr iter,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    data::ColumnMapIteratorPtr val_iter,
    const std::vector<std::string>& val_metrics, const TrainOptions& options,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const bolt::DistributedCommPtr& comm) {
  auto augmented_iter = data::TransformedIterator::make(
      std::move(iter),
      phraseSampling(strong_column_names, weak_column_names, variable_length),
      _state);
  train(augmented_iter, learning_rate, epochs, train_metrics,
        std::move(val_iter), val_metrics, options, callbacks, comm);
}

void MachRetriever::train(
    data::ColumnMapIteratorPtr iter, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    data::ColumnMapIteratorPtr val_iter,
    const std::vector<std::string>& val_metrics, const TrainOptions& options,
    const std::vector<bolt::callbacks::CallbackPtr>& callbacks,
    const bolt::DistributedCommPtr& comm) {
  // Text features
  // Label features
  // Add balancing samples to column maps
  // Extract balancing samples from pre-balancing samples map
  auto train_transform = data::Pipeline::make(
      {_text_transform, _id_transform, _add_mach_memory_samples});

  auto train_data_loader = data::Loader::make(
      std::move(iter), train_transform, _state, _bolt_input_columns,
      _bolt_label_columns, options.batch_size, /* shuffle= */ true,
      options.verbose, options.shuffle_config.min_buffer_size,
      options.shuffle_config.seed);

  data::LoaderPtr val_data_loader;
  if (val_iter) {
    auto val_transform = data::Pipeline::make({_text_transform, _id_transform});
    val_data_loader = data::Loader::make(
        std::move(val_iter), val_transform, _state, _bolt_input_columns,
        _bolt_label_columns, options.batch_size,
        /* shuffle= */ false, options.verbose);
  }

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (options.freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        /* gradient_update_interval */ 1,
                        bolt::python::CtrlCCheck{});

  auto history = trainer.train_with_data_loader(
      /* train_data_loader= */ train_data_loader,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */
      getMetrics(train_metrics, "train_"),
      /* validation_data_loader= */ val_data_loader,
      /* validation_metrics= */
      getMetrics(val_metrics, "val_"),
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks, /* autotune_rehash_rebuild= */ true,
      /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);
}

std::vector<IdScores> MachRetriever::search(data::ColumnMap queries,
                                            uint32_t top_k,
                                            bool sparse_inference) {
  auto in = inputTensors(_text_transform->apply(std::move(queries), *_state));
  auto out = _model->forward(in, sparse_inference).at(0);

  uint32_t num_queries = queries.numRows();
  std::vector<IdScores> predictions(num_queries);
#pragma omp parallel for default(none) \
    shared(out, predictions, k, num_queries) if (num_queries > 1)
  for (uint32_t i = 0; i < num_queries; i++) {
    const BoltVector& out_vec = out->getVector(i);
    predictions[i] = index()->decode(out_vec, top_k, _num_buckets_to_eval);
  }

  return predictions;
}

std::vector<IdScores> MachRetriever::rank(
    data::ColumnMap queries,
    const std::vector<std::unordered_set<uint32_t>>& choices,
    std::optional<uint32_t> top_k, bool sparse_inference) {
  assert(queries.numRows() == choices.size());

  auto in = inputTensors(_text_transform->apply(std::move(queries), *_state));
  auto out = _model->forward(in, sparse_inference).at(0);

  uint32_t num_queries = queries.numRows();
  std::vector<IdScores> predictions(num_queries);
#pragma omp parallel for default(none) \
    shared(out, predictions, k, num_queries) if (num_queries > 1)
  for (uint32_t i = 0; i < num_queries; i++) {
    const BoltVector& out_vec = out->getVector(i);
    predictions[i] = index()->scoreEntities(out_vec, choices[i], top_k);
  }

  return predictions;
}

std::vector<std::vector<uint32_t>> MachRetriever::predictBuckets(
    const data::ColumnMap& columns, bool sparse_inference,
    std::optional<uint32_t> top_k, bool force_non_empty) {
  auto outputs = _model->forward(inputTensors(columns), sparse_inference).at(0);

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

void MachRetriever::upvote(data::ColumnMap upvotes, uint32_t num_upvote_samples,
                           uint32_t num_balancing_samples, float learning_rate,
                           uint32_t epochs, size_t batch_size) {
  upvotes = data::Pipeline({_text_transform, _id_transform})
                .apply(std::move(upvotes), *_state);
  teach(upvotes, learning_rate, num_upvote_samples, num_balancing_samples,
        epochs, batch_size);
}

void MachRetriever::associate(data::ColumnMap from_columns,
                              const data::ColumnMap& to_columns,
                              uint32_t num_buckets,
                              uint32_t num_association_samples,
                              uint32_t num_balancing_samples,
                              float learning_rate, uint32_t epochs,
                              bool force_non_empty, size_t batch_size) {
  auto mach_labels = thirdai::data::ArrayColumn<uint32_t>::make(
      predictBuckets(to_columns, /* sparse_inference= */ false, num_buckets,
                     force_non_empty),
      index()->numBuckets());
  from_columns.setColumn(bucket_column, mach_labels);

  // Add dummy IDs since associations do not have IDs.
  auto doc_ids = thirdai::data::ValueColumn<uint32_t>::make(
      std::vector<uint32_t>(from_columns.numRows(), 0),
      std::numeric_limits<uint32_t>::max());
  from_columns.setColumn(_id_column, doc_ids);

  teach(from_columns, learning_rate, num_association_samples,
        num_balancing_samples, epochs, batch_size);
}

void MachRetriever::teach(data::ColumnMap feedback, float learning_rate,
                          uint32_t feedback_repetitions, uint32_t num_balancers,
                          uint32_t epochs, size_t batch_size) {
  auto balancers =
      _state->machMemory().getSamples(num_balancers * feedback.numRows());

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
    uint32_t num_buckets_to_sample, uint32_t approx_num_hashes_per_bucket,
    uint32_t num_random_hashes, bool load_balancing,
    bool sort_random_hashes) const {
  const auto& mach_index = _state->machIndex();

  uint32_t num_hashes = mach_index->numHashes();

  if (num_buckets_to_sample < mach_index->numHashes()) {
    throw std::invalid_argument(
        "Sampling from fewer buckets than num_hashes is not supported. If "
        "you'd like to introduce using fewer hashes, please reset the number "
        "of hashes for the index.");
  }

  if (num_buckets_to_sample > mach_index->numBuckets()) {
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

  uint32_t num_buckets = mach_index->numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  if (sort_random_hashes) {
    for (uint32_t i = 0; i < num_random_hashes; i++) {
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

  if (num_buckets_to_sample > num_hashes) {
    // If we are sampling more buckets then we end up using we rerank the
    // buckets based on size to load balance the index.
    std::sort(sorted_hashes.begin(),
              sorted_hashes.begin() + num_buckets_to_sample,
              [&mach_index, &cmp, approx_num_hashes_per_bucket, load_balancing](
                  const auto& lhs, const auto& rhs) {
                size_t lhs_size = mach_index->bucketSize(lhs.first);
                size_t rhs_size = mach_index->bucketSize(rhs.first);

                // Give preference to emptier buckets. If buckets are
                // equally empty, use one with the best score.

                if (load_balancing) {
                  if (lhs_size < approx_num_hashes_per_bucket &&
                      rhs_size < approx_num_hashes_per_bucket) {
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
  if (num_random_hashes > num_hashes) {
    throw std::invalid_argument(
        "num_random_hashes cannot be greater than num hashes.");
  }

  uint32_t num_informed_hashes =
      sort_random_hashes ? num_hashes : (num_hashes - num_random_hashes);

  for (uint32_t i = 0; i < num_informed_hashes; i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes.push_back(hash);
  }

  if (!sort_random_hashes) {
    for (uint32_t i = 0; i < num_random_hashes; i++) {
      if (load_balancing) {
        uint32_t random_hash;

        do {
          random_hash = int_dist(rand);
        } while (mach_index->bucketSize(random_hash) >=
                 approx_num_hashes_per_bucket);

        new_hashes.push_back(random_hash);

      } else {
        new_hashes.push_back(int_dist(rand));
      }
    }
  }

  return new_hashes;
}

float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},    {900, 0.2},    {1800, 0.1},     {4000, 0.05},
      {10000, 0.02}, {20000, 0.01}, {1000000, 0.005}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return sparsity_values.back().second;
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
          index(), _num_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<bolt::metrics::MachRecall>(
          index(), _num_buckets_to_eval, output, true_class_labels, k);
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

MachRetriever::MachRetriever(
    std::string text_column, const std::string& id_column, uint32_t num_hashes,
    uint32_t output_dim, uint32_t embedding_dim, uint32_t text_feature_dim,
    uint32_t output_bias, uint32_t embedding_bias, bool normalize_embeddings,
    const std::string& output_act_func, const std::string& embedding_act_func,
    const std::string& tokenizer, const std::string& contextual_encoding,
    bool lowercase, float mach_sampling_threshold, uint32_t num_buckets_to_eval,
    size_t memory_max_ids, size_t memory_max_samples_per_id)
    : _state(data::State::make(
          dataset::mach::MachIndex::make(output_dim, num_hashes),
          data::MachMemory::make(input_indices_column, input_values_column,
                                 id_column, bucket_column, memory_max_ids,
                                 memory_max_samples_per_id))),
      _model(defaultModel(text_feature_dim, embedding_dim, output_dim,
                          embedding_bias, output_bias, normalize_embeddings,
                          embedding_act_func, output_act_func)),
      _text_column(std::move(text_column)),
      _id_column(id_column),
      _text_transform(std::make_shared<data::TextTokenizer>(
          /* input_column= */ text_column,
          /* output_indices= */ input_indices_column,
          /* output_values= */ input_values_column,
          /* tokenizer= */ getTextTokenizerFromString(tokenizer),
          /* encoder= */ getTextEncoderFromString(contextual_encoding),
          /* lowercase= */ lowercase,
          /* dim= */ text_feature_dim)),
      _id_transform(
          std::make_shared<data::MachLabel>(id_column, bucket_column)),
      _add_mach_memory_samples(std::make_shared<data::AddMachMemorySamples>()),
      _bolt_input_columns({{input_indices_column, input_values_column}}),
      _bolt_label_columns(
          {data::OutputColumns(label_indices_column,
                               toValueFillType(output_act_func)),
           data::OutputColumns(id_column)}),
      _all_bolt_columns({input_indices_column, input_values_column,
                         label_indices_column, id_column}),
      _mach_sampling_threshold(mach_sampling_threshold),
      _num_buckets_to_eval(num_buckets_to_eval) {}
data::ValueFillType toValueFillType(const std::string& output_act_func) {
  if (text::lower(output_act_func) == "softmax") {
    return data::ValueFillType::SumToOne;
  }
  if (text::lower(output_act_func) == "sigmoid") {
    return data::ValueFillType::Ones;
  }
  throw std::invalid_argument("Invalid output_act_func \"" + output_act_func +
                              R"(". Choose one of "softmax" or "sigmoid".)");
}
bolt::ModelPtr defaultModel(uint32_t text_feature_dim, uint32_t embedding_dim,
                            uint32_t output_dim, bool embedding_bias,
                            bool output_bias, bool normalize_embeddings,
                            const std::string& embedding_act_func,
                            const std::string& output_act_func) {
  auto input = bolt::Input::make(text_feature_dim);

  auto hidden =
      bolt::Embedding::make(embedding_dim, text_feature_dim,
                            text::lower(embedding_act_func), embedding_bias)
          ->apply(input);

  if (normalize_embeddings) {
    hidden = bolt::LayerNorm::make()->apply(hidden);
  }

  auto sparsity = autotuneSparsity(output_dim);
  auto output_act_func_lower = text::lower(output_act_func);
  auto output = bolt::FullyConnected::make(output_dim, hidden->dim(), sparsity,
                                           output_act_func_lower,
                                           /* sampling= */ nullptr,
                                           /* use_bias= */ output_bias)
                    ->apply(hidden);

  auto labels = bolt::Input::make(output_dim);

  bolt::LossPtr loss;
  if (output_act_func_lower == "sigmoid") {
    loss = bolt::BinaryCrossEntropy::make(output, labels);
  } else if (output_act_func_lower == "softmax") {
    loss = bolt::CategoricalCrossEntropy::make(output, labels);
  } else {
    throw std::invalid_argument("Invalid output_act_func \"" + output_act_func +
                                R"(". Choose one of "softmax" or "sigmoid".)");
  }

  return bolt::Model::make(
      {input}, {output}, {loss},
      // We need the hash based labels for training, but the actual
      // document/class ids to compute metrics. Hence we add two labels to the
      // model.
      {bolt::Input::make(std::numeric_limits<uint32_t>::max())});
}
}  // namespace thirdai::automl::mach