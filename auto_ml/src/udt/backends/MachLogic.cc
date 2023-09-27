#include "MachLogic.h"
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/MachNeuronIndex.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/MachPrecision.h>
#include <bolt/src/train/metrics/MachRecall.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/MachLogicStripped.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/State.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachBlock.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <algorithm>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl::udt {

using bolt::metrics::LossMetric;
using bolt::metrics::MachPrecision;
using bolt::metrics::MachRecall;
using bolt::metrics::PrecisionAtK;
using bolt::metrics::RecallAtK;

void ClassifierForMach::updateSamplingStrategy(thirdai::data::State& state) {
  const auto& mach_index = state.machIndex();

  auto output_layer = bolt::FullyConnected::cast(
      _classifier->model()->opExecutionOrder().back());

  const auto& neuron_index = output_layer->kernel()->neuronIndex();

  float index_sparsity = mach_index->sparsity();

  if (index_sparsity > 0 && index_sparsity <= _mach_sampling_threshold) {
    // TODO(Nicholas) add option to specify new neuron index in set sparsity.
    output_layer->setSparsity(index_sparsity, false, false);
    auto new_index = bolt::MachNeuronIndex::make(mach_index);
    output_layer->kernel()->setNeuronIndex(new_index);

  } else {
    if (std::dynamic_pointer_cast<bolt::MachNeuronIndex>(neuron_index)) {
      float sparsity = utils::autotuneSparsity(mach_index->numBuckets());

      auto sampling_config = bolt::DWTASamplingConfig::autotune(
          mach_index->numBuckets(), sparsity,
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

MachLogic::MachLogic(const data::ColumnDataTypes& input_data_types,
                     const data::UserProvidedTemporalRelationships&
                         temporal_tracking_relationships,
                     const std::string& target_name, bool integer_target,
                     const bolt::ModelPtr& model, bool freeze_hash_tables,
                     uint32_t num_buckets, float mach_sampling_threshold,
                     bool rlhf, uint32_t num_balancing_docs,
                     uint32_t num_balancing_samples_per_doc,
                     const data::TabularOptions& tabular_options)
    : _classifier(utils::Classifier::make(model, freeze_hash_tables),
                  mach_sampling_threshold),
      _default_top_k_to_return(defaults::MACH_TOP_K_TO_RETURN),
      _num_buckets_to_eval(defaults::MACH_NUM_BUCKETS_TO_EVAL) {
  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?

  if (!integer_target) {
    throw std::invalid_argument(
        "Option 'integer_target=True' must be specified when using extreme "
        "classification options.");
  }

  auto temporal_relationships = data::TemporalRelationshipsAutotuner::autotune(
      input_data_types, temporal_tracking_relationships,
      tabular_options.lookahead);

  _featurizer = std::make_shared<MachFeaturizer>(
      input_data_types, temporal_relationships, target_name, num_buckets,
      tabular_options);

  if (rlhf) {
    _rlhf_sampler = std::make_optional<RLHFSampler>(
        num_balancing_docs, num_balancing_samples_per_doc);
  }
}

py::object MachLogic::train(const dataset::DataSourcePtr& data,
                            thirdai::data::StatePtr& state, float learning_rate,
                            uint32_t epochs,
                            const std::vector<std::string>& train_metrics,
                            const dataset::DataSourcePtr& val_data,
                            const std::vector<std::string>& val_metrics,
                            const std::vector<CallbackPtr>& callbacks,
                            TrainOptions options,
                            const bolt::DistributedCommPtr& comm) {
  addBalancingSamples(data, *state);

  auto train_data_loader = _featurizer->getDataLoader(
      data, state, options.batchSize(), /* shuffle= */ true, options.verbose,
      options.shuffle_config);

  thirdai::data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        _featurizer->getDataLoader(val_data, state, defaults::BATCH_SIZE,
                                   /* shuffle= */ false, options.verbose);
  }

  return _classifier.classifier(*state)->train(
      train_data_loader, learning_rate, epochs,
      getMetrics(train_metrics, "train_", *state), val_data_loader,
      getMetrics(val_metrics, "val_", *state), callbacks, options, comm);
}

py::object MachLogic::trainBatch(const MapInputBatch& batch,
                                 thirdai::data::State& state,
                                 float learning_rate) {
  auto& model = _classifier.classifier(state)->model();

  auto [inputs, labels] = _featurizer->featurizeTrainingBatch(batch, state);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  return py::none();
}

py::object MachLogic::trainWithHashes(const MapInputBatch& batch,
                                      thirdai::data::State& state,
                                      float learning_rate) {
  auto& model = _classifier.classifier(state)->model();

  auto [inputs, labels] =
      _featurizer->featurizeHashesTrainingBatch(batch, state);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  return py::none();
}

py::object MachLogic::evaluate(const dataset::DataSourcePtr& data,
                               thirdai::data::StatePtr& state,
                               const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose) {
  auto data_loader =
      _featurizer->getDataLoader(data, state, defaults::BATCH_SIZE,
                                 /* shuffle= */ false, verbose);

  return _classifier.classifier(*state)->evaluate(
      data_loader, getMetrics(metrics, "val_", *state), sparse_inference,
      verbose);
}

py::object MachLogic::predict(const MapInput& sample,
                              thirdai::data::State& state,
                              bool sparse_inference,
                              std::optional<uint32_t> top_k) {
  return py::cast(predictImpl({sample}, state, sparse_inference, top_k).at(0));
}

std::vector<std::vector<std::pair<uint32_t, double>>> MachLogic::predictImpl(
    const MapInputBatch& samples, thirdai::data::State& state,
    bool sparse_inference, std::optional<uint32_t> top_k) {
  auto outputs = _classifier.classifier(state)
                     ->model()
                     ->forward(_featurizer->featurizeInputBatch(samples, state),
                               sparse_inference)
                     .at(0);

  return thirdai::automl::mach::rankedEntitiesFromOutputs(
      *outputs, *state.machIndex(), top_k.value_or(_default_top_k_to_return),
      _num_buckets_to_eval);
}

py::object MachLogic::predictBatch(const MapInputBatch& samples,
                                   thirdai::data::State& state,
                                   bool sparse_inference,
                                   std::optional<uint32_t> top_k) {
  return py::cast(predictImpl(samples, state, sparse_inference, top_k));
}

py::object MachLogic::predictHashes(const MapInput& sample,
                                    thirdai::data::State& state,
                                    bool sparse_inference, bool force_non_empty,
                                    std::optional<uint32_t> num_hashes) {
  return py::cast(predictHashesImpl({sample}, state, sparse_inference,
                                    force_non_empty, num_hashes)
                      .at(0));
}

py::object MachLogic::predictHashesBatch(const MapInputBatch& samples,
                                         thirdai::data::State& state,
                                         bool sparse_inference,
                                         bool force_non_empty,
                                         std::optional<uint32_t> num_hashes) {
  return py::cast(predictHashesImpl(samples, state, sparse_inference,
                                    force_non_empty, num_hashes));
}

std::vector<std::vector<uint32_t>> MachLogic::predictHashesImpl(
    const MapInputBatch& samples, thirdai::data::State& state,
    bool sparse_inference, bool force_non_empty,
    std::optional<uint32_t> num_hashes) {
  auto outputs = _classifier.classifier(state)
                     ->model()
                     ->forward(_featurizer->featurizeInputBatch(samples, state),
                               sparse_inference)
                     .at(0);

  return mach::rankedBucketsFromOutputs(*outputs, *state.machIndex(),
                                        force_non_empty, num_hashes);
}

py::object MachLogic::outputCorrectness(const MapInputBatch& samples,
                                        thirdai::data::State& state,
                                        const std::vector<uint32_t>& labels,
                                        bool sparse_inference,
                                        std::optional<uint32_t> num_hashes) {
  std::vector<std::vector<uint32_t>> top_buckets =
      predictHashesImpl(samples, state, sparse_inference,
                        /* force_non_empty = */ true, num_hashes);

  std::vector<uint32_t> matching_buckets(labels.size());
  std::exception_ptr hashes_err;

#pragma omp parallel for default(none) \
    shared(labels, top_buckets, matching_buckets, hashes_err, state)
  for (uint32_t i = 0; i < labels.size(); i++) {
    try {
      std::vector<uint32_t> hashes = state.machIndex()->getHashes(labels[i]);
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

  return py::cast(matching_buckets);
}

void MachLogic::setModel(const ModelPtr& model) { _classifier.setModel(model); }

py::object MachLogic::coldstart(
    const dataset::DataSourcePtr& data, thirdai::data::StatePtr& state,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  addBalancingSamples(data, *state, strong_column_names, weak_column_names);

  // TODO(MACHREFACTOR): This is almost exactly the same as training. Can we
  // figure out how to abstract out succinct and coherent computational units?
  auto train_data_loader = _featurizer->getColdStartDataLoader(
      data, state, strong_column_names, weak_column_names,
      /* fast_approximation= */ false, options.batchSize(),
      /* shuffle= */ true, options.verbose, options.shuffle_config);

  thirdai::data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        _featurizer->getDataLoader(val_data, state, defaults::BATCH_SIZE,
                                   /* shuffle= */ false, options.verbose);
  }

  return _classifier.classifier(*state)->train(
      train_data_loader, learning_rate, epochs,
      getMetrics(train_metrics, "train_", *state), val_data_loader,
      getMetrics(val_metrics, "val_", *state), callbacks, options, comm);
}

py::object MachLogic::embedding(const MapInputBatch& sample,
                                thirdai::data::State& state) {
  return _classifier.classifier(state)->embedding(
      _featurizer->featurizeInputBatch(sample, state));
}

uint32_t expectInteger(const Label& label) {
  if (!std::holds_alternative<uint32_t>(label)) {
    throw std::invalid_argument("Must use integer label.");
  }
  return std::get<uint32_t>(label);
}

py::object MachLogic::entityEmbedding(const Label& label,
                                      thirdai::data::State& state) {
  auto outputs = _classifier.classifier(state)->model()->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto fc_layer = fc->kernel();

  auto averaged_embedding = mach::averageBucketEmbeddings(
      expectInteger(label), *state.machIndex(), *fc_layer);

  NumpyArray<float> np_weights(averaged_embedding.size());

  std::copy(averaged_embedding.begin(), averaged_embedding.end(),
            np_weights.mutable_data());

  return std::move(np_weights);
}

void MachLogic::introduceDocuments(
    const dataset::DataSourcePtr& data, thirdai::data::State& state,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes, bool fast_approximation) {
  const auto& mach_index = state.machIndex();

  auto data_and_doc_ids = _featurizer->featurizeForIntroduceDocuments(
      data, state, strong_column_names, weak_column_names, fast_approximation,
      defaults::BATCH_SIZE);

  uint32_t num_buckets_to_sample =
      num_buckets_to_sample_opt.value_or(mach_index->numHashes());

  std::unordered_map<uint32_t, std::vector<TopKActivationsQueue>> top_k_per_doc;

  bolt::python::CtrlCCheck ctrl_c_check;

  for (const auto& [input, doc_ids] : data_and_doc_ids) {
    // Note: using sparse inference here could cause issues because the
    // mach index sampler will only return nonempty buckets, which could
    // cause new docs to only be mapped to buckets already containing
    // entities.
    auto scores = _classifier.classifier(state)->model()->forward(input).at(0);

    for (uint32_t i = 0; i < scores->batchSize(); i++) {
      uint32_t label = doc_ids[i];
      top_k_per_doc[label].push_back(
          scores->getVector(i).findKLargestActivations(num_buckets_to_sample));
    }

    ctrl_c_check();
  }

  for (auto& [doc, top_ks] : top_k_per_doc) {
    auto hashes = topHashesForDoc(std::move(top_ks), state,
                                  num_buckets_to_sample, num_random_hashes);
    mach_index->insert(doc, hashes);

    ctrl_c_check();
  }

  addBalancingSamples(data, state, strong_column_names, weak_column_names);
}

void MachLogic::introduceDocument(
    const MapInput& document, thirdai::data::State& state,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label,
    std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes) {
  auto samples = _featurizer->featurizeInputColdStart(
      document, state, strong_column_names, weak_column_names);

  introduceLabelHelper(samples, state, new_label, num_buckets_to_sample,
                       num_random_hashes);
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

// TODO(MACHREFACTOR): why is this soooo long T-T What's going on here?
std::vector<uint32_t> MachLogic::topHashesForDoc(
    std::vector<TopKActivationsQueue>&& top_k_per_sample,
    thirdai::data::State& state, uint32_t num_buckets_to_sample,
    uint32_t num_random_hashes) {
  const auto& mach_index = state.machIndex();

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
                size_t lhs_size = mach_index->bucketSize(lhs.first);
                size_t rhs_size = mach_index->bucketSize(rhs.first);

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

  uint32_t num_buckets = mach_index->numBuckets();
  std::uniform_int_distribution<uint32_t> int_dist(0, num_buckets - 1);
  std::mt19937 rand(global_random::nextSeed());

  for (uint32_t i = 0; i < num_random_hashes; i++) {
    new_hashes.push_back(int_dist(rand));
  }

  return new_hashes;
}

void MachLogic::introduceLabel(
    const MapInputBatch& samples, thirdai::data::State& state,
    const Label& new_label, std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  introduceLabelHelper(_featurizer->featurizeInputBatch(samples, state), state,
                       new_label, num_buckets_to_sample_opt, num_random_hashes);
}

void MachLogic::introduceLabelHelper(
    const bolt::TensorList& samples, thirdai::data::State& state,
    const Label& new_label, std::optional<uint32_t> num_buckets_to_sample_opt,
    uint32_t num_random_hashes) {
  // Note: using sparse inference here could cause issues because the
  // mach index sampler will only return nonempty buckets, which could
  // cause new docs to only be mapped to buckets already containing
  // entities.
  auto output = _classifier.classifier(state)
                    ->model()
                    ->forward(samples, /* use_sparsity = */ false)
                    .at(0);

  uint32_t num_buckets_to_sample =
      num_buckets_to_sample_opt.value_or(state.machIndex()->numHashes());

  mach::addEntityToIndex(num_buckets_to_sample, num_random_hashes, *output,
                         *state.machIndex(), expectInteger(new_label));
}

void MachLogic::forget(const Label& label, thirdai::data::State& state) {
  state.machIndex()->erase(expectInteger(label));

  if (state.machIndex()->numEntities() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }
}

// TODO(MACHREFACTOR): Take in iterator instead of data source
void MachLogic::addBalancingSamples(
    const dataset::DataSourcePtr& data, thirdai::data::State& state,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  if (_rlhf_sampler) {
    data->restart();

    // TODO(Geordie / Nick) Right now, we only load MAX_BALANCING_SAMPLES
    // samples to avoid the overhead of loading the entire dataset. It's
    // possible this won't load enough samples to cover all classes.
    // We may try to keep streaming data until all classes are covered or load
    // the entire dataset and see if it makes a difference. For now we just
    // sample from 5x more rows than we need samples, to hopefully get a wider
    // range of samples.
    auto samples = _featurizer->getBalancingSamples(
        data, state, strong_column_names, weak_column_names,
        defaults::MAX_BALANCING_SAMPLES, defaults::MAX_BALANCING_SAMPLES * 5);
    // TODO(MACHREFACTOR): This can be a transformation. CONT
    // May need to put rlhf sampler in state.
    for (const auto& [doc_id, rlhf_sample] : samples) {
      _rlhf_sampler->addSample(doc_id, rlhf_sample);
    }
    // TODO(MACHREFACTOR): This can be a transformation. END
    data->restart();
  }
}

void MachLogic::requireRLHFSampler() {
  if (!_rlhf_sampler) {
    throw std::runtime_error(
        "This model was not configured to support rlhf. Please pass "
        "{'rlhf': "
        "True} in the model options or call enable_rlhf().");
  }
}

// TODO(MACHREFACTOR): Take in columns / column map
void MachLogic::associate(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    thirdai::data::State& state, uint32_t n_buckets,
    uint32_t n_association_samples, uint32_t n_balancing_samples,
    float learning_rate, uint32_t epochs) {
  auto teaching_samples = getAssociateSamples(rlhf_samples, state, n_buckets,
                                              n_association_samples);

  teach(teaching_samples, state, rlhf_samples.size() * n_balancing_samples,
        learning_rate, epochs);
}

// TODO(MACHREFACTOR): Take in columns / column map, then apply transformation
void MachLogic::upvote(
    const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
    thirdai::data::State& state, uint32_t n_upvote_samples,
    uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
  std::vector<RlhfSample> teaching_samples;

  teaching_samples.reserve(rlhf_samples.size() * n_upvote_samples);
  for (const auto& [source, target] : rlhf_samples) {
    RlhfSample sample = {source, state.machIndex()->getHashes(target)};

    for (size_t i = 0; i < n_upvote_samples; i++) {
      teaching_samples.push_back(sample);
    }
  }

  teach(teaching_samples, state, rlhf_samples.size() * n_balancing_samples,
        learning_rate, epochs);
}

void MachLogic::teach(const std::vector<RlhfSample>& rlhf_samples,
                      thirdai::data::State& state, uint32_t n_balancing_samples,
                      float learning_rate, uint32_t epochs) {
  requireRLHFSampler();

  auto samples = _rlhf_sampler->balancingSamples(n_balancing_samples);

  samples.insert(samples.end(), rlhf_samples.begin(), rlhf_samples.end());

  auto columns = _featurizer->featurizeRlhfSamples(samples, state);
  columns.shuffle();
  auto [data, labels] =
      _featurizer->columnsToTensors(columns, defaults::ASSOCIATE_BATCH_SIZE);

  for (uint32_t e = 0; e < epochs; e++) {
    for (size_t i = 0; i < data.size(); i++) {
      _classifier.classifier(state)->model()->trainOnBatch(data.at(i),
                                                           labels.at(i));
      _classifier.classifier(state)->model()->updateParameters(learning_rate);
    }
  }
}

std::vector<RlhfSample> MachLogic::getAssociateSamples(
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    thirdai::data::State& state, size_t n_buckets,
    size_t n_association_samples) {
  std::string text_column = _featurizer->textDatasetConfig().textColumn();
  MapInputBatch batch;
  for (const auto& [_, target] : rlhf_samples) {
    batch.push_back({{text_column, target}});
  }

  auto all_predicted_hashes =
      predictHashesImpl(batch, state, /* sparse_inference = */ false,
                        /* force_non_empty = */ true);

  std::mt19937 rng(global_random::nextSeed());

  // TODO(MACHREFACTOR): Make column map? What if we allowed mach model as
  // state? :0 And what if State was more of an ephemeral wrapper?

  std::vector<RlhfSample> associate_samples;
  associate_samples.reserve(rlhf_samples.size() * n_association_samples);

  for (size_t i = 0; i < rlhf_samples.size(); i++) {
    for (size_t j = 0; j < n_association_samples; j++) {
      const auto& all_buckets = rlhf_samples[i].second;
      std::vector<uint32_t> sampled_buckets;
      std::sample(all_buckets.begin(), all_buckets.end(),
                  std::back_inserter(sampled_buckets), n_buckets, rng);

      associate_samples.emplace_back(rlhf_samples[i].first,
                                     all_predicted_hashes[i]);
    }
  }

  return associate_samples;
}

py::object MachLogic::associateTrain(
    const dataset::DataSourcePtr& balancing_data, thirdai::data::State& state,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  return associateColdStart(balancing_data, state, {}, {}, rlhf_samples,
                            n_buckets, n_association_samples, learning_rate,
                            epochs, metrics, options);
}

py::object MachLogic::associateColdStart(
    const dataset::DataSourcePtr& balancing_data, thirdai::data::State& state,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
    uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    TrainOptions options) {
  // TODO(nicholas): make sure max_in_memory_batches is none

  auto featurized_data = _featurizer->featurizeDataset(
      balancing_data, state, strong_column_names, weak_column_names);

  auto associate_samples = getAssociateSamples(rlhf_samples, state, n_buckets,
                                               n_association_samples);

  auto featurized_rlhf_data =
      _featurizer->featurizeRlhfSamples(associate_samples, state);

  auto columns = featurized_data.concat(featurized_rlhf_data);
  columns.shuffle();

  auto dataset = _featurizer->columnsToTensors(columns, options.batchSize());

  bolt::Trainer trainer(_classifier.classifier(state)->model());

  auto output_metrics =
      trainer.train(/* train_data= */ dataset,
                    /* learning_rate= */ learning_rate, /* epochs= */ epochs,
                    /* train_metrics= */ getMetrics(metrics, "train_", state),
                    /* validation_data= */ {},
                    /* validation_metrics= */ {},
                    /* steps_per_validation= */ {},
                    /* use_sparsity_in_validation= */ false,
                    /* callbacks= */ {},
                    /* autotune_rehash_rebuild= */ true,
                    /* verbose= */ options.verbose,
                    /* logging_interval= */ options.logging_interval);

  return py::cast(output_metrics);
}

void MachLogic::setDecodeParams(uint32_t top_k_to_return,
                                thirdai::data::State& state,
                                uint32_t num_buckets_to_eval) {
  if (top_k_to_return == 0 || num_buckets_to_eval == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t num_buckets = state.machIndex()->numBuckets();
  if (num_buckets_to_eval > num_buckets) {
    throw std::invalid_argument(
        "Cannot eval with num_buckets_to_eval greater than " +
        std::to_string(num_buckets) + ".");
  }

  uint32_t num_classes = state.machIndex()->numEntities();
  if (top_k_to_return > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to "
        "predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  _default_top_k_to_return = top_k_to_return;
  _num_buckets_to_eval = num_buckets_to_eval;
}

void MachLogic::setMachSamplingThreshold(float threshold) {
  _classifier.setMachSamplingThreshold(threshold);
}

InputMetrics MachLogic::getMetrics(const std::vector<std::string>& metric_names,

                                   const std::string& prefix,
                                   thirdai::data::State& state) {
  const auto& model = _classifier.classifier(state)->model();
  if (model->outputs().size() != 1 || model->labels().size() != 2 ||
      model->losses().size() != 1) {
    throw std::invalid_argument(
        "Expected model to have single input, two labels, and one "
        "loss.");
  }

  bolt::ComputationPtr output = model->outputs().front();
  bolt::ComputationPtr hash_labels = model->labels().front();
  bolt::ComputationPtr true_class_labels = model->labels().back();
  bolt::LossPtr loss = model->losses().front();

  auto mach_index = state.machIndex();

  InputMetrics metrics;
  for (const auto& name : metric_names) {
    if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachPrecision>(
          mach_index, _num_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<MachRecall>(
          mach_index, _num_buckets_to_eval, output, true_class_labels, k);
    } else if (std::regex_match(name, std::regex("hash_precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 15, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<PrecisionAtK>(output, hash_labels, k);
    } else if (std::regex_match(name, std::regex("hash_recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 12, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<RecallAtK>(output, hash_labels, k);
    } else if (name == "loss") {
      metrics[prefix + name] = std::make_shared<LossMetric>(loss);
    } else {
      throw std::invalid_argument(
          "Invalid metric '" + name +
          "'. Please use precision@k, recall@k, or loss.");
    }
  }

  return metrics;
}

bolt::TensorPtr MachLogic::placeholderDocIds(uint32_t batch_size) {
  return bolt::Tensor::sparse(batch_size, std::numeric_limits<uint32_t>::max(),
                              /* nonzeros= */ 1);
}

template void MachLogic::serialize(cereal::BinaryInputArchive&);
template void MachLogic::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachLogic::serialize(Archive& archive) {
  archive(_classifier, _featurizer, _default_top_k_to_return,
          _num_buckets_to_eval, _rlhf_sampler);
}

}  // namespace thirdai::automl::udt