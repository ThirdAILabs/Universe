#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/MachFeaturizer.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <data/src/transformations/State.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <pybind11/pytypes.h>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::udt {

using bolt::metrics::InputMetrics;

using Label = std::variant<uint32_t, std::string>;

class ClassifierForMach {
 public:
  explicit ClassifierForMach(std::shared_ptr<utils::Classifier> classifier,
                             uint32_t mach_sampling_threshold)
      : _classifier(std::move(classifier)),
        _mach_sampling_threshold(mach_sampling_threshold) {}

  ClassifierForMach() {}

  auto classifier() const { return _classifier; }

  void updateSamplingStrategy(thirdai::data::State& state);

  void setMachSamplingThreshold(uint32_t threshold) {
    _mach_sampling_threshold = threshold;
  }

 private:
  std::shared_ptr<utils::Classifier> _classifier;
  uint32_t _mach_sampling_threshold;

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(_classifier, _mach_sampling_threshold);
  }
};

class MachLogic {
 public:
  MachLogic(const data::ColumnDataTypes& input_data_types,
            const data::UserProvidedTemporalRelationships&
                temporal_tracking_relationships,
            const std::string& target_name, uint32_t n_target_classes,
            bool integer_target, const bolt::ModelPtr& model,
            bool freeze_hash_tables, uint32_t num_buckets, uint32_t num_hashes,
            uint32_t mach_sampling_threshold, bool rlhf,
            uint32_t num_balancing_docs, uint32_t num_balancing_samples_per_doc,
            const data::TabularOptions& tabular_options);

  py::object train(const dataset::DataSourcePtr& data,
                   thirdai::data::StatePtr& state, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options, const bolt::DistributedCommPtr& comm);

  py::object trainBatch(const MapInputBatch& batch, thirdai::data::State& state,
                        float learning_rate);

  py::object trainWithHashes(const MapInputBatch& batch,
                             thirdai::data::State& state, float learning_rate);

  py::object evaluate(const dataset::DataSourcePtr& data,
                      thirdai::data::StatePtr& state,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose);

  py::object predict(const MapInput& sample, thirdai::data::State& state,
                     bool sparse_inference, std::optional<uint32_t> top_k);

  py::object predictBatch(const MapInputBatch& samples,
                          thirdai::data::State& state, bool sparse_inference,
                          std::optional<uint32_t> top_k);

  py::object predictHashes(const MapInput& sample, thirdai::data::State& state,
                           bool sparse_inference, bool force_non_empty,
                           std::optional<uint32_t> num_hashes);

  py::object predictHashesBatch(const MapInputBatch& samples,
                                thirdai::data::State& state,
                                bool sparse_inference, bool force_non_empty,
                                std::optional<uint32_t> num_hashes);

  py::object outputCorrectness(const MapInputBatch& samples,
                               thirdai::data::State& state,
                               const std::vector<uint32_t>& labels,
                               bool sparse_inference,
                               std::optional<uint32_t> num_hashes);

  ModelPtr model() const { return _classifier.classifier()->model(); }

  void setModel(const ModelPtr& model);

  void updateTemporalTrackers(const MapInput& sample,
                              thirdai::data::State& state) {
    _featurizer->updateTemporalTrackers(sample, state);
  }

  void updateTemporalTrackersBatch(const MapInputBatch& samples,
                                   thirdai::data::State& state) {
    _featurizer->updateTemporalTrackersBatch(samples, state);
  }

  void resetTemporalTrackers(thirdai::data::State& state) {
    _featurizer->resetTemporalTrackers(state);
  }

  const TextDatasetConfig& textDatasetConfig() const {
    return _featurizer->textDatasetConfig();
  }

  py::object coldstart(const dataset::DataSourcePtr& data,
                       thirdai::data::StatePtr& state,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& train_metrics,
                       const dataset::DataSourcePtr& val_data,
                       const std::vector<std::string>& val_metrics,
                       const std::vector<CallbackPtr>& callbacks,
                       TrainOptions options,
                       const bolt::DistributedCommPtr& comm);

  py::object embedding(const MapInputBatch& sample,
                       thirdai::data::State& state);

  /**
   * This method is still experimental, we should test to see when these
   * embeddings are useful and which tweaks like summing vs averaging and tanh
   * vs reul make a difference.
   */
  py::object entityEmbedding(const Label& label, thirdai::data::State& state);

  void introduceDocuments(const dataset::DataSourcePtr& data,
                          thirdai::data::State& state,
                          const std::vector<std::string>& strong_column_names,
                          const std::vector<std::string>& weak_column_names,
                          std::optional<uint32_t> num_buckets_to_sample,
                          uint32_t num_random_hashes, bool fast_approximation);

  void introduceDocument(const MapInput& document, thirdai::data::State& state,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         const Label& new_label,
                         std::optional<uint32_t> num_buckets_to_sample,
                         uint32_t num_random_hashes);

  void introduceLabel(const MapInputBatch& samples, thirdai::data::State& state,
                      const Label& new_label,
                      std::optional<uint32_t> num_buckets_to_sample,
                      uint32_t num_random_hashes);

  static void forget(const Label& label, thirdai::data::State& state);

  void associate(
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      thirdai::data::State& state, uint32_t n_buckets,
      uint32_t n_association_samples, uint32_t n_balancing_samples,
      float learning_rate, uint32_t epochs);

  void upvote(const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
              thirdai::data::State& state, uint32_t n_upvote_samples,
              uint32_t n_balancing_samples, float learning_rate,
              uint32_t epochs);

  py::object associateTrain(
      const dataset::DataSourcePtr& balancing_data, thirdai::data::State& state,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options);

  py::object associateColdStart(
      const dataset::DataSourcePtr& balancing_data, thirdai::data::State& state,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options);

  void setDecodeParams(uint32_t top_k_to_return, thirdai::data::State& state,
                       uint32_t num_buckets_to_eval);

  void verifyCanDistribute() const {
    if (_featurizer->hasTemporalTransformations()) {
      throw std::invalid_argument(
          "UDT with temporal relationships cannot be trained in a distributed "
          "setting.");
    }
  }

  void setMachSamplingThreshold(float threshold);

  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc) {
    if (_rlhf_sampler.has_value()) {
      std::cout << "rlhf already enabled." << std::endl;
      return;
    }

    _rlhf_sampler = std::make_optional<RLHFSampler>(
        num_balancing_docs, num_balancing_samples_per_doc);
  }

  void clearIndex(thirdai::data::State& state) {
    state.machIndex()->clear();

    if (_rlhf_sampler) {
      _rlhf_sampler->clear();
    }
  }

  MachLogic() {}

 private:
  std::vector<std::vector<std::pair<uint32_t, double>>> predictImpl(
      const MapInputBatch& samples, thirdai::data::State& state,
      bool sparse_inference, std::optional<uint32_t> top_k);

  std::vector<std::vector<uint32_t>> predictHashesImpl(
      const MapInputBatch& samples, thirdai::data::State& state,
      bool sparse_inference, bool force_non_empty = true,
      std::optional<uint32_t> num_hashes = std::nullopt);

  void introduceLabelHelper(const bolt::TensorList& samples,
                            thirdai::data::State& state, const Label& new_label,
                            std::optional<uint32_t> num_buckets_to_sample_opt,
                            uint32_t num_random_hashes);

  void teach(const std::vector<RlhfSample>& rlhf_samples,
             thirdai::data::State& state, uint32_t n_balancing_samples,
             float learning_rate, uint32_t epochs);

  std::vector<RlhfSample> getAssociateSamples(
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      thirdai::data::State& state, size_t n_buckets,
      size_t n_association_samples);

  void updateSamplingStrategy(thirdai::data::State& state);

  void addBalancingSamples(
      const dataset::DataSourcePtr& data, thirdai::data::State& state,
      const std::vector<std::string>& strong_column_names = {},
      const std::vector<std::string>& weak_column_names = {});

  void requireRLHFSampler();

  static std::vector<uint32_t> topHashesForDoc(
      std::vector<TopKActivationsQueue>&& top_k_per_sample,
      thirdai::data::State& state, uint32_t num_buckets_to_sample,
      uint32_t num_random_hashes = 0);

  InputMetrics getMetrics(const std::vector<std::string>& metric_names,
                          const std::string& prefix,
                          thirdai::data::State& state);

  // Mach requires two sets of labels. The buckets for each doc/class for
  // computing losses when training, and also the original doc/class ids for
  // computing metrics. In some methods like trainWithHashes, or trainOnBatch we
  // don't have/need the doc/class ids for metrics so we use this method to get
  // an empty placeholder to pass to the model.
  static bolt::TensorPtr placeholderDocIds(uint32_t batch_size);

  static uint32_t autotuneMachOutputDim(uint32_t n_target_classes) {
    // TODO(david) update this
    if (n_target_classes < defaults::MACH_MIN_TARGET_CLASSES) {
      return n_target_classes;
    }
    return n_target_classes / defaults::MACH_DEFAULT_OUTPUT_RANGE_SCALEDOWN;
  }

  static uint32_t autotuneMachNumHashes(uint32_t n_target_classes,
                                        uint32_t output_range) {
    // TODO(david) update this
    (void)n_target_classes;
    (void)output_range;
    return defaults::MACH_DEFAULT_NUM_REPETITIONS;
  }

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  ClassifierForMach _classifier;

  MachFeaturizerPtr _featurizer;

  uint32_t _default_top_k_to_return;
  uint32_t _num_buckets_to_eval;

  std::optional<RLHFSampler> _rlhf_sampler;
};

}  // namespace thirdai::automl::udt