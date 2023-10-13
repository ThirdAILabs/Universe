#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/featurization/UDTTransformationFactory.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Mach.h>
#include <data/src/ColumnMap.h>
#include <data/src/ColumnMapIterator.h>
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

class UDTMachClassifier final : public UDTBackend {
 public:
  UDTMachClassifier(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, const CategoricalDataTypePtr& target,
      uint32_t n_target_classes, bool integer_target,
      const TabularOptions& tabular_options,
      const std::optional<std::string>& model_config,
      config::ArgumentMap user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::DistributedCommPtr& comm) final;

  py::object trainBatch(const MapInputBatch& batch, float learning_rate,
                        const std::vector<std::string>& metrics) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  void updateTemporalTrackers(const MapInput& sample) final {
    auto table = thirdai::data::ColumnMap::fromMapInput(sample);
    _featurizer->trainInputTransform()->apply(std::move(table), *_state);
  }

  void updateTemporalTrackersBatch(const MapInputBatch& samples) final {
    auto table = thirdai::data::ColumnMap::fromMapInputBatch(samples);
    _featurizer->trainInputTransform()->apply(std::move(table), *_state);
  }

  void resetTemporalTrackers() final { _state->clearHistoryTrackers(); }

  const TextDatasetConfig& textDatasetConfig() const final {
    return _featurizer->textDatasetConfig();
  }

  py::object outputCorrectness(const MapInputBatch& samples,
                               const std::vector<uint32_t>& labels,
                               bool sparse_inference,
                               std::optional<uint32_t> num_hashes) final;

  ModelPtr model() const final { return _mach->model(); }

  void setModel(const ModelPtr& model) final;

  FeaturizerPtr featurizer() const final { return nullptr; }

  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& train_metrics,
                       const dataset::DataSourcePtr& val_data,
                       const std::vector<std::string>& val_metrics,
                       const std::vector<CallbackPtr>& callbacks,
                       TrainOptions options,
                       const bolt::DistributedCommPtr& comm) final;

  py::object embedding(const MapInputBatch& sample) final;

  /**
   * This method is still experimental, we should test to see when these
   * embeddings are useful and which tweaks like summing vs averaging and tanh
   * vs reul make a difference.
   */
  py::object entityEmbedding(const Label& label) final;

  void introduceDocuments(const dataset::DataSourcePtr& data,
                          const std::vector<std::string>& strong_column_names,
                          const std::vector<std::string>& weak_column_names,
                          std::optional<uint32_t> num_buckets_to_sample,
                          uint32_t num_random_hashes, bool fast_approximation,
                          bool verbose) final;

  void introduceDocument(const MapInput& document,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         const Label& new_label,
                         std::optional<uint32_t> num_buckets_to_sample,
                         uint32_t num_random_hashes) final;

  void introduceLabel(const MapInputBatch& samples, const Label& new_label,
                      std::optional<uint32_t> num_buckets_to_sample,
                      uint32_t num_random_hashes) final;

  void forget(const Label& label) final;

  void clearIndex() final {
    _mach->eraseAllEntities();

    if (auto sampler = _state->rlhfSampler()) {
      sampler->clear();
    }
  }

  void associate(
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples,
      uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) final;

  void upvote(const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
              uint32_t n_upvote_samples, uint32_t n_balancing_samples,
              float learning_rate, uint32_t epochs) final;

  py::object associateTrain(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) final;

  py::object associateColdStart(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) final;

  void setDecodeParams(uint32_t top_k_to_return,
                       uint32_t num_buckets_to_eval) final;

  void verifyCanDistribute() const final {
    if (_featurizer->hasTemporalTransform()) {
      throw std::invalid_argument(
          "UDT with temporal relationships cannot be trained in a distributed "
          "setting.");
    }
  }

  dataset::mach::MachIndexPtr getIndex() const final { return _mach->index(); }

  void setIndex(const dataset::mach::MachIndexPtr& index) final;

  void setMachSamplingThreshold(float threshold) final;

 private:
  py::object associateTrainOnColumnMap(
      data::ColumnMap train_data,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options);

  const auto& inputColumns() const { return _featurizer->inputColumns(); }

  void requireRLHFSampler();

  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc) final {
    if (_state->rlhfSampler()) {
      std::cout << "rlhf already enabled." << std::endl;
      return;
    }

    _state->setRlhfSampler(
        RLHFSampler::make(num_balancing_docs, num_balancing_samples_per_doc));
  }

  std::vector<uint32_t> topHashesForDoc(
      std::vector<TopKActivationsQueue>&& top_k_per_sample,
      uint32_t num_buckets_to_sample, uint32_t num_random_hashes = 0) const;

  InputMetrics getMetrics(const std::vector<std::string>& metric_names,
                          const std::string& prefix);

  static void warnOnNonHashBasedMetrics(
      const std::vector<std::string>& metrics);

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

  UDTMachClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  utils::MachPtr _mach;

  char _delimiter;
  UDTTransformationFactoryPtr _featurizer;
  thirdai::data::StatePtr _state;

  uint32_t _default_top_k_to_return;
  uint32_t _num_buckets_to_eval;
};

}  // namespace thirdai::automl::udt