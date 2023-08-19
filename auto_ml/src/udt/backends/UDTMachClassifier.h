#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
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
  UDTMachClassifier(const data::ColumnDataTypes& input_data_types,
                    const data::UserProvidedTemporalRelationships&
                        temporal_tracking_relationships,
                    const std::string& target_name,
                    const data::CategoricalDataTypePtr& target,
                    uint32_t n_target_classes, bool integer_target,
                    const data::TabularOptions& tabular_options,
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

  py::object trainWithHashes(const MapInputBatch& batch, float learning_rate,
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

  py::object predictHashes(const MapInput& sample, bool sparse_inference,
                           bool force_non_empty,
                           std::optional<uint32_t> num_hashes) final;

  py::object predictHashesBatch(const MapInputBatch& samples,
                                bool sparse_inference, bool force_non_empty,
                                std::optional<uint32_t> num_hashes) final;

  py::object outputCorrectness(const MapInputBatch& samples,
                               const std::vector<uint32_t>& labels,
                               bool sparse_inference,
                               std::optional<uint32_t> num_hashes) final;

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final;

  data::ColumnDataTypes dataTypes() const final {
    return _dataset_factory->dataTypes();
  }

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

  py::object embedding(const MapInput& sample) final;

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
    _mach_label_block->index()->clear();

    updateSamplingStrategy();

    if (_rlhf_sampler) {
      _rlhf_sampler->clear();
    }
  }

  void associate(
      const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
      uint32_t n_buckets, uint32_t n_association_samples,
      uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) final;

  void upvote(
      const std::vector<std::pair<MapInput, uint32_t>>& source_target_samples,
      uint32_t n_upvote_samples, uint32_t n_balancing_samples,
      float learning_rate, uint32_t epochs) final;

  py::object associateTrain(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) final;

  py::object associateColdStart(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

  void setDecodeParams(uint32_t top_k_to_return,
                       uint32_t num_buckets_to_eval) final;

  void verifyCanDistribute() const final {
    _dataset_factory->verifyCanDistribute();
  }

  dataset::mach::MachIndexPtr getIndex() final {
    return _mach_label_block->index();
  }

  void setIndex(const dataset::mach::MachIndexPtr& index) final;

  void setMachSamplingThreshold(float threshold) final;

 private:
  std::vector<std::vector<std::pair<uint32_t, double>>> predictImpl(
      const MapInputBatch& samples, bool sparse_inference,
      std::optional<uint32_t> top_k);

  std::vector<std::vector<uint32_t>> predictHashesImpl(
      const MapInputBatch& samples, bool sparse_inference,
      bool force_non_empty = true,
      std::optional<uint32_t> num_hashes = std::nullopt);

  void teach(const std::vector<std::pair<MapInput, std::vector<uint32_t>>>&
                 source_target_samples,
             uint32_t n_buckets, uint32_t n_teaching_samples,
             uint32_t n_balancing_samples, float learning_rate,
             uint32_t epochs);

  std::vector<std::pair<MapInput, std::vector<uint32_t>>> getAssociateSamples(
      const std::vector<std::pair<MapInput, MapInput>>& source_target_samples);

  cold_start::ColdStartMetaDataPtr getColdStartMetaData() final {
    return std::make_shared<cold_start::ColdStartMetaData>(
        /* label_delimiter = */ _mach_label_block->delimiter(),
        /* label_column_name = */ _mach_label_block->columnName());
  }

  std::string textColumnForDocumentIntroduction();

  void updateSamplingStrategy();

  void addBalancingSamples(const dataset::DataSourcePtr& data);

  void requireRLHFSampler();

  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc) final {
    if (_rlhf_sampler.has_value()) {
      std::cout << "rlhf already enabled." << std::endl;
      return;
    }

    _rlhf_sampler = std::make_optional<RLHFSampler>(
        num_balancing_docs, num_balancing_samples_per_doc);
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

  std::shared_ptr<utils::Classifier> _classifier;

  dataset::mach::MachBlockPtr _mach_label_block;
  data::TabularDatasetFactoryPtr _dataset_factory;
  data::TabularDatasetFactoryPtr _pre_hashed_labels_dataset_factory;

  uint32_t _default_top_k_to_return;
  uint32_t _num_buckets_to_eval;
  float _mach_sampling_threshold;

  std::optional<RLHFSampler> _rlhf_sampler;
};

}  // namespace thirdai::automl::udt