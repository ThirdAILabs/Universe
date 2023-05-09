#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/featurization/TabularOptions.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <stdexcept>

namespace thirdai::automl::udt {

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
                    const config::ArgumentMap& user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::optional<ValidationDataSource>& validation,
                   std::optional<size_t> batch_size_opt,
                   std::optional<size_t> max_in_memory_batches,
                   const std::vector<std::string>& metrics,
                   const std::vector<CallbackPtr>& callbacks, bool verbose,
                   std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object trainBatch(const MapInputBatch& batch, float learning_rate,
                        const std::vector<std::string>& metrics) final;

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class) final;

  py::object trainWithHashes(const MapInputBatch& batch,
                             const std::vector<uint32_t>& hashes, float learning_rate,
                             const std::vector<std::string>& metrics) final;

  py::object predictKHashes(const MapInput& sample, bool sparse_inference,
                            uint32_t k) final;

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final;

  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& metrics,
                       const std::optional<ValidationDataSource>& validation,
                       const std::vector<CallbackPtr>& callbacks,
                       std::optional<size_t> max_in_memory_batches,
                       bool verbose) final;

  py::object embedding(const MapInput& sample) final;

  /**
   * This method is still experimental, we should test to see when these
   * embeddings are useful and which tweaks like summing vs averaging and tanh
   * vs reul make a difference.
   */
  py::object entityEmbedding(const Label& label) final;

  void introduceDocuments(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names) final;

  void introduceDocument(const MapInput& document,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         const Label& new_label) final;

  void introduceLabel(const MapInputBatch& samples,
                      const Label& new_label) final;

  void forget(const Label& label) final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

  void setDecodeParams(uint32_t min_num_eval_results,
                       uint32_t top_k_per_eval_aggregation) final;

  void verifyCanDistribute() const final {
    _dataset_factory->verifyCanDistribute();
  }

  TextEmbeddingModelPtr getTextEmbeddingModel(
      float distance_cutoff) const final;

 private:
  bool integerTarget() const {
    return static_cast<bool>(
        dataset::mach::asNumericIndex(_mach_label_block->index()));
  }

  cold_start::ColdStartMetaDataPtr getColdStartMetaData() final {
    return std::make_shared<cold_start::ColdStartMetaData>(
        /* label_delimiter = */ _mach_label_block->delimiter(),
        /* label_column_name = */ _mach_label_block->columnName());
  }

  std::string variantToString(const Label& variant);

  std::string textColumnForDocumentIntroduction();

  std::unordered_map<Label, MapInputBatch> aggregateSamplesByDoc(
      const thirdai::data::ColumnMap& augmented_data,
      const std::string& text_column_name,
      const std::string& label_column_name);

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
  uint32_t _min_num_eval_results;
  uint32_t _top_k_per_eval_aggregation;
};

}  // namespace thirdai::automl::udt