#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/MachBlocks.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <stdexcept>

namespace thirdai::automl::udt {

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

  py::object train(
      const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
      const std::optional<ValidationDataSource>& validation,
      std::optional<size_t> batch_size_opt,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class) final;

  bolt::BoltGraphPtr model() const final { return _classifier->model(); }

  void setModel(const bolt::BoltGraphPtr& model) final {
    bolt::BoltGraphPtr& curr_model = _classifier->model();
    if (curr_model->outputDim() != curr_model->outputDim()) {
      throw std::invalid_argument("Output dim mismatch in set_model.");
    }
    curr_model = model;
  }

  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& metrics,
                       const std::optional<ValidationDataSource>& validation,
                       const std::vector<bolt::CallbackPtr>& callbacks,
                       bool verbose) final;

  py::object embedding(const MapInput& sample) final;

  py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

 private:
  std::vector<std::variant<std::string, uint32_t>> machSingleDecode(
      const BoltVector& output);

  static uint32_t autotuneMachOutputDim(uint32_t n_target_classes) {
    // TODO(david) update this
    return n_target_classes / 25;
  }

  static uint32_t autotuneMachNumHashes(uint32_t n_target_classes,
                                        uint32_t output_range) {
    // TODO(david) update this
    (void)n_target_classes;
    (void)output_range;
    return 7;
  }

  UDTMachClassifier() : _classifier(nullptr, false) {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  std::shared_ptr<utils::Classifier> _classifier;
  dataset::MachIndexPtr _mach_index;
  dataset::CategoricalBlockPtr _multi_hash_label_block;
  data::TabularDatasetFactoryPtr _dataset_factory;
};

}  // namespace thirdai::automl::udt