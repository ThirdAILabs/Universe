#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTClassifier final : public UDTBackend {
 public:
  UDTClassifier(const data::ColumnDataTypes& input_data_types,
                const data::UserProvidedTemporalRelationships&
                    temporal_tracking_relationships,
                const std::string& target_name,
                data::CategoricalDataTypePtr target, uint32_t n_target_classes,
                bool integer_target,
                const data::TabularOptions& tabular_options,
                const std::optional<std::string>& model_config,
                const config::ArgumentMap& user_args);

  void train(const dataset::DataSourcePtr& data, float learning_rate,
             uint32_t epochs, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
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

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final;

  std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class)
      final;

  void coldstart(const dataset::DataSourcePtr& data,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 float learning_rate, uint32_t epochs,
                 const std::vector<std::string>& metrics,
                 const std::optional<Validation>& validation,
                 const std::vector<bolt::CallbackPtr>& callbacks,
                 bool verbose) final;

  py::object embedding(const MapInput& sample) final;

  py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) final;

  std::string className(uint32_t class_id) const final {
    if (_class_name_to_neuron) {
      return _class_name_to_neuron->getString(class_id);
    }
    return std::to_string(class_id);
  }

  bolt::BoltGraphPtr model() const final { return _model; }

  void setModel(bolt::BoltGraphPtr model) final {
    utils::trySetModel(_model, model);
  }

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

  void verifyCanDistribute() const final {
    if (!integerTarget()) {
      throw std::invalid_argument(
          "UDT with a categorical target cannot be trained in distributed "
          "setting without integer_target=True. Please convert the categorical "
          "target column into an integer target to train UDT in a distributed "
          "setting.");
    }

    _dataset_factory->verifyCanDistribute();
  }

 private:
  dataset::CategoricalBlockPtr labelBlock(
      const std::string& target_name,
      data::CategoricalDataTypePtr& target_config, uint32_t n_target_classes,
      bool integer_target, bool normalize_target_categories);

  uint32_t labelToNeuronId(
      const std::variant<uint32_t, std::string>& label) const;

  bool integerTarget() const { return !_class_name_to_neuron; }

  UDTClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  dataset::ThreadSafeVocabularyPtr _class_name_to_neuron;
  dataset::CategoricalBlockPtr _label_block;

  bolt::BoltGraphPtr _model;
  data::TabularDatasetFactoryPtr _dataset_factory;

  bool _freeze_hash_tables;

  std::optional<float> _binary_prediction_threshold;
};

}  // namespace thirdai::automl::udt