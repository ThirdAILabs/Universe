#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/backends/UDTMachClassifier.h>
#include <dataset/src/DataSource.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::udt {

/**
 * UDT is composed of various backends that implement the logic specific to
 * different models, classification, regression, etc. This class users the
 * arguments supplied by the user to determine what backend to use for the given
 * task/dataset and then stores that corresponding backend within it. This
 * pattern of composition allows us to have different backends for different
 * model types, but without exposing that implementation detail to the user and
 * presenting a single class for them to interact with. This class also act as a
 * common point where we can implement common features, for instance telemetry,
 * that we want for all types of models.
 */
class UDT {
 public:
  UDT(data::ColumnDataTypes data_types,
      const data::UserProvidedTemporalRelationships&
          temporal_tracking_relationships,
      const std::string& target_col, std::optional<uint32_t> n_target_classes,
      bool integer_target, std::string time_granularity, uint32_t lookahead,
      char delimiter, const std::optional<std::string>& model_config,
      const config::ArgumentMap& user_args);

  UDT(std::optional<std::string> incorrect_column_name,
      std::string correct_column_name, const std::string& dataset_size,
      char delimiter, const std::optional<std::string>& model_config,
      const config::ArgumentMap& user_args);

  UDT(const std::string& file_format, uint32_t n_target_classes,
      uint32_t input_dim, const std::optional<std::string>& model_config,
      const config::ArgumentMap& user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options);

  py::object trainBatch(const MapInputBatch& batch, float learning_rate,
                        const std::vector<std::string>& metrics);

  void setOutputSparsity(float sparsity, bool rebuild_hash_tables);

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k);

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k);

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k);

  std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class);

  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& train_metrics,
                       const dataset::DataSourcePtr& val_data,
                       const std::vector<std::string>& val_metrics,
                       const std::vector<CallbackPtr>& callbacks,
                       TrainOptions options);

  cold_start::ColdStartMetaDataPtr getColdStartMetaData() {
    return _backend->getColdStartMetaData();
  }

  py::object embedding(const MapInput& sample) {
    return _backend->embedding(sample);
  }

  py::object entityEmbedding(const std::variant<uint32_t, std::string>& label) {
    return _backend->entityEmbedding(label);
  }

  std::string className(uint32_t class_id) const {
    return _backend->className(class_id);
  }

  void updateTemporalTrackers(const MapInput& sample) {
    if (auto tabular_factory = _backend->tabularDatasetFactory()) {
      tabular_factory->updateTemporalTrackers(sample);
    }
  }

  void updateTemporalTrackersBatch(const MapInputBatch& samples) {
    if (auto tabular_factory = _backend->tabularDatasetFactory()) {
      tabular_factory->updateTemporalTrackersBatch(samples);
    }
  }

  void resetTemporalTrackers() {
    if (auto tabular_factory = _backend->tabularDatasetFactory()) {
      tabular_factory->resetTemporalTrackers();
    }
  }

  void updateMetadata(const std::string& column, const MapInput& sample) {
    if (auto tabular_factory = _backend->tabularDatasetFactory()) {
      tabular_factory->updateMetadata(column, sample);
    }
  }

  void updateMetadataBatch(const std::string& column,
                           const MapInputBatch& samples) {
    if (auto tabular_factory = _backend->tabularDatasetFactory()) {
      tabular_factory->updateMetadataBatch(column, samples);
    }
  }

  void indexNodes(const dataset::DataSourcePtr& source) {
    return _backend->indexNodes(source);
  }

  void clearGraph() { return _backend->clearGraph(); }

  ModelPtr model() const { return _backend->model(); }

  void setModel(const ModelPtr& model) { _backend->setModel(model); }

  std::shared_ptr<UDTMachClassifier> neuralDB() const {
    return std::dynamic_pointer_cast<UDTMachClassifier>(_backend);
  }

  std::vector<uint32_t> modelDims() const;

  data::ColumnDataTypes dataTypes() const { return _backend->dataTypes(); }

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const {
    return _backend->tabularDatasetFactory();
  }

  void verifyCanDistribute() const { _backend->verifyCanDistribute(); }

  void save(const std::string& filename) const;

  void saveImpl(const std::string& filename) const;

  void checkpoint(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<UDT> load(const std::string& filename);

  static std::shared_ptr<UDT> load_stream(std::istream& input_stream);

 private:
  UDT() {}

  static bool hasGraphInputs(const data::ColumnDataTypes& data_types);

  static void throwUnsupportedUDTConfigurationError(
      const data::CategoricalDataTypePtr& target_as_categorical,
      const data::NumericalDataTypePtr& target_as_numerical,
      const data::SequenceDataTypePtr& target_as_sequence,
      bool has_graph_inputs);

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  std::shared_ptr<UDTBackend> _backend;
};

}  // namespace thirdai::automl::udt