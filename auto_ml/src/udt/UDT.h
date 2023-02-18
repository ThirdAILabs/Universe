#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <string>

namespace thirdai::automl::udt {

class UDT {
 public:
  UDT(data::ColumnDataTypes data_types,
      const data::UserProvidedTemporalRelationships&
          temporal_tracking_relationships,
      const std::string& target_col, std::optional<uint32_t> n_target_classes,
      bool integer_target, std::string time_granularity, uint32_t lookahead,
      char delimiter, const std::optional<std::string>& model_config,
      const config::ArgumentMap& user_args);

  void train(const dataset::DataSourcePtr& data, float learning_rate,
             uint32_t epochs, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
             std::optional<size_t> max_in_memory_batches,
             const std::vector<std::string>& metrics,
             const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
             bool verbose, std::optional<uint32_t> logging_interval) {
    _backend->train(data, learning_rate, epochs, validation, batch_size,
                    max_in_memory_batches, metrics, callbacks, verbose,
                    logging_interval);
  }

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics) {
    return _backend->evaluate(data, metrics, sparse_inference,
                              return_predicted_class, verbose, return_metrics);
  }

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) {
    return _backend->predict(sample, sparse_inference, return_predicted_class);
  }

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) {
    return _backend->predictBatch(sample, sparse_inference,
                                  return_predicted_class);
  }

  std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class) {
    return _backend->explain(sample, target_class);
  }

  void coldstart(const dataset::DataSourcePtr& data,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 float learning_rate, uint32_t epochs,
                 const std::vector<std::string>& metrics,
                 const std::optional<Validation>& validation,
                 const std::vector<bolt::CallbackPtr>& callbacks,
                 bool verbose) {
    return _backend->coldstart(data, strong_column_names, weak_column_names,
                               learning_rate, epochs, metrics, validation,
                               callbacks, verbose);
  }

  py::object embedding(const MapInput& sample) {
    return _backend->embedding(sample);
  }

  py::object embedding(const std::variant<uint32_t, std::string>& label) {
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

  void save(const std::string& filename);

  static std::shared_ptr<UDT> load(const std::string& filename);

 private:
  UDT() {}

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  std::unique_ptr<UDTBackend> _backend;
};

}  // namespace thirdai::automl::udt