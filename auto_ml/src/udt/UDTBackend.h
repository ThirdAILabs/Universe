#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::automl::udt {

class Validation {
 public:
  Validation(dataset::DataSourcePtr data, std::vector<std::string> metrics,
             std::optional<uint32_t> steps_per_validation = std::nullopt,
             bool sparse_inference = false)
      : _data(std::move(data)),
        _metrics(std::move(metrics)),
        _steps_per_validation(steps_per_validation),
        _sparse_inference(sparse_inference) {}

  const auto& data() const { return _data; }

  const auto& metrics() const { return _metrics; }

  const auto& stepsPerValidation() const { return _steps_per_validation; }

  bool sparseInference() const { return _sparse_inference; }

 private:
  dataset::DataSourcePtr _data;
  std::vector<std::string> _metrics;
  std::optional<uint32_t> _steps_per_validation;
  bool _sparse_inference;
};

class UDTBackend {
 public:
  virtual void train(
      const dataset::DataSourcePtr& train_data, uint32_t epochs,
      float learning_rate, const std::optional<Validation>& validation,
      std::optional<size_t> batch_size,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& train_metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval) = 0;

  virtual py::object evaluate(const dataset::DataSourcePtr& data,
                              const std::vector<std::string>& metrics,
                              bool sparse_inference,
                              bool return_predicted_class, bool verbose,
                              bool return_metrics) = 0;

  virtual py::object predict(const MapInput& sample, bool sparse_inference,
                             bool return_predicted_class) = 0;

  virtual py::object predictBatch(const MapInputBatch& sample,
                                  bool sparse_inference,
                                  bool return_predicted_class) = 0;

  virtual std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class) {
    (void)sample;
    (void)target_class;
    throw notSupported("explain");
  }

  virtual void coldstart(const dataset::DataSourcePtr& original_source,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         uint32_t epochs, float learning_rate,
                         const std::vector<std::string>& train_metrics,
                         const std::optional<Validation>& validation,
                         bool verbose) {
    (void)original_source;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)epochs;
    (void)learning_rate;
    (void)train_metrics;
    (void)validation;
    (void)verbose;
    throw notSupported("cold_start");
  }

  virtual py::object embedding(const MapInput& sample) {
    (void)sample;
    throw notSupported("embedding");
  }

  virtual py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) {
    (void)label;
    throw notSupported("entity_embedding");
  }

  virtual std::string className(uint32_t class_id) const {
    (void)class_id;
    throw notSupported("class_name");
  }

  virtual data::TabularDatasetFactoryPtr tabularDatasetFactory() const {
    return nullptr;
  }

 private:
  static std::runtime_error notSupported(const std::string& name) {
    return std::runtime_error("Method '" + name +
                              "' is not supported for this type of model.");
  }
};

}  // namespace thirdai::automl::udt