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

/**
 * Stores necessary information for validation, used to simplify args and avoid
 * having to pass in multiple optional arguments and verify that they are
 * correctly specified together.
 */
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

/**
 * This is an interface for the backends that are used in a UDT model. To add a
 * new backend a user must implement the required methods (train, evaluate,
 * predict, etc.) and any desired optional methods (explainability, cold start,
 * etc.). These methods are designed to be general in their arguments and
 * support the options that are required for most backends, though some backends
 * may not use all of the args. For instance return_predicted_class is not
 * applicable for regression models.
 */
class UDTBackend {
 public:
  /**
   * Trains the model on the given dataset.
   */
  virtual void train(
      const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
      const std::optional<Validation>& validation,
      std::optional<size_t> batch_size,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval) = 0;

  /**
   * Performs evaluate of the model on the given dataset and returns the
   * activations produced by the model by default. If return_predicted_class is
   * specified it should return the predicted classes if its a classification
   * task instead of the activations. If return metrics is specified then it
   * should return the metrics computed instead of any activations.
   */
  virtual py::object evaluate(const dataset::DataSourcePtr& data,
                              const std::vector<std::string>& metrics,
                              bool sparse_inference,
                              bool return_predicted_class, bool verbose,
                              bool return_metrics,
                              std::optional<uint32_t> top_k) = 0;

  /**
   * Performs inference on a single sample and returns the resulting
   * activations. If return_predicted_class is specified it should return the
   * predicted classes if its a classification task instead of the activations.
   */
  virtual py::object predict(const MapInput& sample, bool sparse_inference,
                             bool return_predicted_class,
                             std::optional<uint32_t> top_k) = 0;

  /**
   * Performs inference on a batch of samples in parallel and returns the
   * resulting activations. If return_predicted_class is specified it should
   * return the predicted classes if its a classification task instead of the
   * activations.
   */
  virtual py::object predictBatch(const MapInputBatch& sample,
                                  bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) = 0;

  /**
   * Returns the model used.
   */
  virtual bolt::BoltGraphPtr model() const {
    throw notSupported("accessing underlying model");
  }

  /**
   * Sets a new model. This is used during distributed training to update the
   * backend with the trained model.
   */
  virtual void setModel(const bolt::BoltGraphPtr& model) {
    (void)model;
    throw notSupported("modifying underlying model");
  }

  /**
   * Determines if the model can support distributed training. By default
   * backends do not support distributed training.
   */
  virtual void verifyCanDistribute() const {
    throw notSupported("train_distributed");
  }

  /**
   * Generates an explaination of the prediction for a given sample. Optional
   * method that is not supported by default for backends.
   */
  virtual std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class) {
    (void)sample;
    (void)target_class;
    throw notSupported("explain");
  }

  /**
   * Performs cold start pretraining. Optional method that is not supported by
   * default for backends.
   */
  virtual void coldstart(const dataset::DataSourcePtr& data,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         float learning_rate, uint32_t epochs,
                         const std::vector<std::string>& metrics,
                         const std::optional<Validation>& validation,
                         const std::vector<bolt::CallbackPtr>& callbacks,
                         bool verbose) {
    (void)data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)learning_rate;
    (void)epochs;
    (void)metrics;
    (void)validation;
    (void)callbacks;
    (void)verbose;
    throw notSupported("cold_start");
  }

  /**
   * Returns some embedding representation for the given sample. Optional method
   * that is not supported by default for backends.
   */
  virtual py::object embedding(const MapInput& sample) {
    (void)sample;
    throw notSupported("embedding");
  }

  /**
   * Returns an embedding for the given class (label) in the model. Optional
   * method that is not supported by default for backends.
   */
  virtual py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) {
    (void)label;
    throw notSupported("entity_embedding");
  }

  /**
   * Returns the class name associated with a given neuron. Optional method that
   * is not supported by default for backends.
   */
  virtual std::string className(uint32_t class_id) const {
    (void)class_id;
    throw notSupported("class_name");
  }

  /**
   * Returns the tabular dataset factor if it is used for the model. If a
   * backend implements this method then UDT instances that use it will support
   * methods relating to temporal tracking and metadata.
   */
  virtual data::TabularDatasetFactoryPtr tabularDatasetFactory() const {
    return nullptr;
  }

  virtual ~UDTBackend() = default;

 protected:
  UDTBackend() {}

  static std::runtime_error notSupported(const std::string& name) {
    return std::runtime_error("Method '" + name +
                              "' is not supported for this type of model.");
  }

 private:
  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

}  // namespace thirdai::automl::udt