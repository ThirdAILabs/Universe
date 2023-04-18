#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/embedding_prototype/TextEmbeddingModel.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/Validation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::automl::udt {

/**
 * This is an interface for the backends that are used in a UDT model. To
 * add a new backend a user must implement the required methods (train,
 * evaluate, predict, etc.) and any desired optional methods
 * (explainability, cold start, etc.). These methods are designed to be
 * general in their arguments and support the options that are required for
 * most backends, though some backends may not use all of the args. For
 * instance return_predicted_class is not applicable for regression models.
 */
class UDTBackend {
 public:
  /**
   * Trains the model on the given dataset.
   */
  virtual py::object train(
      const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
      const std::optional<ValidationDataSource>& validation,
      std::optional<size_t> batch_size,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval) = 0;

  /**
   * Trains the model on a batch of samples.
   */
  virtual py::object trainBatch(const MapInputBatch& batch, float learning_rate,
                                const std::vector<std::string>& metrics) {
    (void)batch;
    (void)learning_rate;
    (void)metrics;
    throw notSupported("train_batch");
  }

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
                              bool return_metrics) = 0;

  /**
   * Performs inference on a single sample and returns the resulting
   * activations. If return_predicted_class is specified it should return the
   * predicted classes if its a classification task instead of the activations.
   */
  virtual py::object predict(const MapInput& sample, bool sparse_inference,
                             bool return_predicted_class) = 0;

  /**
   * Performs inference on a batch of samples in parallel and returns the
   * resulting activations. If return_predicted_class is specified it should
   * return the predicted classes if its a classification task instead of the
   * activations.
   */
  virtual py::object predictBatch(const MapInputBatch& sample,
                                  bool sparse_inference,
                                  bool return_predicted_class) = 0;

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
  virtual py::object coldstart(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      const std::optional<ValidationDataSource>& validation,
      const std::vector<bolt::CallbackPtr>& callbacks,
      std::optional<size_t> max_in_memory_batches, bool verbose) {
    (void)data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)learning_rate;
    (void)epochs;
    (void)metrics;
    (void)validation;
    (void)callbacks;
    (void)max_in_memory_batches;
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

  /**
   * Returns metadata for ColdStart which are needed to be passed to
   * ColdStartPreprocessing. Optional Method that is not supported by
   * defaults for backends. This method is primarily used for distributed
   * training.
   */
  virtual cold_start::ColdStartMetaDataPtr getColdStartMetaData() {
    throw notSupported("getColdStartMetaData");
  }

  virtual void indexNodes(const dataset::DataSourcePtr& source) {
    (void)source;
    throw notSupported("index_nodes");
  }

  virtual void clearGraph() { throw notSupported("clear_graph"); }

  /**
   * Used for UDTMachClassifier.
   */
  virtual void setDecodeParams(uint32_t min_num_eval_results,
                               uint32_t top_k_per_eval_aggregation) {
    (void)min_num_eval_results;
    (void)top_k_per_eval_aggregation;
    throw notSupported("set_decode_params");
  }

  /**
   * Introduces new documents to the model and used in conjunction with
   * coldstart.
   */
  virtual void introduceDocuments(const dataset::DataSourcePtr& data) {
    (void)data;
    throw notSupported("introduce_documents");
  }

  /**
   * Introduces a new label to the model given a batch of representative samples
   * of that label.
   */
  virtual void introduce(const MapInputBatch& sample,
                         const std::variant<uint32_t, std::string>& new_label) {
    (void)sample;
    (void)new_label;
    throw notSupported("introduce");
  }

  /**
   * Forget a given label such that it is impossible to predict in the future.
   */
  virtual void forget(const std::variant<uint32_t, std::string>& label) {
    (void)label;
    throw notSupported("forget");
  }

  /*
   * Returns a model that embeds text using the hidden layer of the UDT model.
   */
  virtual TextEmbeddingModelPtr getTextEmbeddingModel(
      const std::string& activation_func, float distance_cutoff) const {
    (void)activation_func;
    (void)distance_cutoff;
    throw notSupported("get_text_embedding_model");
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