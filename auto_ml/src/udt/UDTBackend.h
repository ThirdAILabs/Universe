#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::automl::udt {

using bolt::train::callbacks::CallbackPtr;

using bolt::nn::model::ModelPtr;

struct TrainOptions {
  std::optional<size_t> batch_size = std::nullopt;
  std::optional<size_t> max_in_memory_batches = std::nullopt;
  std::optional<uint32_t> steps_per_validation = std::nullopt;
  bool sparse_validation = false;
  bool verbose = true;
  std::optional<uint32_t> logging_interval = std::nullopt;
  dataset::DatasetShuffleConfig shuffle_config =
      dataset::DatasetShuffleConfig();
};

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
  virtual py::object train(const dataset::DataSourcePtr& data,
                           float learning_rate, uint32_t epochs,
                           const std::vector<std::string>& train_metrics,
                           const dataset::DataSourcePtr& val_data,
                           const std::vector<std::string>& val_metrics,
                           const std::vector<CallbackPtr>& callbacks,
                           TrainOptions options) = 0;

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

  virtual void setOutputSparsity(float sparsity, bool rebuild_hash_tables) {
    (void)sparsity;
    (void)rebuild_hash_tables;
    throw notSupported("Method not supported for the model");
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
                              bool sparse_inference, bool verbose,
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

  virtual py::object predictBatchHashes(const MapInputBatch& sample,
                                        bool sparse_inference,
                                        std::optional<uint32_t> top_k) {
    (void)sample;
    (void)sparse_inference;
    (void)top_k;
    throw notSupported("output frequency");
  }

  virtual py::object outputCorrectness(const MapInputBatch& sample,
                                       const std::vector<uint32_t>& labels,
                                       bool sparse_inference, uint32_t top_k) {
    (void)sample;
    (void)labels;
    (void)sparse_inference;
    (void)top_k;
    throw notSupported("output correctness");
  }

  /**
   * Returns the model used.
   */
  virtual ModelPtr model() const {
    throw notSupported("accessing underlying model");
  }

  /**
   * Sets a new model. This is used during distributed training to update the
   * backend with the trained model.
   */
  virtual void setModel(const ModelPtr& model) {
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
      uint32_t epochs, const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<CallbackPtr>& callbacks, TrainOptions options) {
    (void)data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)learning_rate;
    (void)epochs;
    (void)train_metrics;
    (void)val_data;
    (void)val_metrics;
    (void)callbacks;
    (void)options;
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

  virtual data::ColumnDataTypes dataTypes() const {
    throw notSupported("data_types");
  }

  /**
   * Returns metadata for ColdStart which are needed to be passed to
   * ColdStartPreprocessing. Optional Method that is not supported by
   * defaults for backends. This method is primarily used for distributed
   * training.
   */
  virtual cold_start::ColdStartMetaDataPtr getColdStartMetaData() {
    throw notSupported("get_cold_start_meta_data");
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
   * Introduces new documents to the model from a data source. Used in
   * conjunction with coldstart.
   */
  virtual void introduceDocuments(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes,
      bool fast_approximation) {
    (void)data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)num_buckets_to_sample;
    (void)num_random_hashes;
    (void)fast_approximation;
    throw notSupported("introduce_documents");
  }

  /**
   * Introduces a single new document to the model from an in memory map input.
   * Used in conjunction with coldstart.
   */
  virtual void introduceDocument(
      const MapInput& document,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::variant<uint32_t, std::string>& new_label,
      std::optional<uint32_t> num_buckets_to_sample,
      uint32_t num_random_hashes) {
    (void)document;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)new_label;
    (void)num_buckets_to_sample;
    (void)num_random_hashes;
    throw notSupported("introduce_document");
  }

  /**
   * Introduces a new label to the model given a batch of representative samples
   * of that label.
   */
  virtual void introduceLabel(
      const MapInputBatch& sample,
      const std::variant<uint32_t, std::string>& new_label,
      std::optional<uint32_t> num_buckets_to_sample,
      uint32_t num_random_hashes) {
    (void)sample;
    (void)new_label;
    (void)num_buckets_to_sample;
    (void)num_random_hashes;
    throw notSupported("introduce_label");
  }

  /**
   * Forget a given label such that it is impossible to predict in the future.
   */
  virtual void forget(const std::variant<uint32_t, std::string>& label) {
    (void)label;
    throw notSupported("forget");
  }

  /**
   * Clears the internal index for Mach.
   */
  virtual void clearIndex() { throw notSupported("clear_index"); }

  /**
   * Used in UDTMachClassifier, assumes each of the samples in the input batch
   * has the target column mapping to space separated strings representing the
   * actual output metaclasses to predict in mach.
   */
  virtual py::object trainWithHashes(const MapInputBatch& batch,
                                     float learning_rate,
                                     const std::vector<std::string>& metrics) {
    (void)batch;
    (void)learning_rate;
    (void)metrics;
    throw notSupported("train_with_hashes");
  }

  /**
   * Used in UDTMachClassifier, returns the predicted hashes from the input
   * sample.
   */
  virtual py::object predictHashes(const MapInput& sample,
                                   bool sparse_inference,
                                   std::optional<uint32_t> top_k,
                                   bool return_non_empty) {
    (void)sample;
    (void)sparse_inference;
    (void)top_k;
    (void)return_non_empty;
    throw notSupported("predict_hashes");
  }

  /**
   * Used for fine tuning in UDTMachClassifier.
   */
  virtual void associate(
      const std::vector<std::pair<MapInput, MapInput>>& source_target_samples,
      uint32_t n_buckets, uint32_t n_association_samples,
      uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
    (void)source_target_samples;
    (void)n_association_samples;
    (void)n_balancing_samples;
    (void)n_buckets;
    (void)learning_rate;
    (void)epochs;
    throw notSupported("associate");
  }

  /**
   * Used for fine tuning in UDTMachClassifier.
   */
  virtual void upvote(
      const std::vector<std::pair<MapInput, uint32_t>>& source_target_samples,
      uint32_t n_upvote_samples, uint32_t n_balancing_samples,
      float learning_rate, uint32_t epochs) {
    (void)source_target_samples;
    (void)n_upvote_samples;
    (void)n_balancing_samples;
    (void)learning_rate;
    (void)epochs;
    throw notSupported("upvote");
  }

  virtual void enableRlhf(uint32_t num_balancing_docs,
                          uint32_t num_balancing_samples_per_doc) {
    (void)num_balancing_docs;
    (void)num_balancing_samples_per_doc;
    throw notSupported("enable_rlhf");
  }

  /**
   * Gets the internal index for UDTMachClassifier.
   */
  virtual dataset::mach::MachIndexPtr getIndex() {
    throw notSupported("get_index");
  }

  /**
   * Sets the internal index for UDTMachClassifier.
   */
  virtual void setIndex(const dataset::mach::MachIndexPtr& index) {
    (void)index;
    throw notSupported("set_index");
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