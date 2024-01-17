#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/trainer/DistributedComm.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/udt/Defaults.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/mach/MachIndex.h>
#include <pybind11/pybind11.h>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::automl::udt {

using bolt::callbacks::CallbackPtr;

using bolt::ModelPtr;

using Label = std::variant<uint32_t, std::string>;

struct TrainOptions {
  std::optional<size_t> batch_size = std::nullopt;
  std::optional<size_t> max_in_memory_batches = std::nullopt;
  std::optional<uint32_t> steps_per_validation = std::nullopt;
  bool sparse_validation = false;
  bool verbose = true;
  std::optional<uint32_t> logging_interval = std::nullopt;
  dataset::DatasetShuffleConfig shuffle_config =
      dataset::DatasetShuffleConfig();

  size_t batchSize() const { return batch_size.value_or(defaults::BATCH_SIZE); }
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
  virtual py::object train(const dataset::DataSourcePtr& data,
                           float learning_rate, uint32_t epochs,
                           const std::vector<std::string>& train_metrics,
                           const dataset::DataSourcePtr& val_data,
                           const std::vector<std::string>& val_metrics,
                           const std::vector<CallbackPtr>& callbacks,
                           TrainOptions options,
                           const bolt::DistributedCommPtr& comm) = 0;

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

  virtual py::object evaluate(const dataset::DataSourcePtr& data,
                              const std::vector<std::string>& metrics,
                              bool sparse_inference, bool verbose,
                              std::optional<uint32_t> top_k) = 0;

  virtual py::object predict(const MapInput& sample, bool sparse_inference,
                             bool return_predicted_class,
                             std::optional<uint32_t> top_k) = 0;

  virtual py::object predictBatch(const MapInputBatch& sample,
                                  bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) = 0;

  virtual ar::ConstArchivePtr toArchive(bool with_optimizer) const = 0;

  virtual py::object scoreBatch(const MapInputBatch& samples,
                                const std::vector<std::vector<Label>>& classes,
                                std::optional<uint32_t> top_k) {
    (void)samples;
    (void)classes;
    (void)top_k;
    throw notSupported("scoring");
  }

  virtual py::object outputCorrectness(const MapInputBatch& sample,
                                       const std::vector<uint32_t>& labels,
                                       bool sparse_inference,
                                       std::optional<uint32_t> num_hashes) {
    (void)sample;
    (void)labels;
    (void)sparse_inference;
    (void)num_hashes;
    throw notSupported("output correctness");
  }

  virtual ModelPtr model() const {
    throw notSupported("accessing underlying model");
  }

  virtual void setModel(const ModelPtr& model) {
    (void)model;
    throw notSupported("modifying underlying model");
  }

  virtual FeaturizerPtr featurizer() const { return nullptr; }

  virtual void verifyCanDistribute() const {
    throw notSupported("train_distributed");
  }

  virtual std::vector<std::pair<std::string, float>> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class) {
    (void)sample;
    (void)target_class;
    throw notSupported("explain");
  }

  virtual py::object coldstart(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<CallbackPtr>& callbacks, TrainOptions options,
      const bolt::DistributedCommPtr& comm) {
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
    (void)comm;
    (void)variable_length;
    throw notSupported("cold_start");
  }

  virtual py::object embedding(const MapInputBatch& sample) {
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

  virtual TextDatasetConfig textDatasetConfig() const {
    throw notSupported("text_dataset_config");
  }

  virtual void indexNodes(const dataset::DataSourcePtr& source) {
    (void)source;
    throw notSupported("index_nodes");
  }

  virtual void clearGraph() { throw notSupported("clear_graph"); }

  virtual void setDecodeParams(uint32_t top_k_to_return,
                               uint32_t num_buckets_to_eval) {
    (void)top_k_to_return;
    (void)num_buckets_to_eval;
    throw notSupported("set_decode_params");
  }

  virtual void introduceDocuments(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<uint32_t> num_buckets_to_sample, uint32_t num_random_hashes,
      bool fast_approximation, bool verbose) {
    (void)data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)num_buckets_to_sample;
    (void)num_random_hashes;
    (void)fast_approximation;
    (void)verbose;
    throw notSupported("introduce_documents");
  }

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

  virtual void forget(const std::variant<uint32_t, std::string>& label) {
    (void)label;
    throw notSupported("forget");
  }

  virtual void clearIndex() { throw notSupported("clear_index"); }

  virtual py::object trainWithHashes(const MapInputBatch& batch,
                                     float learning_rate,
                                     const std::vector<std::string>& metrics) {
    (void)batch;
    (void)learning_rate;
    (void)metrics;
    throw notSupported("train_with_hashes");
  }

  virtual py::object predictHashes(const MapInput& sample,
                                   bool sparse_inference, bool force_non_empty,
                                   std::optional<uint32_t> num_hashes) {
    (void)sample;
    (void)sparse_inference;
    (void)force_non_empty;
    (void)num_hashes;
    throw notSupported("predict_hashes");
  }

  virtual py::object predictHashesBatch(const MapInputBatch& samples,
                                        bool sparse_inference,
                                        bool force_non_empty,
                                        std::optional<uint32_t> num_hashes) {
    (void)samples;
    (void)sparse_inference;
    (void)force_non_empty;
    (void)num_hashes;
    throw notSupported("predict_hashes_batch");
  }

  virtual void associate(
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples,
      uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
    (void)rlhf_samples;
    (void)n_association_samples;
    (void)n_balancing_samples;
    (void)n_buckets;
    (void)learning_rate;
    (void)epochs;
    throw notSupported("associate");
  }

  virtual void upvote(
      const std::vector<std::pair<std::string, uint32_t>>& rlhf_samples,
      uint32_t n_upvote_samples, uint32_t n_balancing_samples,
      float learning_rate, uint32_t epochs) {
    (void)rlhf_samples;
    (void)n_upvote_samples;
    (void)n_balancing_samples;
    (void)learning_rate;
    (void)epochs;
    throw notSupported("upvote");
  }

  virtual py::object associateTrain(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) {
    (void)balancing_data;
    (void)rlhf_samples;
    (void)n_buckets;
    (void)n_association_samples;
    (void)learning_rate;
    (void)epochs;
    (void)metrics;
    (void)options;
    throw notSupported("associate_train");
  }

  virtual py::object associateColdStart(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) {
    (void)balancing_data;
    (void)strong_column_names;
    (void)weak_column_names;
    (void)rlhf_samples;
    (void)n_buckets;
    (void)n_association_samples;
    (void)learning_rate;
    (void)epochs;
    (void)metrics;
    (void)options;
    throw notSupported("associate_cold_start");
  }

  virtual void enableRlhf(uint32_t num_balancing_docs,
                          uint32_t num_balancing_samples_per_doc) {
    (void)num_balancing_docs;
    (void)num_balancing_samples_per_doc;
    throw notSupported("enable_rlhf");
  }

  virtual dataset::mach::MachIndexPtr getIndex() const {
    throw notSupported("get_index");
  }

  virtual void setIndex(const dataset::mach::MachIndexPtr& index) {
    (void)index;
    throw notSupported("set_index");
  }

  virtual void setMachSamplingThreshold(float threshold) {
    (void)threshold;
    throw notSupported("set_mach_sampling_threshold");
  }

  virtual void saveCppClassifier(const std::string& save_path) const {
    (void)save_path;
    throw notSupported("save_cpp_classifier");
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