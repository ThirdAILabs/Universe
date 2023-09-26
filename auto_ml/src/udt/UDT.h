#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/trainer/DistributedComm.h>
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
                   TrainOptions options, const bolt::DistributedCommPtr& comm);

  py::object trainBatch(const MapInputBatch& batch, float learning_rate,
                        const std::vector<std::string>& metrics);

  void setOutputSparsity(float sparsity, bool rebuild_hash_tables);

  /**
   * Performs evaluate of the model on the given dataset and returns the
   * activations produced by the model by default. If return_predicted_class is
   * specified it should return the predicted classes if its a classification
   * task instead of the activations. If return metrics is specified then it
   * should return the metrics computed instead of any activations.
   */
  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k);

  /**
   * Performs inference on a single sample and returns the resulting
   * activations. If return_predicted_class is specified it should return the
   * predicted classes if its a classification task instead of the activations.
   */
  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k);

  /**
   * Performs inference on a batch of samples in parallel and returns the
   * resulting activations. If return_predicted_class is specified it should
   * return the predicted classes if its a classification task instead of the
   * activations.
   */
  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k);

  py::object outputCorrectness(const MapInputBatch& sample,
                               const std::vector<uint32_t>& labels,
                               bool sparse_inference,
                               std::optional<uint32_t> num_hashes) {
    return _backend->outputCorrectness(sample, labels, sparse_inference,
                                       num_hashes);
  }

  /**
   * Generates an explaination of the prediction for a given sample. Optional
   * method that is not supported by default for backends.
   */
  std::vector<std::pair<std::string, float>> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class);

  /**
   * Performs cold start pretraining. Optional method that is not supported by
   * default for backends.
   */
  py::object coldstart(const dataset::DataSourcePtr& data,
                       const std::vector<std::string>& strong_column_names,
                       const std::vector<std::string>& weak_column_names,
                       float learning_rate, uint32_t epochs,
                       const std::vector<std::string>& train_metrics,
                       const dataset::DataSourcePtr& val_data,
                       const std::vector<std::string>& val_metrics,
                       const std::vector<CallbackPtr>& callbacks,
                       TrainOptions options,
                       const bolt::DistributedCommPtr& comm);

  /**
   * Returns some embedding representation for the given sample. Optional method
   * that is not supported by default for backends.
   */
  py::object embedding(const MapInputBatch& sample) {
    return _backend->embedding(sample);
  }

  /**
   * Returns an embedding for the given class (label) in the model. Optional
   * method that is not supported by default for backends.
   */
  py::object entityEmbedding(const std::variant<uint32_t, std::string>& label) {
    return _backend->entityEmbedding(label);
  }

  /**
   * Returns the class name associated with a given neuron. Optional method that
   * is not supported by default for backends.
   */
  std::string className(uint32_t class_id) const {
    return _backend->className(class_id);
  }

  void updateTemporalTrackers(const MapInput& sample) {
    _backend->updateTemporalTrackers(sample);
  }

  void updateTemporalTrackersBatch(const MapInputBatch& samples) {
    _backend->updateTemporalTrackersBatch(samples);
  }

  void resetTemporalTrackers() { _backend->resetTemporalTrackers(); }

  void indexNodes(const dataset::DataSourcePtr& source) {
    return _backend->indexNodes(source);
  }

  void clearGraph() { return _backend->clearGraph(); }

  /**
   * Used for UDTMachClassifier to set parameters used in decoding entities for
   * the various methods in UDT (ie evaluate, mach metrics for train, etc)
   * @param top_k_to_return is the number of entities for mach to decode
   * @param num_buckets_to_eval is the number of output buckets used to source
   * candidate entities. We then calculate a score for each candidate and return
   * the best.
   */
  void setDecodeParams(uint32_t top_k_to_return, uint32_t num_buckets_to_eval) {
    return _backend->setDecodeParams(top_k_to_return, num_buckets_to_eval);
  }

  /**
   * Returns the underlying BOLT model used.
   */
  ModelPtr model() { return _backend->model(); }

  /**
   * Sets a new model. This is used during distributed training to update the
   * backend with the trained model.
   */
  void setModel(const ModelPtr& model) { _backend->setModel(model); }

  std::vector<uint32_t> modelDims() const { return _backend->modelDims(); }

  const TextDatasetConfig& textDatasetConfig() const {
    return _backend->textDatasetConfig();
  }

  /**
   * Used in UDTMachClassifier to introduce new documents to the model from a
   * data source. Used in conjunction with coldstart. At a high level, introduce
   * documents works by predicting the output hashes of a document and adding
   * that document to the index with those hashes. This way, any pretraining
   * done can be transfered to new documents through the weights of the model.
   * The typical way to introduce involves creating cold_start samples out of
   * the strong and weak columns, running inference on the model for each of
   * those samples, and doing some sort of frequency aggregation to find the
   * best place for the new document in the index.
   *
   * @param num_buckets_to_sample By default we will predict however many hashes
   * are already used in the index (set at construction time) and simply set
   * those hashes for that documents. However, if num_buckets_to_sample is
   * specified, we will predict more hashes than necessary and "place" the
   * document into the least occupied buckets. num_buckets_to_sample must be
   * larger than num_hashes for the index.
   * @param num_random_hashes By default this is 0. Specifies the number of
   * hashes that will be completely random when introducing new documents.
   * Increasing num_random hashes will gradually decrease zero shot accuracy
   * while improving the load balance.
   * @param fast_approximation By default this is false. When fast_approximation
   * is true, we will not introduce with cold start samples + frequency
   * aggregation and instead we will just concatenate all the text from the
   * strong and weak columns and pass that through the model a single time,
   * costing about a 10-15% drop in zero shot accuracy but ultimately speeding
   * it up by 5-10x.
   */
  void introduceDocuments(const dataset::DataSourcePtr& data,
                          const std::vector<std::string>& strong_column_names,
                          const std::vector<std::string>& weak_column_names,
                          std::optional<uint32_t> num_buckets_to_sample,
                          uint32_t num_random_hashes, bool fast_approximation,
                          bool verbose) {
    licensing::entitlements().verifyDataSource(data);

    _backend->introduceDocuments(data, strong_column_names, weak_column_names,
                                 num_buckets_to_sample, num_random_hashes,
                                 fast_approximation, verbose);
  }

  /**
   * Used in UDTMachClassifier. Introduces a single new document to the model
   * from an in memory map input. Used in conjunction with coldstart.
   */
  void introduceDocument(const MapInput& document,
                         const std::vector<std::string>& strong_column_names,
                         const std::vector<std::string>& weak_column_names,
                         const std::variant<uint32_t, std::string>& new_label,
                         std::optional<uint32_t> num_buckets_to_sample,
                         uint32_t num_random_hashes) {
    licensing::entitlements().verifyFullAccess();

    _backend->introduceDocument(document, strong_column_names,
                                weak_column_names, new_label,
                                num_buckets_to_sample, num_random_hashes);
  }

  /**
   * Used in UDTMachClassifier. Introduces a new label to the model given a
   * batch of representative samples of that label. Uses frequency aggregation
   * based on the outputs of each sample and adds to the internal index.
   */
  void introduceLabel(const MapInputBatch& sample,
                      const std::variant<uint32_t, std::string>& new_label,
                      std::optional<uint32_t> num_buckets_to_sample,
                      uint32_t num_random_hashes) {
    licensing::entitlements().verifyFullAccess();

    _backend->introduceLabel(sample, new_label, num_buckets_to_sample,
                             num_random_hashes);
  }

  /**
   * Used in UDTMachClassifier to forget a given label such that it is
   * impossible to predict in the future.
   */
  void forget(const std::variant<uint32_t, std::string>& label) {
    _backend->forget(label);
  }

  /**
   * Used in UDTMachClassifier. Clears the internal index.
   */
  void clearIndex() { _backend->clearIndex(); }

  /**
   * Used in UDTMachClassifier, assumes each of the samples in the input batch
   * has the target column mapping to space separated strings representing the
   * actual output metaclasses to predict in mach.
   */
  py::object trainWithHashes(const MapInputBatch& batch, float learning_rate,
                             const std::vector<std::string>& metrics) {
    licensing::entitlements().verifyFullAccess();

    return _backend->trainWithHashes(batch, learning_rate, metrics);
  }

  /**
   * Used in UDTMachClassifier, returns the predicted hashes from the input
   * sample. If num_hashes is not provided, will return the number of hashes
   * used in the index by default.
   */
  py::object predictHashes(const MapInput& sample, bool sparse_inference,
                           bool force_non_empty,
                           std::optional<uint32_t> num_hashes) {
    return _backend->predictHashes(sample, sparse_inference, force_non_empty,
                                   num_hashes);
  }

  py::object predictHashesBatch(const MapInputBatch& samples,
                                bool sparse_inference, bool force_non_empty,
                                std::optional<uint32_t> num_hashes) {
    return _backend->predictHashesBatch(samples, sparse_inference,
                                        force_non_empty, num_hashes);
  }

  /**
   * Used for fine tuning in UDTMachClassifier. Predicts the outputs of the
   * target samples and trains the model to map the source samples to those
   * outputs.
   *
   * @param source_target_samples Which source samples to map to which target
   * samples within the same training loop.
   * @param n_buckets The number of target hashes to predict and train with.
   * @param n_association_samples We will replicate each associate sample many
   * times to ensure that this change is reflected in the model.
   * n_association_samples specifies the multiplier for the associate samples.
   * @param n_balancing_samples In order to not overfit the model to only these
   * association samples, we collect balancing samples during training and
   * coldstarting which are indicative of the task that the model was trained
   * on. We will include n_balancing_samples random samples from this collection
   * of data in order to prevent overfitting.
   */
  void associate(
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples,
      uint32_t n_balancing_samples, float learning_rate, uint32_t epochs) {
    _backend->associate(rlhf_samples, n_buckets, n_association_samples,
                        n_balancing_samples, learning_rate, epochs);
  }

  /**
   * Used for fine tuning in UDTMachClassifier. Trains the model to map the
   * source samples to the hashes of the target label.
   *
   * @param source_target_samples Which source samples to map to which target
   * label within the same training loop.
   * @param n_upvote_samples We will replicate each upvote sample many
   * times to ensure that this change is reflected in the model.
   * n_upvote_samples specifies the multiplier for the upvote samples.
   * @param n_balancing_samples In order to not overfit the model to only these
   * association samples, we collect balancing samples during training and
   * coldstarting which are indicative of the task that the model was trained
   * on. We will include n_balancing_samples random samples from this collection
   * of data in order to prevent overfitting.
   */
  void upvote(const std::vector<std::pair<std::string, uint32_t>>&
                  source_target_samples,
              uint32_t n_upvote_samples, uint32_t n_balancing_samples,
              float learning_rate, uint32_t epochs) {
    licensing::entitlements().verifyFullAccess();

    _backend->upvote(source_target_samples, n_upvote_samples,
                     n_balancing_samples, learning_rate, epochs);
  }

  py::object associateTrain(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) {
    licensing::entitlements().verifyDataSource(balancing_data);

    return _backend->associateTrain(balancing_data, rlhf_samples, n_buckets,
                                    n_association_samples, learning_rate,
                                    epochs, metrics, options);
  }

  py::object associateColdStart(
      const dataset::DataSourcePtr& balancing_data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      const std::vector<std::pair<std::string, std::string>>& rlhf_samples,
      uint32_t n_buckets, uint32_t n_association_samples, float learning_rate,
      uint32_t epochs, const std::vector<std::string>& metrics,
      TrainOptions options) {
    licensing::entitlements().verifyDataSource(balancing_data);

    return _backend->associateColdStart(
        balancing_data, strong_column_names, weak_column_names, rlhf_samples,
        n_buckets, n_association_samples, learning_rate, epochs, metrics,
        options);
  }

  /**
   * Tells the model to begin collecting balancing samples from train and cold
   * start calls. Without this enabled the model won't allow RLHF calls (upvote
   * and associate). This can also be specified in the constructor options with
   * "rlhf": true.
   */
  void enableRlhf(uint32_t num_balancing_docs,
                  uint32_t num_balancing_samples_per_doc) {
    _backend->enableRlhf(num_balancing_docs, num_balancing_samples_per_doc);
  }

  /**
   * Gets the internal index for UDTMachClassifier.
   */
  dataset::mach::MachIndexPtr getIndex() {
    licensing::entitlements().verifyFullAccess();

    return _backend->getIndex();
  }

  /**
   * Sets the internal index for UDTMachClassifier.
   */
  void setIndex(const dataset::mach::MachIndexPtr& index) {
    licensing::entitlements().verifyFullAccess();

    _backend->setIndex(index);
  }

  /**
   * Sets the threshold for changing the type of sampling in Mach.
   */
  void setMachSamplingThreshold(float threshold) {
    _backend->setMachSamplingThreshold(threshold);
  }

  /**
   * Determines if the model can support distributed training. By default
   * backends do not support distributed training.
   */
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

  std::unique_ptr<UDTBackend> _backend;
};

}  // namespace thirdai::automl::udt