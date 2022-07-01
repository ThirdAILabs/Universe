#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

template <typename BATCH_T>
class Model {
 public:
  Model() : _batch_iter(0), _epoch_count(0) {
    thirdai::licensing::LicenseWrapper::checkLicense();
  }
  /**
   * This function takes in a dataset and training parameters and trains the
   * network for the specified number of epochs with the given parameters. Note
   * that it can be called multiple times to train a network. Returns a map that
   * gives access to the times per epoch and any metrics that were computed
   * during training.
   */
  MetricData train(
      // Train dataset
      std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
      // Train labels
      const dataset::BoltDatasetPtr& train_labels,
      // Loss function to use
      const LossFunction& loss_fn,
      // Learning rate for training
      float learning_rate,
      // Number of training epochs
      uint32_t epochs,
      // Rehash, rebuild parameters for hash functions/tables
      uint32_t rehash = 0, uint32_t rebuild = 0,
      // Metrics to compute during training
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true);

  /**
   * This function takes in a streaming dataset and training parameters and
   * trains the network on the streaming dataset. Note that it can be called
   * multiple times to train a network. Returns a map that gives access to the
   * training time and any metrics that were computed during training.
   */
  MetricData trainOnStream(
      // Train dataset
      std::shared_ptr<dataset::StreamingDataset<BATCH_T>> train_data,
      // Loss function to use
      const LossFunction& loss_fn,
      // Learning rate for training
      float learning_rate,
      // After how many batches to rebuild hash tables
      uint32_t rehash_batch = 20,
      // After how many batches to recreate hash functions
      uint32_t rebuild_batch = 100,
      // Metrics to compute during training
      const std::vector<std::string>& metric_names = {},
      // Interval at which to log metrics when processing stream,
      uint32_t metric_log_batch_interval = 0,
      // Restrict printouts
      bool verbose = true);

  /**
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets. Metrics can be passed in to be computed for
   * the test set, and the function optionally store the activations for the
   * output layer in the output_activations array.
   */
  InferenceMetricData predict(
      // Test dataset
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
      // Test labels
      const dataset::BoltDatasetPtr& labels,
      // Array to store output active neurons in. This should be null if it is
      // not desired for the output values to be returned or if the output is
      // dense.
      uint32_t* output_active_neurons,
      // Array to store output activations in, will not return activations if
      // this is null
      float* output_activations,
      // Use sparse inference
      bool use_sparse_inference = false,
      // Metrics to compute
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true,
      // Limit the number of batches used in the dataset
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

  /**
   * This function takes in a streaming dataset and uses it to evaluate the
   * model. Metrics can be passed in to be computed for the test set, and will
   * be returned by the function. Additionally a callback can optionally be
   * provided that will be called on the output of the model for each batch.
   */
  InferenceMetricData predictOnStream(
      // Test dataset
      const std::shared_ptr<dataset::StreamingDataset<BATCH_T>> test_data,
      // Use sparse inference
      bool use_sparse_inference = false,
      // Metrics to compute
      const std::vector<std::string>& metric_names = {},
      // We choose not to store final layer activations for a streaming dataset
      // as streaming datasets could be large enough that storing all of the
      // activation is not possible and the size of the dataset is not known at
      // the beginning, so instead we provide the ability to have a callback
      // which is called with the output activations after every batch.
      std::optional<std::function<void(const bolt::BoltBatch&, uint32_t)>>
          batch_callback = std::nullopt,
      // Restrict printouts
      bool verbose = true);

  void processTrainingBatch(BATCH_T& batch_inputs, BoltBatch& outputs,
                            const BoltBatch& batch_labels,
                            const LossFunction& loss_fn, float learning_rate,

                            MetricAggregator& metrics);

  void processTestBatch(const BATCH_T& batch_inputs, BoltBatch& outputs,
                        const BoltBatch* batch_labels,
                        uint32_t* output_active_neurons,
                        float* output_activations,
                        uint64_t inference_output_dim,
                        MetricAggregator& metrics, bool compute_metrics);

  void updateSampling(uint32_t rehash_batch, uint32_t rebuild_batch);

  // Computes forward path through the network.
  virtual void forward(uint32_t batch_index, const BATCH_T& input,
                       BoltVector& output, const BoltVector* labels) = 0;

  // Backpropagates gradients through the network
  virtual void backpropagate(uint32_t batch_index, BATCH_T& input,
                             BoltVector& output) = 0;

  // Performs parameter updates for the network.
  virtual void updateParameters(float learning_rate, uint32_t iter) = 0;

  // Called for network to allocate any necessary state to store activations and
  // gradients.
  virtual void initializeNetworkState(uint32_t batch_size,
                                      bool use_sparsity) = 0;

  // Construct new hash functions (primarly for fully connected layers).
  virtual void reBuildHashFunctions() = 0;

  // Rebuild any hash tables (primarly for fully connected layers).
  virtual void buildHashTables() = 0;

  // Allocates storage for activations and gradients for output layer.
  virtual BoltBatch getOutputs(uint32_t batch_size, bool use_sparsity) = 0;

  virtual uint32_t getOutputDim() const = 0;

  // Gets the dimension of the output layer during inference (depends of if
  // sparse inference is enabled).
  virtual uint32_t getInferenceOutputDim(bool using_sparsity) const = 0;

  virtual ~Model() = default;

  /**
   * shallow layer: Layer without optimizer state
   * setShallow sets the layer to shallow or non-shallow, ie, it can remove or
   * initialize the optimizer respectively
   * Only called for trimming the model or for resuming training.
   */
  virtual void setShallow(bool shallow) = 0;

  /**
   * setShallowSave sets whether layer should be saved shallowly, ie, whether
   * layers should be saved with or without the optimizer state
   * Called right before saving the model so that archive method knows whether
   * or not to store the optimizer state.
   */
  virtual void setShallowSave(bool shallow) = 0;

  virtual bool anyLayerShallow() = 0;

 protected:
  uint32_t getRehashBatch(uint32_t rehash, uint32_t batch_size,
                          uint32_t data_len);

  uint32_t getRebuildBatch(uint32_t rebuild, uint32_t batch_size,
                           uint32_t data_len);

  uint32_t _batch_iter;

 private:
  constexpr bool checkBatchInterval(uint32_t num_batches) {
    return (_batch_iter % num_batches) == (num_batches - 1);
  }

  uint32_t _epoch_count;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_epoch_count, _batch_iter);
  }
};

}  // namespace thirdai::bolt
