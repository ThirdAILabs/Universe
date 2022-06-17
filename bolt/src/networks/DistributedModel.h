#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/vector.hpp>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/metrics/Metric.h>
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
class DistributedModel {
 public:
 BoltBatch _outputs;

  DistributedModel() : _batch_iter(0), _epoch_count(0) {
    thirdai::licensing::LicenseWrapper::checkLicense();
  }


  /**
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets. Metrics can be passed in to be computed for
   * the test set, and the function optionally store the activations for the
   * output layer in the output_activations array.
   */
  InferenceMetricData predictDistributed(
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
  InferenceMetricData predictOnStreamDistributed(
      // Test dataset
      const std::shared_ptr<dataset::StreamingDataset<BATCH_T>>& test_data,
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


  void processTestBatchDistributed(const BATCH_T& batch_inputs, BoltBatch& outputs,
                        const BoltBatch* batch_labels,
                        uint32_t* output_active_neurons,
                        float* output_activations, MetricAggregator& metrics,
                        bool compute_metrics);
  //Distributed Functions
  void initTrainDistributed(
    std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& train_data,
    const dataset::BoltDatasetPtr& train_labels,
    // Clang tidy is disabled for this line because it wants to pass by
    // reference, but shared_ptrs should not be passed by reference
    uint32_t rehash, uint32_t rebuild, bool verbose);

    void calculateGradientDistributed(
      uint32_t batch,
    const LossFunction& loss_fn);

    void updateParametersDistributed(
   float learning_rate);


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
                                      bool force_dense) = 0;

  // Construct new hash functions (primarly for fully connected layers).
  virtual void reBuildHashFunctions() = 0;

  // Rebuild any hash tables (primarly for fully connected layers).
  virtual void buildHashTables() = 0;

  // Shuffles neurons for random sampling.
  virtual void shuffleRandomNeurons() = 0;

  // Allocates storage for activations and gradients for output layer.
  virtual BoltBatch getOutputs(uint32_t batch_size, bool force_dense) = 0;

  virtual uint32_t getOutputDim() const = 0;

  // Gets the dimension of the output layer during inference (depends of if
  // sparse inference is enabled).
  virtual uint32_t getInferenceOutputDim() const = 0;

  virtual ~DistributedModel() = default;

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
  uint32_t getRehashBatchDistributed(uint32_t rehash, uint32_t batch_size,
                          uint32_t data_len);

  uint32_t getRebuildBatchDistributed(uint32_t rebuild, uint32_t batch_size,
                           uint32_t data_len);

  uint32_t _batch_iter;

 private:
  uint32_t _epoch_count;
  uint32_t _rebuild_batch;
  uint32_t _rehash_batch;
  std::shared_ptr<dataset::InMemoryDataset<BATCH_T>> _train_data;
  const dataset::BoltDatasetPtr _train_labels;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_epoch_count, _batch_iter);
  }
};

}  // namespace thirdai::bolt
