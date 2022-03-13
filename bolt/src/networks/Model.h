#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/Dataset.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

template <typename BATCH_T>
class Model {
 public:
  Model() : _epoch_count(0), _batch_iter(0), _sparse_inference_enabled(false) {}
  /**
   * This function takes in a dataset and training parameters and trains the
   * network for the specified number of epochs with the given parameters. Note
   * that it can be called multiple times to train a network. Returns a map that
   * gives access to the times per epoch and any metrics that were computed
   * during training.
   */
  std::unordered_map<std::string, std::vector<double>> train(
      // Train dataset
      const dataset::InMemoryDataset<BATCH_T>& train_data,
      // Loss function to use
      const std::string& loss_fn_name,
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
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets. Metrics can be passed in to be computed for
   the test set, and the function optionally store the activations for the
   output layer in the output_activations array.
   */
  void predict(
      // Test dataset
      const dataset::InMemoryDataset<BATCH_T>& test_data,
      // Array to store output activations in, will not return activations if
      // this is null
      float* output_activations,
      // Metrics to compute
      const std::vector<std::string>& metric_names = {},
      // Restrict printouts
      bool verbose = true,
      // Limit the number of batches used in the dataset
      uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

  // Sets flag to enable sparse inference.
  void useSparseInference() {
    _sparse_inference_enabled = true;
    useSparseInferenceImpl();
  }

  // Computes forward path through the network.
  virtual void forward(uint32_t batch_index, BATCH_T& input,
                       BoltVector& output) = 0;

  // Backpropagates gradients through the network
  virtual void backward(uint32_t batch_index, BATCH_T& input,
                        BoltVector& output) = 0;

  // Performs parameter updates for the network.
  virtual void updateParameters(float learning_rate) = 0;

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

  // Any network specific behavior that must be invoked when sparse inference is
  // enabled.
  virtual void useSparseInferenceImpl() = 0;

  // Allocates storage for activations and gradients for output layer.
  virtual BoltBatch getOutputs(bool force_dense) = 0;

  // Gets the dimension of the output layer.
  virtual uint32_t outputDim() = 0;

 private:
  std::unique_ptr<LossFunction> getLossFunction(const std::string& fn_name);

  uint32_t getRehashBatch(uint32_t rehash, uint32_t batch_size,
                          uint32_t data_len);

  uint32_t getRebuildBatch(uint32_t rebuild, uint32_t batch_size,
                           uint32_t data_len);

  uint32_t _epoch_count;

 protected:
  uint32_t _batch_iter;
  bool _sparse_inference_enabled;
};

}  // namespace thirdai::bolt
