#include <bolt/src/BoltVector.h>

template <typename BATCH_T>
class Network {
 public:
  /**
   * This function takes in a dataset and training parameters and trains the
   * network for the specified number of epochs with the given parameters. Note
   * that it can be called multiple times to train a network. This function
   * returns a list of the durations (in seconds) of each epoch.
   */
  std::vector<int64_t> train(
      const dataset::InMemoryDataset<BATCH_T>& train_data, float learning_rate,
      uint32_t epochs, uint32_t rehash = 0, uint32_t rebuild = 0);

  /**
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets.
   */
  float predict(const dataset::InMemoryDataset<BATCH_T>& test_data,
                uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

  virtual void forward(uint32_t batch_index, const BATCH_T& input,
                       BoltVector& output) = 0;

  virtual void backward(uint32_t batch_index, const BATCH_T& input,
                        BoltVector& output) = 0;

  virtual void initializeNetworkState(uint32_t batch_size,
                                      bool force_dense) = 0;

  virtual void reBuildHashFunctions() = 0;

  virtual void buildHashTables() = 0;

  virtual void shuffleRandomNeurons() = 0;

  virtual void useSparseInference() = 0;
};