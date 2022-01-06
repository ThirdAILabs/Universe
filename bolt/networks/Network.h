#pragma once

#include <bolt/layers/FullyConnectedLayer.h>
#include <dataset/src/Dataset.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Network {
 public:
  Network(std::vector<FullyConnectedLayerConfig> configs, uint64_t input_dim);

  /**
   * This function takes in a dataset and training parameters and trains the
   * network for the specified number of epochs with the given parameters. Note
   * that it can be called multiple times to train a network. This function
   * returns a list of the durations (in seconds) of each epoch.
   */
  std::vector<int64_t> train(
      const dataset::InMemoryDataset<dataset::SparseBatch>& train_data,
      float learning_rate, uint32_t epochs, uint32_t rehash = 0,
      uint32_t rebuild = 0);

  /**
   * This function takes in a test dataset and uses it to evaluate the model. It
   * returns the final accuracy. The batch_limit parameter limits the number of
   * test batches used, this is intended for intermediate accuracy checks during
   * training with large datasets.
   */
  float test(const dataset::InMemoryDataset<dataset::SparseBatch>& test_data,
             uint32_t batch_limit = std::numeric_limits<uint32_t>::max());

  void forward(uint32_t batch_index, const VectorState& input,
               VectorState& output, const uint32_t* labels, uint32_t label_len);

  template <bool FROM_INPUT>
  void backpropagate(uint32_t batch_index, VectorState& input,
                     VectorState& output);

  void reBuildHashFunctions();

  void buildHashTables();

  void freezeSelectionForInference() {_freeze_Selection_Inference = true;};

  uint32_t getNumLayers() const { return _num_layers; }

  uint32_t getInputDim() const { return _input_dim; }

  std::vector<uint32_t> getLayerSizes() {
    std::vector<uint32_t> layer_sizes;
    for (const auto& c : _configs) {
      layer_sizes.push_back(c.dim);
    }
    return layer_sizes;
  }

  std::vector<std::string> getActivationFunctions() {
    std::vector<std::string> funcs;
    for (const auto& c : _configs) {
      switch (c.act_func) {
        case ActivationFunc::ReLU:
          funcs.emplace_back("ReLU");
          break;

        case ActivationFunc::Softmax:
          funcs.emplace_back("Softmax");
          break;

        case ActivationFunc::MeanSquared:
          funcs.emplace_back("MeanSquared");
          break;
      }
    }
    return funcs;
  }

  ~Network();

 protected:
  std::vector<FullyConnectedLayerConfig> _configs;
  uint64_t _input_dim;
  FullyConnectedLayer** _layers;
  BatchState* _states;
  uint32_t _num_layers;
  uint32_t _iter;
  uint32_t _epoch_count;

  bool _freeze_Selection_Inference;
};

}  // namespace thirdai::bolt