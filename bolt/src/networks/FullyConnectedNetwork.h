#pragma once

#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/BoltInputBatch.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class DLRM;

class FullyConnectedNetwork : public Model<dataset::BoltInputBatch> {
  friend class DLRM;

 public:
  FullyConnectedNetwork(std::vector<FullyConnectedLayerConfig> configs,
                        uint32_t input_dim);

  void initializeNetworkState(uint32_t batch_size, bool force_dense) final;

  void forward(uint32_t batch_index, const dataset::BoltInputBatch& inputs,
               BoltVector& output, int layer_no) final {
    forward(batch_index, inputs[batch_index], output,
            &inputs.labels(batch_index), layer_no);
  }

  void backpropagate(uint32_t batch_index, dataset::BoltInputBatch& inputs,
                     BoltVector& output) final {
    backpropagate<true>(batch_index, inputs[batch_index], output);
  }

  void updateParameters(float learning_rate, uint32_t iter) final {
    for (auto& layer : _layers) {
      layer->updateParameters(learning_rate, iter, BETA1, BETA2, EPS);
    }
  }

  void reBuildHashFunctions() final {
    if (_sparse_inference_enabled) {
      return;
    }
    for (auto& layer : _layers) {
      layer->reBuildHashFunction();
    }
  }

  void buildHashTables() final {
    if (_sparse_inference_enabled) {
      return;
    }
    for (auto& layer : _layers) {
      layer->buildHashTables();
    }
  }

  void shuffleRandomNeurons() final {
    if (_sparse_inference_enabled) {
      return;
    }
    for (auto& layer : _layers) {
      layer->shuffleRandNeurons();
    }
  }

  BoltBatch getOutputs(uint32_t batch_size, bool force_dense, int layer_no) final {
    if (layer_no == -1) {
      return _layers.back()->createBatchState(batch_size,
                                            useDenseComputations(force_dense));
    } else {
      return _layers[layer_no]->createBatchState(batch_size,
                                            useDenseComputations(force_dense));
    }
  }

  uint32_t outputDim() const final { return _layers.back()->getDim(); }

  void enableSparseInference() { _layers.back()->forceSparseForInference(); }

  /*
    - Users should be able to set the weights and biases of a particular layer of the network.
    - The format of the weights to be passed in should be clear and well documented.
    - replace the weights buffer with input weights buffer if its of the same size. If fails, should return useful error messages on expected format
  */
  void setWeights(int layer_no, float* weights, int weightsLen) {
    std::copy(_layers[layer_no]->getWeights(), _layers[layer_no]->getWeights() + weightsLen, weights);
  }

  void setBias(int layer_no, float* biases, int biasesLen) {
    std::copy(_layers[layer_no]->getBiases(), _layers[layer_no]->getBiases() + biasesLen, biases);
  }

 private:
  void forward(uint32_t batch_index, const BoltVector& input,
               BoltVector& output, const BoltVector* labels, int layer_no);

  template <bool FROM_INPUT>
  void backpropagate(uint32_t batch_index, BoltVector& input,
                     BoltVector& output);

  bool useDenseComputations(bool force_dense) const {
    return force_dense && !_sparse_inference_enabled;
  }

 protected:
  uint64_t _input_dim;
  std::vector<std::shared_ptr<FullyConnectedLayer>> _layers;
  std::vector<BoltBatch> _states;
  uint32_t _num_layers;
  bool _sparse_inference_enabled;
};

}  // namespace thirdai::bolt