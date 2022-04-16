#pragma once

#include <wrappers/src/LicenseWrapper.h>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
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
               BoltVector& output, bool train) final {
    forward(batch_index, inputs[batch_index], output,
            train ? &inputs.labels(batch_index) : nullptr);
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

  BoltBatch getOutputs(uint32_t batch_size, bool force_dense) final {
    return _layers.back()->createBatchState(batch_size,
                                            useDenseComputations(force_dense));
  }

  uint32_t outputDim() const final { return _layers.back()->getDim(); }

  void enableSparseInference() {
    _sparse_inference_enabled = true;
    _layers.back()->forceSparseForInference();
  }

 private:
  void forward(uint32_t batch_index, const BoltVector& input,
               BoltVector& output, const BoltVector* labels);

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

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Model<dataset::BoltInputBatch>>(this),
            _input_dim, _layers, _num_layers, _sparse_inference_enabled);
  }

 protected:
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  FullyConnectedNetwork() {
    thirdai::licensing::LicenseWrapper::checkLicense();
  };
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedNetwork)
