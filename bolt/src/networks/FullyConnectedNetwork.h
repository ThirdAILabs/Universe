#pragma once

#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "Model.h"
#include "DistributedModel.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/SequentialLayer.h>
#include <dataset/src/Dataset.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class DLRM;

class FullyConnectedNetwork : public Model<bolt::BoltBatch>, public DistributedModel<bolt::BoltBatch> {
  friend class DLRM;

 public:
  FullyConnectedNetwork(SequentialConfigList configs, uint32_t input_dim, bool is_distributed=false);

  void initializeNetworkState(uint32_t batch_size, bool force_dense) final;

  void forward(uint32_t batch_index, const bolt::BoltBatch& inputs,
               BoltVector& output, const BoltVector* labels) final {
    forward(batch_index, inputs[batch_index], output, labels);
  }

  void backpropagate(uint32_t batch_index, bolt::BoltBatch& inputs,
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


  void buildNetworkSummary(std::stringstream& summary,
                           bool detailed = false) const {
    summary << "========= Bolt Network =========\n";
    summary << "InputLayer (Layer 0): dim=" << _input_dim << "\n";
    uint32_t layerNum = 1;
    for (const auto& layer : _layers) {
      summary << "FullyConnectedLayer (Layer " << layerNum << "): ";
      layer->buildLayerSummary(summary, detailed);
      ++layerNum;
    }
    summary << "================================";
  }

  void printSummary(bool detailed = false) const {
    std::stringstream summary;
    this->buildNetworkSummary(summary, detailed);
    std::cout << summary.str() << std::endl;
  }

  BoltBatch getOutputs(uint32_t batch_size, bool force_dense) final {
    return _layers.back()->createBatchState(batch_size,
                                            useDenseComputations(force_dense));
  }

  uint32_t getOutputDim() const final { return _layers.back()->getDim(); }

  uint32_t getInferenceOutputDim() const final {
    return _layers.back()->getInferenceOutputDim();
  }

  bool anyLayerShallow() final {
    bool shallow = false;
    for (uint32_t i = 0; i < _num_layers; i++) {
      shallow |= _layers[i]->isShallow();
    }
    return shallow;
  }

  void setShallow(bool shallow) final {
    for (uint32_t i = 0; i < _num_layers; i++) {
      _layers[i]->setShallow(shallow);
    }
  }

  void setShallowSave(bool shallow) final {
    for (uint32_t i = 0; i < _num_layers; i++) {
      _layers[i]->setShallowSave(shallow);
    }
  }

  uint32_t getInputDim() const { return _layers.front()->getInputDim(); }

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
  std::vector<std::shared_ptr<SequentialLayer>> _layers;
  std::vector<BoltBatch> _states;
  uint32_t _num_layers;
  bool _sparse_inference_enabled;
  bool _is_distributed;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Model<bolt::BoltBatch>>(this), _input_dim,
            _layers, _num_layers, _sparse_inference_enabled);
  }

 protected:
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  FullyConnectedNetwork(){};
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::FullyConnectedNetwork)
