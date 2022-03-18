#pragma once

#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <vector>

namespace thirdai::bolt {

class DLRM final : public Model<dataset::ClickThroughBatch> {
 public:
  DLRM(EmbeddingLayerConfig embedding_config,
       std::vector<FullyConnectedLayerConfig> bottom_mlp_configs,
       std::vector<FullyConnectedLayerConfig> top_mlp_configs,
       uint32_t dense_feature_dim);

 private:
  void forward(uint32_t batch_index, const dataset::ClickThroughBatch& inputs,
               BoltVector& output) final;

  void backpropagate(uint32_t batch_index, dataset::ClickThroughBatch& inputs,
                     BoltVector& output) final;

  void updateParameters(float learning_rate, uint32_t iter) final {
    _bottom_mlp.updateParameters(learning_rate, iter);
    _embedding_layer.updateParameters(learning_rate, iter, BETA1, BETA2, EPS);
    _top_mlp.updateParameters(learning_rate, iter);
  }

  void initializeNetworkState(uint32_t batch_size, bool force_dense) final;

  void shuffleRandomNeurons() final {
    _bottom_mlp.shuffleRandomNeurons();
    _top_mlp.shuffleRandomNeurons();
  }

  void reBuildHashFunctions() final {
    _bottom_mlp.reBuildHashFunctions();
    _top_mlp.reBuildHashFunctions();
  }

  void buildHashTables() final {
    _bottom_mlp.buildHashTables();
    _top_mlp.buildHashTables();
  }

  uint32_t outputDim() const final { return _top_mlp.outputDim(); }

  BoltBatch getOutputs(uint32_t batch_size, bool force_dense) final {
    return _top_mlp.getOutputs(batch_size, force_dense);
  }

  EmbeddingLayer _embedding_layer;
  FullyConnectedNetwork _bottom_mlp;
  FullyConnectedNetwork _top_mlp;

  uint32_t _concat_layer_dim;
  BoltBatch _concat_layer_state;
  std::vector<BoltVector> _embedding_layer_output;
  std::vector<BoltVector> _bottom_mlp_output;

  uint32_t _iter;
  uint32_t _epoch_count;

  bool _softmax;

 protected:
  uint32_t _output_dim;
};

}  // namespace thirdai::bolt