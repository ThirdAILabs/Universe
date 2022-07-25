#pragma once

#include "Model.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/EmbeddingLayer.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/batch_types/ClickThroughBatch.h>
#include <exceptions/src/Exceptions.h>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class DLRM : public Model<dataset::ClickThroughBatch> {
 public:
  DLRM(EmbeddingLayerConfig embedding_config,
       SequentialConfigList bottom_mlp_configs,
       SequentialConfigList top_mlp_configs, uint32_t dense_feature_dim);

  uint32_t getOutputDim() const final { return _top_mlp.getOutputDim(); }

  uint32_t getInferenceOutputDim(bool using_sparsity) const final {
    return _top_mlp.getInferenceOutputDim(using_sparsity);
  }

 private:
  void forward(uint32_t batch_index, const dataset::ClickThroughBatch& inputs,
               BoltVector& output, const BoltVector* labels) final;

  void backpropagate(uint32_t batch_index, dataset::ClickThroughBatch& inputs,
                     BoltVector& output) final;

  void updateParameters(float learning_rate, uint32_t iter) final {
    _bottom_mlp.updateParameters(learning_rate, iter);
    _embedding_layer.updateParameters(learning_rate, iter, BETA1, BETA2, EPS);
    _top_mlp.updateParameters(learning_rate, iter);
  }

  void initializeNetworkState(uint32_t batch_size, bool use_sparsity) final;

  void reBuildHashFunctions() final {
    _bottom_mlp.reBuildHashFunctions();
    _top_mlp.reBuildHashFunctions();
  }

  void buildHashTables() final {
    _bottom_mlp.buildHashTables();
    _top_mlp.buildHashTables();
  }

  BoltBatch getOutputs(uint32_t batch_size, bool use_sparsity) final {
    return _top_mlp.getOutputs(batch_size, use_sparsity);
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