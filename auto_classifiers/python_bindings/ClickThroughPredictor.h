#pragma once

#include <bolt/python_bindings/ConversionUtils.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/DlrmAttention.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <dataset/src/NumpyDataset.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::python {

using thirdai::dataset::numpy::NumpyArray;

class ClickThroughPredictor {
 public:
  ClickThroughPredictor(uint32_t num_dense_features,
                        uint32_t num_categorical_features) {
    auto dense_input = Input::make(num_dense_features);
    auto bottom_layer =
        FullyConnectedNode::makeDense(/* dim= */ 32, /* activation =*/"relu");
    bottom_layer->addPredecessor(dense_input);

    auto categorical_input = Input::makeTokenInput(
        std::numeric_limits<uint32_t>::max(),
        std::pair<uint32_t, uint32_t>{num_categorical_features,
                                      num_categorical_features});

    auto embedding = EmbeddingNode::make(
        /* num_embedding_lookups= */ 8, /* lookup_size= */ 4,
        /* log_embedding_block_size= */ 20, /* reduction= */ "concatenate");
    embedding->addInput(categorical_input);

    auto feature_interaction =
        DlrmAttentionNode::make()->setPredecessors(bottom_layer, embedding);

    NodePtr top_mlp_output = ConcatenateNode::make()->setConcatenatedNodes(
        {bottom_layer, feature_interaction});

    for (uint32_t i = 0; i < 3; i++) {
      top_mlp_output =
          FullyConnectedNode::make(
              /* dim= */ 500, /* sparsity= */ 0.4, /* activation= */ "relu",
              /* sampling_config= */ std::make_shared<RandomSamplingConfig>())
              ->addPredecessor(top_mlp_output);
    }

    auto output = FullyConnectedNode::makeDense(/* dim= */ 2,
                                                /* activation= */ "softmax");
    output->addPredecessor(top_mlp_output);

    _model = std::make_shared<BoltGraph>(
        /* inputs= */ std::vector<InputPtr>{dense_input, categorical_input},
        /* output= */ output);
    _model->compile(std::make_shared<CategoricalCrossEntropyLoss>());
  }

  void train(const NumpyArray<float>& dense_features,
             const NumpyArray<uint32_t>& categorical_features,
             const NumpyArray<uint32_t>& labels, uint32_t epochs,
             float learning_rate, uint32_t batch_size) {
    auto dense_dataset =
        dataset::numpy::numpyToBoltVectorDataset(dense_features, batch_size);

    auto categorical_dataset = dataset::numpy::numpyToBoltVectorDataset(
        categorical_features, batch_size);

    auto labels_dataset =
        dataset::numpy::numpyToBoltVectorDataset(labels, batch_size);

    _model->train({dense_dataset, categorical_dataset}, labels_dataset,
                  TrainConfig::makeConfig(learning_rate, epochs));
  }

  py::tuple evaluate(const NumpyArray<float>& dense_features,
                     const NumpyArray<uint32_t>& categorical_features) {
    auto dense_dataset = dataset::numpy::numpyToBoltVectorDataset(
        dense_features, /* batch_size= */ 2048);

    auto categorical_dataset = dataset::numpy::numpyToBoltVectorDataset(
        categorical_features, /* batch_size= */ 2048);

    auto [metrics, output] =
        _model->predict({dense_dataset, categorical_dataset}, nullptr,
                        PredictConfig::makeConfig().returnActivations());

    // This returns a tuple of (metrics, activations) since the output is dense,
    // we are only interested in the activations here.
    return constructNumpyActivationsArrays(metrics, output)[1];
  }

  float predict(const NumpyArray<float>& dense_features,
                const NumpyArray<uint32_t>& categorical_features) {
    if (dense_features.ndim() != 1) {
      throw std::invalid_argument(
          "Expected dense features to be 1D array in predict.");
    }
    if (categorical_features.ndim() != 1) {
      throw std::invalid_argument(
          "Expected categorical features to be 1D array in predict.");
    }

    BoltVector dense_input(dense_features.shape(0), /* is-dense= */ true,
                           /* has_gradient= */ false);
    std::copy(dense_features.data(),
              dense_features.data() + dense_features.shape(0),
              dense_input.activations);

    BoltVector categorical_input(categorical_features.shape(0),
                                 /* is_dense= */ false,
                                 /* has_gradient= */ false);
    std::copy(categorical_features.data(),
              categorical_features.data() + categorical_features.shape(0),
              categorical_input.active_neurons);
    std::fill_n(categorical_input.activations, categorical_features.shape(0),
                1.0);

    BoltVector output = _model->predictSingle(
        {dense_input, categorical_input}, /* use_sparse_inference= */ false);

    return output.activations[0];
  }

 private:
  BoltGraphPtr _model;
};

}  // namespace thirdai::bolt::python