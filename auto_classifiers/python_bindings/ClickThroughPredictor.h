#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/DlrmAttention.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <dataset/src/NumpyDataset.h>
#include <pybind11/pybind11.h>
#include <limits>
#include <stdexcept>

namespace py = pybind11;

namespace thirdai::bolt::python {

using thirdai::dataset::numpy::NumpyArray;

class ClickThroughPredictor {
 public:
  class ClickThroughPredictorConfig {
   public:
    explicit ClickThroughPredictorConfig(const std::string& size) {
      if (size == "small") {
        _log_embedding_block_size = 20;
        _top_mlp_layer_dim = 300;
        _top_mlp_layer_sparsity = 0.4;
      } else if (size == "medium") {
        _log_embedding_block_size = 24;
        _top_mlp_layer_dim = 500;
        _top_mlp_layer_sparsity = 0.4;
      } else if (size == "large") {
        _log_embedding_block_size = 27;
        _top_mlp_layer_dim = 1000;
        _top_mlp_layer_sparsity = 0.3;
      } else {
        throw std::invalid_argument(
            "Invalid size parameter '" + size +
            "'. Please use 'small', 'medium', or 'large'.");
      }
    }

    uint32_t logEmbeddingBlockSize() const { return _log_embedding_block_size; }

    uint32_t topMlpLayerDim() const { return _top_mlp_layer_dim; }

    float topMlpLayerSparsity() const { return _top_mlp_layer_sparsity; }

   private:
    uint32_t _log_embedding_block_size;
    uint32_t _top_mlp_layer_dim;
    float _top_mlp_layer_sparsity;
  };

  ClickThroughPredictor(const std::string& size, uint32_t num_dense_features,
                        uint32_t num_categorical_features) {
    ClickThroughPredictorConfig config(size);

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
        /* log_embedding_block_size= */ config.logEmbeddingBlockSize(),
        /* reduction= */ "concatenation",
        /* num_tokens_per_input= */ num_categorical_features);
    embedding->addInput(categorical_input);

    auto feature_interaction =
        DlrmAttentionNode::make()->setPredecessors(bottom_layer, embedding);

    NodePtr top_mlp_output = ConcatenateNode::make()->setConcatenatedNodes(
        {bottom_layer, feature_interaction});

    for (uint32_t i = 0; i < 3; i++) {
      top_mlp_output =
          FullyConnectedNode::make(
              /* dim= */ config.topMlpLayerDim(),
              /* sparsity= */ config.topMlpLayerSparsity(),
              /* activation= */ "relu",
              /* sampling_config= */ std::make_shared<RandomSamplingConfig>())
              ->addPredecessor(top_mlp_output);
    }

    auto output = FullyConnectedNode::makeDense(/* dim= */ 1,
                                                /* activation= */ "sigmoid");
    output->addPredecessor(top_mlp_output);

    _model = std::make_shared<BoltGraph>(
        /* inputs= */ std::vector<InputPtr>{dense_input, categorical_input},
        /* output= */ output);
    _model->compile(std::make_shared<BinaryCrossEntropyLoss>());
  }

  void train(const NumpyArray<float>& dense_features,
             const NumpyArray<uint32_t>& categorical_features,
             const NumpyArray<float>& labels, uint32_t epochs,
             float learning_rate, uint32_t batch_size) {
    auto dense_dataset = dataset::numpy::denseNumpyToBoltVectorDataset(
        dense_features, batch_size);

    auto categorical_dataset = dataset::numpy::numpyTokensToBoltDataset(
        categorical_features, batch_size);

    auto labels_dataset =
        dataset::numpy::denseNumpyToBoltVectorDataset(labels, batch_size);

    _model->train({dense_dataset, categorical_dataset}, labels_dataset,
                  TrainConfig::makeConfig(learning_rate, epochs));
  }

  NumpyArray<float> evaluate(const NumpyArray<float>& dense_features,
                             const NumpyArray<uint32_t>& categorical_features) {
    auto dense_dataset = dataset::numpy::denseNumpyToBoltVectorDataset(
        dense_features, /* batch_size= */ 2048);

    auto categorical_dataset = dataset::numpy::numpyTokensToBoltDataset(
        categorical_features, /* batch_size= */ 2048);

    auto [metrics, output] =
        _model->predict({dense_dataset, categorical_dataset}, nullptr,
                        PredictConfig::makeConfig().returnActivations());

    uint32_t num_samples = output.numSamples();
    const float* activations = output.getNonowningActivationPointer();
    py::object activation_handle = py::cast(std::move(output));

    return NumpyArray<float>({num_samples}, {sizeof(float)}, activations,
                             activation_handle);
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

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<ClickThroughPredictor> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<ClickThroughPredictor> deserialize_into(
        new ClickThroughPredictor());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  // Private constructor for cereal.
  ClickThroughPredictor() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model);
  }

  BoltGraphPtr _model;
};

}  // namespace thirdai::bolt::python