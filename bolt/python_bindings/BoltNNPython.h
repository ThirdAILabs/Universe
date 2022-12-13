#pragma once

#include "PybindUtils.h"
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <compression/python_bindings/ConversionUtils.h>
#include <compression/src/CompressionFactory.h>
#include <dataset/src/Datasets.h>
#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <variant>

namespace py = pybind11;

namespace thirdai::bolt::python {

void createBoltNNSubmodule(py::module_& bolt_submodule);

void createLossesSubmodule(py::module_& nn_submodule);

py::tuple dagEvaluatePythonWrapper(BoltGraph& model,
                                   const dataset::BoltDatasetList& data,
                                   const dataset::BoltDatasetPtr& labels,
                                   const EvalConfig& eval_config);

py::tuple dagGetInputGradientSingleWrapper(
    const std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>&
        gradients);
using ParameterArray =
    py::array_t<float, py::array::c_style | py::array::forcecast>;

using SerializedCompressedVector =
    py::array_t<char, py::array::c_style | py::array::forcecast>;

static uint64_t dimensionProduct(const std::vector<uint32_t>& dimensions) {
  uint64_t product = 1;
  for (uint32_t dim : dimensions) {
    product *= dim;
  }
  return product;
}

class ParameterReference {
  using FloatCompressedVector =
      std::variant<thirdai::compression::DragonVector<float>,
                   thirdai::compression::CountSketch<float>>;

 public:
  ParameterReference(float* params, const std::vector<uint32_t>& dimensions)
      : _params(params),
        _dimensions(dimensions),
        _total_dim(dimensionProduct(dimensions)) {}

  ParameterArray copy() const {
    float* params_copy = new float[_total_dim];
    std::copy(_params, _params + _total_dim, params_copy);

    py::capsule free_when_done(
        params_copy, [](void* ptr) { delete static_cast<float*>(ptr); });

    return ParameterArray(_dimensions, params_copy, free_when_done);
  }

  ParameterArray get() const { return ParameterArray(_dimensions, _params); }

  void set(py::array_t<char, py::array::c_style>& new_params) {
    const char* serialized_data =
        py::cast<SerializedCompressedVector>(new_params).data();
    FloatCompressedVector compressed_vector =
        thirdai::compression::python::deserializeCompressedVector<float>(
            serialized_data);
    std::vector<float> full_gradients = std::visit(
        thirdai::compression::DecompressVisitor<float>(), compressed_vector);

    if (full_gradients.size() != dimensionProduct(_dimensions)) {
      throw std::length_error(
          "The sizes of the decompressed vector and parameter reference are "
          "different. Either the compressed vector has been corrupted or you "
          "are trying to set the wrong parameter reference.");
    }

    // TODO(Shubh): Pass in a refernce to _params and avoid std::copy
    std::copy(full_gradients.data(), full_gradients.data() + _total_dim,
              _params);
  }

  void set(py::array_t<float, py::array::c_style | py::array::forcecast>&
               new_params) {
    ParameterArray new_array = py::cast<ParameterArray>(new_params);
    checkNumpyArrayDimensions(_dimensions, new_params);
    std::copy(new_array.data(), new_array.data() + _total_dim, _params);
  }

  SerializedCompressedVector compress(const std::string& compression_scheme,
                                      float compression_density,
                                      uint32_t seed_for_hashing,
                                      uint32_t sample_population_size) {
    FloatCompressedVector compressed_vector = thirdai::compression::compress(
        _params, static_cast<uint32_t>(_total_dim), compression_scheme,
        compression_density, seed_for_hashing, sample_population_size);

    uint32_t serialized_size = std::visit(
        thirdai::compression::SizeVisitor<float>(), compressed_vector);

    char* serialized_compressed_vector = new char[serialized_size];

    std::visit(thirdai::compression::SerializeVisitor<float>(
                   serialized_compressed_vector),
               compressed_vector);

    py::capsule free_when_done(serialized_compressed_vector, [](void* ptr) {
      delete static_cast<char*>(ptr);
    });

    return SerializedCompressedVector(
        serialized_size, serialized_compressed_vector, free_when_done);
  }

  static SerializedCompressedVector concat(
      const py::object& py_compressed_vectors) {
    std::vector<FloatCompressedVector> compressed_vectors =
        thirdai::compression::python::convertPyListToCompressedVectors<float>(
            py_compressed_vectors);
    FloatCompressedVector concatenated_compressed_vector =
        thirdai::compression::concat(std::move(compressed_vectors));

    uint32_t serialized_size =
        std::visit(thirdai::compression::SizeVisitor<float>(),
                   concatenated_compressed_vector);

    char* serialized_compressed_vector = new char[serialized_size];

    std::visit(thirdai::compression::SerializeVisitor<float>(
                   serialized_compressed_vector),
               concatenated_compressed_vector);

    py::capsule free_when_done(serialized_compressed_vector, [](void* ptr) {
      delete static_cast<char*>(ptr);
    });

    return SerializedCompressedVector(
        serialized_size, serialized_compressed_vector, free_when_done);
  }

 private:
  float* _params;
  std::vector<uint32_t> _dimensions;
  uint64_t _total_dim;
};
class GradientReference {
  /**
   * This class implements gradient references, which return flattened
   * gradients. The gradients are flattened and stored as
   * <node_1_bias><node_1_weights><node_2_bias><node_2_weights>. It
   * implements two function get_gradients and set_gradients. It
   * flattens gradients for all the nodes which have needGradientSharing true.
   * Else, would raise an error stating that Gradient Flattening Logic not
   * added.
   *
   * get_gradients: It returns the gradients as a flattened ParameterArray
   *
   * set_gradients: It set the gradients provided as an argument to the bolt
   * model.
   */
 public:
  GradientReference(BoltGraph& model) : _model(model) {
    std::vector<NodePtr> nodes = model.getNodes();

    uint64_t flattened_gradients_dim = 0;
    for (NodePtr node : nodes) {
      if (node->needGradientSharing()) {
        if (node->type() == "embedding") {
          EmbeddingNode* embedding_node =
              dynamic_cast<EmbeddingNode*>(node.get());
          flattened_gradients_dim += static_cast<uint32_t>(
              embedding_node->getRawEmbeddingBlockGradient().size());
        } else if (node->type() == "fc") {
          FullyConnectedNode* fc_node =
              dynamic_cast<FullyConnectedNode*>(node.get());
          flattened_gradients_dim += dimensionProduct({fc_node->outputDim()});
          flattened_gradients_dim +=
              dimensionProduct({fc_node->outputDim(),
                                fc_node->getPredecessors()[0]->outputDim()});
        } else {
          std::string err =
              "Gradient sharing logic is not implemented for " + node->type();
          throw std::invalid_argument(err);
        }
      }
    }
    _flattened_gradients_dim = flattened_gradients_dim;
  }

  ParameterArray get_gradients() {
    std::vector<NodePtr> nodes = _model.getNodes();

    float* param_copy = new float[_flattened_gradients_dim];
    uint64_t node_gradient_pointer = 0;
    for (NodePtr node : nodes) {
      if (node->needGradientSharing()) {
        if (node->type() == "embedding") {
          EmbeddingNode* embedding_node =
              dynamic_cast<EmbeddingNode*>(node.get());
          std::vector<float>& raw_embedding_block_gradient =
              embedding_node->getRawEmbeddingBlockGradient();
          uint32_t embedding_layer_length =
              static_cast<uint32_t>(raw_embedding_block_gradient.size());
          std::copy(
              raw_embedding_block_gradient.data(),
              raw_embedding_block_gradient.data() + embedding_layer_length,
              param_copy + node_gradient_pointer);
          node_gradient_pointer += embedding_layer_length;
        } else if (node->type() == "fc") {
          FullyConnectedNode* fc_node =
              dynamic_cast<FullyConnectedNode*>(node.get());
          uint64_t flattened_node_bias_len =
              dimensionProduct({fc_node->outputDim()});
          std::copy(fc_node->getBiasGradientsPtr(),
                    fc_node->getBiasGradientsPtr() + flattened_node_bias_len,
                    param_copy + node_gradient_pointer);
          node_gradient_pointer += flattened_node_bias_len;

          uint64_t flattened_node_weight_len =
              dimensionProduct({fc_node->outputDim(),
                                fc_node->getPredecessors()[0]->outputDim()});
          std::copy(
              fc_node->getWeightGradientsPtr(),
              fc_node->getWeightGradientsPtr() + flattened_node_weight_len,
              param_copy + node_gradient_pointer);
          node_gradient_pointer += flattened_node_weight_len;
        } else {
          std::string err =
              "Gradient sharing logic is not implemented for " + node->type();
          throw std::invalid_argument(err);
        }
      }
    }

    py::capsule free_when_done(
        param_copy, [](void* ptr) { delete static_cast<float*>(ptr); });

    return ParameterArray(_flattened_gradients_dim, param_copy, free_when_done);
  }

  void set_gradients(ParameterArray& new_params) {
    std::vector<NodePtr> nodes = _model.getNodes();
    uint32_t node_gradient_pointer = 0;
    for (NodePtr node : nodes) {
      if (node->needGradientSharing()) {
        if (node->type() == "embedding") {
          EmbeddingNode* embedding_node =
              dynamic_cast<EmbeddingNode*>(node.get());
          std::vector<float>& raw_embedding_block_gradient =
              embedding_node->getRawEmbeddingBlockGradient();
          uint32_t embedding_layer_length =
              static_cast<uint32_t>(raw_embedding_block_gradient.size());
          std::copy(new_params.data() + node_gradient_pointer,
                    new_params.data() + node_gradient_pointer +
                        embedding_layer_length,
                    raw_embedding_block_gradient.data());
          node_gradient_pointer += embedding_layer_length;
        } else if (node->type() == "fc") {
          FullyConnectedNode* fc_node =
              dynamic_cast<FullyConnectedNode*>(node.get());
          uint64_t flattened_node_bias_len =
              dimensionProduct({fc_node->outputDim()});
          std::copy(new_params.data() + node_gradient_pointer,
                    new_params.data() + node_gradient_pointer +
                        flattened_node_bias_len,
                    fc_node->getBiasGradientsPtr());
          node_gradient_pointer += flattened_node_bias_len;

          uint64_t flattened_node_weight_len =
              dimensionProduct({fc_node->outputDim(),
                                fc_node->getPredecessors()[0]->outputDim()});
          std::copy(new_params.data() + node_gradient_pointer,
                    new_params.data() + node_gradient_pointer +
                        flattened_node_weight_len,
                    fc_node->getWeightGradientsPtr());
          node_gradient_pointer += flattened_node_weight_len;
        } else {
          std::string err =
              "Gradient sharing logic is not implemented for " + node->type();
          throw std::invalid_argument(err);
        }
      }
    }
  }
  BoltGraph& _model;
  uint64_t _flattened_gradients_dim;
};
}  // namespace thirdai::bolt::python
