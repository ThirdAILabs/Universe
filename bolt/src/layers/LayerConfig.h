#pragma once

#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include "LayerUtils.h"
#include "SequentialLayer.h"
#include <cmath>
#include <fstream>
#include <iostream>

namespace thirdai::bolt {

class SequentialLayerConfig {
 public:
  virtual std::shared_ptr<SequentialLayer> createLayer(
      const std::shared_ptr<SequentialLayerConfig>& prev_config,
      const std::shared_ptr<SequentialLayerConfig>& next_config) {
    (void)prev_config;
    (void)next_config;
    throw std::invalid_argument("createLayer() not implemented");
  }

  virtual uint64_t getDim() const {
    throw std::invalid_argument("getDim() not implemented");
  }

  virtual float getSparsity() const {
    throw std::invalid_argument("getSparsity() not implemented");
  }

  virtual ActivationFunction getActFunc() const {
    throw std::invalid_argument("getActFunc() not implemented");
  }

  // virtual void print(std::ostream& out) const {
  //   throw std::invalid_argument("print() not implemented");
  // }

  static void checkSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }
};

// Not meant to be used for anything. Simply implemented to make creating layers
// easier
// TODO(david): refactor to support multidimensional input
class InputConfig final : public SequentialLayerConfig {
 public:
  InputConfig(uint64_t _input_dim) { input_dim = _input_dim; }

  uint64_t getDim() const { return input_dim; }

 private:
  uint64_t input_dim;
};

class FullyConnectedLayerConfig final : public SequentialLayerConfig {
 public:
  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func,
                            SamplingConfig _config)
      : dim(_dim),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config) {
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t _dim, ActivationFunction _act_func)
      : dim(_dim),
        sparsity(1.0),
        act_func(_act_func),
        sampling_config(SamplingConfig()) {
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t _dim, float _sparsity,
                            ActivationFunction _act_func)
      : dim(_dim), sparsity(_sparsity), act_func(_act_func) {
    checkSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t rp = (static_cast<uint32_t>(log2(dim)) / 3) * 3;
      uint32_t k = rp / 3;
      uint32_t rs = (dim * 4) / (1 << rp);
      uint32_t l = sparsity < 0.1 ? 256 : 64;
      sampling_config = SamplingConfig(k, l, rp, rs);
    } else {
      sampling_config = SamplingConfig();
    }
  }

  std::shared_ptr<SequentialLayer> createLayer(
      const std::shared_ptr<SequentialLayerConfig>& prev_config,
      const std::shared_ptr<SequentialLayerConfig>& next_config) {
    if (next_config != nullptr && act_func == ActivationFunction::Softmax) {
      throw std::invalid_argument(
          "Softmax activation function is not supported for hidden layers.");
    }
    return createLayerHelper(prev_config->getDim());
  }

  uint64_t getDim() const { return dim; }

  float getSparsity() const { return sparsity; }

  ActivationFunction getActFunc() const { return act_func; }

  friend std::ostream& operator<<(std::ostream& out,
                                  const FullyConnectedLayerConfig& config) {
    out << "Layer: dim=" << config.dim << ", load_factor=" << config.sparsity;
    switch (config.act_func) {
      case ActivationFunction::ReLU:
        out << ", act_func=ReLU";
        break;
      case ActivationFunction::Softmax:
        out << ", act_func=Softmax";
        break;
      case ActivationFunction::Linear:
        out << ", act_func=Linear";
        break;
    }
    if (config.sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << config.sampling_config.hashes_per_table
          << ", num_tables=" << config.sampling_config.num_tables
          << ", range_pow=" << config.sampling_config.range_pow
          << ", reservoir_size=" << config.sampling_config.reservoir_size
          << "}";
    }
    return out;
  }

 private:
  uint64_t dim;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;

  std::shared_ptr<SequentialLayer> createLayerHelper(uint64_t prev_dim) {
    return std::static_pointer_cast<SequentialLayer>(
        std::make_shared<FullyConnectedLayer>(dim, sparsity, act_func,
                                              sampling_config, prev_dim));
  }
};

class ConvLayerConfig final : public SequentialLayerConfig {
 public:
  ConvLayerConfig(uint64_t _num_filters, float _sparsity,
                  ActivationFunction _act_func, SamplingConfig _config,
                  std::pair<uint32_t, uint32_t> _kernel_size,
                  uint32_t _num_patches)
      : num_filters(_num_filters),
        sparsity(_sparsity),
        act_func(_act_func),
        sampling_config(_config),
        kernel_size(std::move(_kernel_size)),
        num_patches(_num_patches) {
    checkSparsity(sparsity);
  }

  std::shared_ptr<SequentialLayer> createLayer(
      const std::shared_ptr<SequentialLayerConfig>& prev_config,
      const std::shared_ptr<SequentialLayerConfig>& next_config) {
    if (next_config == nullptr)
      throw std::invalid_argument("ConvLayer not supported as final layer.");
    if (act_func != ActivationFunction::ReLU)
      throw std::invalid_argument(
          "Conv layers currently support only ReLU Activation.");
    if (kernel_size.first != kernel_size.second) {
      throw std::invalid_argument(
          "Conv layers currently support only square kernels.");
    }

    std::shared_ptr<ConvLayerConfig> prev_config_conv_cast =
        std::dynamic_pointer_cast<ConvLayerConfig>(prev_config);
    std::shared_ptr<InputConfig> prev_config_input_cast =
        std::dynamic_pointer_cast<InputConfig>(prev_config);

    if (prev_config_input_cast) {  // prev is input layer
      return createLayerHelper(prev_config_input_cast->getDim(), 3, 3,
                               getKernelSize(next_config));
    } else if (!prev_config_conv_cast) {  // prev layer is not ConvLayer
      throw std::invalid_argument(
          "ConvLayer cannot come after another non-Conv layer.");
    } else {  // prev layer is ConvLayer
      return createLayerHelper(
          prev_config_conv_cast->getDim(), prev_config_conv_cast->num_filters,
          prev_config_conv_cast->num_filters * prev_config_conv_cast->sparsity,
          getKernelSize(next_config));
    }
  }

  uint64_t getDim() const { return num_filters * num_patches; }

  float getSparsity() const { return sparsity; }

  ActivationFunction getActFunc() const { return act_func; }

 private:
  uint64_t num_filters;
  float sparsity;
  ActivationFunction act_func;
  SamplingConfig sampling_config;
  std::pair<uint32_t, uint32_t> kernel_size;
  uint32_t num_patches;

  std::shared_ptr<SequentialLayer> createLayerHelper(
      uint64_t prev_dim, uint32_t prev_num_filters,
      uint32_t prev_num_sparse_filters,
      std::pair<uint32_t, uint32_t> next_kernel_size) {
    return std::static_pointer_cast<SequentialLayer>(
        std::make_shared<ConvLayer>(num_filters, sparsity, act_func,
                                    sampling_config, kernel_size, num_patches,
                                    prev_dim, prev_num_filters,
                                    prev_num_sparse_filters, next_kernel_size));
  }

  std::pair<uint32_t, uint32_t> getKernelSize(
      const std::shared_ptr<SequentialLayerConfig>& next_config) {
    std::shared_ptr<ConvLayerConfig> next_config_casted =
        std::dynamic_pointer_cast<ConvLayerConfig>(next_config);
    std::pair<uint32_t, uint32_t> next_kernel_size;
    if (next_config_casted) {
      next_kernel_size =
          std::pair<uint32_t, uint32_t>(next_config_casted->kernel_size.first,
                                        next_config_casted->kernel_size.second);
    } else {
      next_kernel_size = std::pair<uint32_t, uint32_t>(1, 1);
    }
    return next_kernel_size;
  }
};

struct EmbeddingLayerConfig {
  uint32_t num_embedding_lookups;
  uint32_t lookup_size;
  uint32_t log_embedding_block_size;

  EmbeddingLayerConfig(uint32_t _num_embedding_lookups, uint32_t _lookup_size,
                       uint32_t _log_embedding_block_size)
      : num_embedding_lookups(_num_embedding_lookups),
        lookup_size(_lookup_size),
        log_embedding_block_size(_log_embedding_block_size) {}
};

}  // namespace thirdai::bolt