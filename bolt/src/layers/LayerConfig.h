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
  virtual std::shared_ptr<SequentialLayer> createInputLayer(
      uint64_t input_dim, const SequentialLayerConfig& next_config) = 0;

  virtual std::shared_ptr<SequentialLayer> createHiddenLayer(
      const SequentialLayerConfig& prev_config,
      const SequentialLayerConfig& next_config) = 0;

  virtual std::shared_ptr<SequentialLayer> createOutputLayer(
      const SequentialLayerConfig& prev_config) = 0;

  virtual std::shared_ptr<SequentialLayer> createSingleHiddenLayer(
      uint64_t input_dim) = 0;

  virtual uint64_t getDim() const = 0;

  virtual float getSparsity() const = 0;

  virtual ActivationFunction getActFunc() const = 0;

  virtual void print(std::ostream& out) const = 0;

  friend std::ostream& operator<<(std::ostream& out,
                                  const SequentialLayerConfig& config) {
    config.print(out);
    return out;
  }

  static void checkSparsity(float sparsity) {
    if (0.2 < sparsity && sparsity < 1.0) {
      std::cout << "WARNING: Using large load_factor value " << sparsity
                << " in Layer, consider decreasing load_factor" << std::endl;
    }
  }
};

class FullyConnectedLayerConfig final : public SequentialLayerConfig {
 public:
  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            ActivationFunction act_func, SamplingConfig config)
      : _dim(dim),
        _sparsity(sparsity),
        _act_func(act_func),
        _sampling_config(config) {
    checkSparsity(sparsity);
  }

  FullyConnectedLayerConfig(uint64_t dim, ActivationFunction act_func)
      : _dim(dim),
        _sparsity(1.0),
        _act_func(act_func),
        _sampling_config(SamplingConfig()) {
    checkSparsity(_sparsity);
  }

  FullyConnectedLayerConfig(uint64_t dim, float sparsity,
                            ActivationFunction act_func)
      : _dim(dim), _sparsity(sparsity), _act_func(act_func) {
    checkSparsity(sparsity);
    if (sparsity < 1.0) {
      uint32_t rp = (static_cast<uint32_t>(log2(dim)) / 3) * 3;
      uint32_t k = rp / 3;
      uint32_t rs = (dim * 4) / (1 << rp);
      uint32_t l = sparsity < 0.1 ? 256 : 64;
      _sampling_config = SamplingConfig(k, l, rp, rs);
    } else {
      _sampling_config = SamplingConfig();
    }
  }

  std::shared_ptr<SequentialLayer> createInputLayer(
      uint64_t input_dim, const SequentialLayerConfig& next_config) {
    (void)next_config;
    return createAsSequentialLayer(input_dim);
  }

  std::shared_ptr<SequentialLayer> createHiddenLayer(
      const SequentialLayerConfig& prev_config,
      const SequentialLayerConfig& next_config) {
    (void)next_config;
    return createAsSequentialLayer(prev_config.getDim());
  }

  std::shared_ptr<SequentialLayer> createOutputLayer(
      const SequentialLayerConfig& prev_config) {
    return createAsSequentialLayer(prev_config.getDim());
  }

  std::shared_ptr<SequentialLayer> createSingleHiddenLayer(uint64_t input_dim) {
    return createAsSequentialLayer(input_dim);
  }

  std::shared_ptr<SequentialLayer> createLayer(
      const SequentialLayerConfig* const prev_config,
      const SequentialLayerConfig* const next_config) {
    if (next_config != nullptr && _act_func == ActivationFunction::Softmax) {
      throw std::invalid_argument(
          "Softmax activation function is not supported for hidden layers.");
    }
    return createAsSequentialLayer(prev_config->getDim());
  }

  uint64_t getDim() const { return _dim; }

  float getSparsity() const { return _sparsity; }

  ActivationFunction getActFunc() const { return _act_func; }

  void print(std::ostream& out) const {
    out << "Layer: dim=" << _dim << ", load_factor=" << _sparsity;
    switch (_act_func) {
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
    if (_sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << _sampling_config.hashes_per_table
          << ", num_tables=" << _sampling_config.num_tables
          << ", range_pow=" << _sampling_config.range_pow
          << ", reservoir_size=" << _sampling_config.reservoir_size << "}";
    }
  }

 private:
  uint64_t _dim;
  float _sparsity;
  ActivationFunction _act_func;
  SamplingConfig _sampling_config;

  std::shared_ptr<SequentialLayer> createAsSequentialLayer(uint64_t prev_dim) {
    return std::static_pointer_cast<SequentialLayer>(
        std::make_shared<FullyConnectedLayer>(_dim, _sparsity, _act_func,
                                              _sampling_config, prev_dim));
  }
};

class ConvLayerConfig final : public SequentialLayerConfig {
 public:
  ConvLayerConfig(uint64_t num_filters, float sparsity,
                  ActivationFunction act_func, SamplingConfig config,
                  std::pair<uint32_t, uint32_t> kernel_size,
                  uint32_t num_patches)
      : _num_filters(num_filters),
        _sparsity(sparsity),
        _act_func(act_func),
        _sampling_config(config),
        _kernel_size(std::move(kernel_size)),
        _num_patches(num_patches) {
    checkSparsity(sparsity);
  }

  std::shared_ptr<SequentialLayer> createInputLayer(
      uint64_t input_dim, const SequentialLayerConfig& next_config) {
    return createAsSequentialLayer(input_dim, 1, 1,
                                   getNextKernelSize(next_config));
  }

  std::shared_ptr<SequentialLayer> createHiddenLayer(
      const SequentialLayerConfig& prev_config,
      const SequentialLayerConfig& next_config) {
    try {
      const ConvLayerConfig& prev_conv_config =
          dynamic_cast<const ConvLayerConfig&>(prev_config);
      return createAsSequentialLayer(prev_conv_config.getDim(),
                                     prev_conv_config.getNumFilters(),
                                     prev_conv_config.getNumSparseFilters(),
                                     getNextKernelSize(next_config));
    } catch (std::exception& e) {
      (void)e;
      throw std::invalid_argument(
          "ConvLayer only comes after the input or another ConvLayer.");
    }
  }

  std::shared_ptr<SequentialLayer> createOutputLayer(
      const SequentialLayerConfig& prev_config) {
    (void)prev_config;
    throw std::invalid_argument("ConvLayer not supported as final layer.");
  }

  std::shared_ptr<SequentialLayer> createSingleHiddenLayer(uint64_t input_dim) {
    (void)input_dim;
    throw std::invalid_argument("ConvLayer not supported as final layer.");
  }

  uint64_t getDim() const { return _num_filters * _num_patches; }

  float getSparsity() const { return _sparsity; }

  ActivationFunction getActFunc() const { return _act_func; }

  uint64_t getNumFilters() const { return _num_filters; }

  uint64_t getNumSparseFilters() const { return _num_filters * _sparsity; }

  std::pair<uint32_t, uint32_t> getKernelSize() const { return _kernel_size; }

  void print(std::ostream& out) const {
    out << "Layer: num_filters=" << _num_filters
        << ", load_factor=" << _sparsity << ", num_patches=" << _num_patches;
    switch (_act_func) {
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
    if (_sparsity < 1.0) {
      out << ", sampling: {";
      out << "hashes_per_table=" << _sampling_config.hashes_per_table
          << ", num_tables=" << _sampling_config.num_tables
          << ", range_pow=" << _sampling_config.range_pow
          << ", reservoir_size=" << _sampling_config.reservoir_size << "}";
    }
  }

 private:
  uint64_t _num_filters;
  float _sparsity;
  ActivationFunction _act_func;
  SamplingConfig _sampling_config;
  std::pair<uint32_t, uint32_t> _kernel_size;
  uint32_t _num_patches;

  std::shared_ptr<SequentialLayer> createAsSequentialLayer(
      uint64_t prev_dim, uint32_t prev_num_filters,
      uint32_t prev_num_sparse_filters,
      std::pair<uint32_t, uint32_t> next_kernel_size) {
    if (_act_func != ActivationFunction::ReLU)
      throw std::invalid_argument(
          "Conv layers currently support only ReLU Activation.");
    if (_kernel_size.first != _kernel_size.second) {
      throw std::invalid_argument(
          "Conv layers currently support only square kernels.");
    }

    return std::static_pointer_cast<SequentialLayer>(
        std::make_shared<ConvLayer>(_num_filters, _sparsity, _act_func,
                                    _sampling_config, _kernel_size,
                                    _num_patches, prev_dim, prev_num_filters,
                                    prev_num_sparse_filters, next_kernel_size));
  }

  std::pair<uint32_t, uint32_t> getNextKernelSize(
      const SequentialLayerConfig& next_config) {
    try {
      const ConvLayerConfig& next_conv_config =
          dynamic_cast<const ConvLayerConfig&>(next_config);
      return next_conv_config.getKernelSize();
    } catch (std::exception& e) {
      return std::make_pair(1, 1);
    }
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