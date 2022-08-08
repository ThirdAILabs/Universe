#pragma once

#include "compression/src/CompressedVector.h"
#include "utils/utils.h"

namespace thirdai::optim {

class Optimizer {
 public:
  virtual float dWeight(float lr, uint64_t indx, float grad, float B1, float B2,
                        float B1_bias_corrected, float B2_bias_corrected,
                        float eps) = 0;
  virtual float dBias(float lr, uint64_t cur_neuron, float grad, float B1,
                      float B2, float B1_bias_corrected,
                      float B2_bias_corrected, float eps) = 0;

  virtual void reset() = 0;
  virtual ~Optimizer() = default;
};

class Adam : public Optimizer {
 public:
  Adam(uint64_t dim, uint64_t prev_dim)
      : _w_momentum(dim * prev_dim, 0),
        _w_velocity(dim * prev_dim, 0),
        _b_momentum(dim, 0),
        _b_velocity(dim, 0),
        _dim(dim),
        _prev_dim(prev_dim) {
    reset();
  }

  float dWeight(float lr, uint64_t indx, float grad, float B1, float B2,
                float B1_bias_corrected, float B2_bias_corrected,
                float eps) final {
    _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
    _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;
    float dW = lr * (_w_momentum[indx] / B1_bias_corrected) /
               (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);
    return dW;
  }

  float dBias(float lr, uint64_t cur_neuron, float grad, float B1, float B2,
              float B1_bias_corrected, float B2_bias_corrected,
              float eps) final {
    _b_momentum[cur_neuron] = B1 * _b_momentum[cur_neuron] + (1 - B1) * grad;
    _b_velocity[cur_neuron] =
        B2 * _b_velocity[cur_neuron] + (1 - B2) * grad * grad;
    float dbias =
        lr * (_b_momentum[cur_neuron] / B1_bias_corrected) /
        (std::sqrt(_b_velocity[cur_neuron] / B2_bias_corrected) + eps);
    return dbias;
  }

  void reset() final {
    _w_momentum.assign(_dim * _prev_dim, 0);
    _w_velocity.assign(_dim * _prev_dim, 0);

    _b_momentum.assign(_dim, 0);
    _b_velocity.assign(_dim, 0);
  }

  void clear() {
    // _w_gradient.clear();
    // _b_gradient.clear();

    _w_momentum.clear();
    _w_velocity.clear();

    _b_momentum.clear();
    _b_velocity.clear();
  }

 private:
  std::vector<float> _w_momentum;
  std::vector<float> _w_velocity;
  std::vector<float> _b_momentum;
  std::vector<float> _b_velocity;

  uint64_t _dim;
  uint64_t _prev_dim;
};

class CompressedAdam : public Optimizer {
 public:
  CompressedAdam(uint64_t dim, uint64_t prev_dim)
      : _w_momentum(dim * prev_dim, 0),
        _w_velocity(dim * prev_dim, 0),
        _b_momentum(dim, 0),
        _b_velocity(dim, 0),
        _dim(dim),
        _prev_dim(prev_dim) {
    reset();
  };

  float dWeight(float lr, uint64_t indx, float grad, float B1, float B2,
                float B1_bias_corrected, float B2_bias_corrected,
                float eps) final {
    _w_momentum.set(indx, B1 * _w_momentum[indx] + (1 - B1) * grad);
    _w_velocity.set(indx, B2 * _w_velocity[indx] + (1 - B2) * grad * grad);

    BOLT_TRACE(B1);
    BOLT_TRACE(_w_momentum[indx]);
    BOLT_TRACE(1 - B1);

    BOLT_TRACE(B2);
    BOLT_TRACE(1 - B2);
    BOLT_TRACE(grad * grad);

    assert(!std::isnan(_w_momentum[indx]));
    assert(!std::isnan(_w_velocity[indx]));

    float dW = lr * (_w_momentum[indx] / B1_bias_corrected) /
               (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);
    return dW;
  }

  float dBias(float lr, uint64_t cur_neuron, float grad, float B1, float B2,
              float B1_bias_corrected, float B2_bias_corrected,
              float eps) final {
    if (std::isnan(grad)) {
      BOLT_TRACE(grad);
    }
    BOLT_TRACE(lr);
    BOLT_TRACE(B1_bias_corrected);
    BOLT_TRACE(B2_bias_corrected);
    _b_momentum.set(cur_neuron, B1 * _b_momentum[cur_neuron] + (1 - B1) * grad);
    _b_velocity.set(cur_neuron,
                    B2 * _b_velocity[cur_neuron] + (1 - B2) * grad * grad);

    assert(!std::isnan(_b_momentum[cur_neuron]));
    assert(!std::isnan(_b_velocity[cur_neuron]));

    float dbias =
        lr * (_b_momentum[cur_neuron] / B1_bias_corrected) /
        (std::sqrt(_b_velocity[cur_neuron] / B2_bias_corrected) + eps);
    return dbias;
  }

  void reset() final {
    _w_momentum.assign(_dim * _prev_dim, 0);
    _w_velocity.assign(_dim * _prev_dim, 0);

    _b_momentum.assign(_dim, 0);
    _b_velocity.assign(_dim, 0);
  }

  void clear() {
    // _w_gradient.clear();
    // _b_gradient.clear();
    _w_momentum.clear();
    _w_velocity.clear();

    _b_momentum.clear();
    _b_velocity.clear();
  }

 private:
  using AdamParamsType = ::thirdai::bolt::BiasedSketch<float>;
  AdamParamsType _w_momentum;
  AdamParamsType _w_velocity;
  AdamParamsType _b_momentum;
  AdamParamsType _b_velocity;

  uint64_t _dim;
  uint64_t _prev_dim;
};

inline std::unique_ptr<Optimizer> make_optimizer(uint64_t dim,
                                                 uint64_t& prev_dim) {
  return std::make_unique<Adam>(dim, prev_dim),
  // return std::make_unique<CompressedAdam>(dim, prev_dim);
}

}  // namespace thirdai::optim
