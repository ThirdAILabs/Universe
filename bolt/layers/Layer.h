#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

struct VectorState {
  uint32_t* active_neurons;
  float* activations;
  float* gradients;
  uint32_t len;

  VectorState() {}

  constexpr explicit VectorState(uint32_t* an, float* a, float* g, uint32_t l)
      : active_neurons(an), activations(a), gradients(g), len(l) {}

  constexpr static VectorState makeDenseState(float* a, float* g, uint32_t l) {
    return VectorState(nullptr, a, g, l);
  }

  constexpr static VectorState makeSparseInputState(uint32_t* an, float* a,
                                                    uint32_t l) {
    return VectorState(an, a, nullptr, l);
  }

  constexpr static VectorState makeDenseInputState(float* a, uint32_t l) {
    return VectorState(nullptr, a, nullptr, l);
  }

  template <typename BATCH_T>
  static VectorState makeInputStateFromBatch(const BATCH_T& input_batch,
                                             uint32_t i);

  friend std::ostream& operator<<(std::ostream& out, const VectorState& state) {
    bool dense = state.active_neurons == nullptr;
    for (uint32_t i = 0; i < state.len; i++) {
      printf("\t%u: \t%0.3f, \t%0.3f\n", dense ? i : state.active_neurons[i],
             state.activations[i], state.gradients[i]);
    }
    return out;
  }
};

class BatchState {
 private:
  VectorState* _vector_states;
  uint32_t* _active_neurons_buf;
  float* _activations_buf;
  float* _gradients_buf;
  uint32_t _batch_size;

 public:
  BatchState()
      : _vector_states(nullptr),
        _active_neurons_buf(nullptr),
        _activations_buf(nullptr),
        _gradients_buf(nullptr) {}

  BatchState(const uint32_t dim, const uint32_t batch_size, bool is_dense)
      : _batch_size(batch_size) {
    _vector_states = new VectorState[batch_size];

    if (!is_dense) {
      _active_neurons_buf = new uint32_t[batch_size * dim];
    } else {
      _active_neurons_buf = nullptr;
    }
    _activations_buf = new float[batch_size * dim];
    _gradients_buf = new float[batch_size * dim];

    for (uint32_t i = 0; i < batch_size; i++) {
      uint32_t* an_ptr = is_dense ? nullptr : _active_neurons_buf + (i * dim);
      _vector_states[i] = VectorState(an_ptr, _activations_buf + (i * dim),
                                      _gradients_buf + (i * dim), dim);
    }
  }

  VectorState& operator[](size_t i) {
    assert(i < _batch_size);
    return _vector_states[i];
  }

  const VectorState& operator[](size_t i) const {
    assert(i < _batch_size);
    return _vector_states[i];
  }

  BatchState(const BatchState& other) = delete;

  BatchState(BatchState&& other)
      : _vector_states(other._vector_states),
        _active_neurons_buf(other._active_neurons_buf),
        _activations_buf(other._activations_buf),
        _gradients_buf(other._gradients_buf),
        _batch_size(other._batch_size) {
    other._vector_states = nullptr;
    other._active_neurons_buf = nullptr;
    other._activations_buf = nullptr;
    other._gradients_buf = nullptr;
  }

  BatchState& operator=(const BatchState& other) = delete;

  BatchState& operator=(BatchState&& other) {
    this->_vector_states = other._vector_states;
    this->_active_neurons_buf = other._active_neurons_buf;
    this->_activations_buf = other._activations_buf;
    this->_batch_size = other._batch_size;

    other._vector_states = nullptr;
    other._active_neurons_buf = nullptr;
    other._activations_buf = nullptr;
    other._gradients_buf = nullptr;

    return *this;
  }

  ~BatchState() {
    delete[] _vector_states;
    delete[] _active_neurons_buf;
    delete[] _activations_buf;
    delete[] _gradients_buf;
  }

  friend std::ostream& operator<<(std::ostream& out, const BatchState& state) {
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    for (uint32_t i = 0; i < state._batch_size; i++) {
      std::cout << "Vector: " << i << ": " << state._vector_states[i]
                << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    return out;
  }
};

}  // namespace thirdai::bolt