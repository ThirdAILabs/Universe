#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

struct BoltVector {
  uint32_t* active_neurons;
  float* activations;
  float* gradients;
  uint32_t len;

  BoltVector() {}

  constexpr explicit BoltVector(uint32_t* an, float* a, float* g, uint32_t l)
      : active_neurons(an), activations(a), gradients(g), len(l) {}

  constexpr static BoltVector makeDenseState(float* a, float* g, uint32_t l) {
    return BoltVector(nullptr, a, g, l);
  }

  constexpr static BoltVector makeSparseInputState(uint32_t* an, float* a,
                                                   uint32_t l) {
    return BoltVector(an, a, nullptr, l);
  }

  constexpr static BoltVector makeDenseInputState(float* a, uint32_t l) {
    return BoltVector(nullptr, a, nullptr, l);
  }

  template <typename BATCH_T>
  static BoltVector makeInputStateFromBatch(const BATCH_T& input_batch,
                                            uint32_t i);

  friend std::ostream& operator<<(std::ostream& out, const BoltVector& state) {
    bool dense = state.active_neurons == nullptr;
    for (uint32_t i = 0; i < state.len; i++) {
      printf("\t%u: \t%0.3f, \t%0.3f\n", dense ? i : state.active_neurons[i],
             state.activations[i],
             state.gradients != nullptr ? state.gradients[i] : 0.0);
    }
    return out;
  }
};

class BoltBatch {
 private:
  BoltVector* _vector_states;
  uint32_t* _active_neurons_buf;
  float* _activations_buf;
  float* _gradients_buf;
  uint32_t _batch_size;

 public:
  BoltBatch()
      : _vector_states(nullptr),
        _active_neurons_buf(nullptr),
        _activations_buf(nullptr),
        _gradients_buf(nullptr) {}

  BoltBatch(const uint32_t dim, const uint32_t batch_size, bool is_dense)
      : _batch_size(batch_size) {
    _vector_states = new BoltVector[batch_size];

    if (!is_dense) {
      _active_neurons_buf = new uint32_t[batch_size * dim];
    } else {
      _active_neurons_buf = nullptr;
    }
    _activations_buf = new float[batch_size * dim];
    _gradients_buf = new float[batch_size * dim];

    for (uint32_t i = 0; i < batch_size; i++) {
      uint32_t* an_ptr = is_dense ? nullptr : _active_neurons_buf + (i * dim);
      _vector_states[i] = BoltVector(an_ptr, _activations_buf + (i * dim),
                                     _gradients_buf + (i * dim), dim);
    }
  }

  BoltVector& operator[](size_t i) {
    assert(i < _batch_size);
    return _vector_states[i];
  }

  const BoltVector& operator[](size_t i) const {
    assert(i < _batch_size);
    return _vector_states[i];
  }

  BoltBatch(const BoltBatch& other) = delete;

  BoltBatch(BoltBatch&& other) : _batch_size(other._batch_size) {
    delete[] this->_vector_states;
    this->_vector_states = other._vector_states;
    delete[] this->_active_neurons_buf;
    this->_active_neurons_buf = other._active_neurons_buf;
    delete[] this->_activations_buf;
    this->_activations_buf = other._activations_buf;
    delete[] this->_gradients_buf;
    this->_gradients_buf = other._gradients_buf;

    other._vector_states = nullptr;
    other._active_neurons_buf = nullptr;
    other._activations_buf = nullptr;
    other._gradients_buf = nullptr;
  }

  BoltBatch& operator=(const BoltBatch& other) = delete;

  BoltBatch& operator=(BoltBatch&& other) {
    delete[] this->_vector_states;
    this->_vector_states = other._vector_states;
    delete[] this->_active_neurons_buf;
    this->_active_neurons_buf = other._active_neurons_buf;
    delete[] this->_activations_buf;
    this->_activations_buf = other._activations_buf;
    delete[] this->_gradients_buf;
    this->_gradients_buf = other._gradients_buf;

    this->_batch_size = other._batch_size;

    other._vector_states = nullptr;
    other._active_neurons_buf = nullptr;
    other._activations_buf = nullptr;
    other._gradients_buf = nullptr;

    return *this;
  }

  ~BoltBatch() {
    delete[] _vector_states;
    delete[] _active_neurons_buf;
    delete[] _activations_buf;
    delete[] _gradients_buf;
  }

  friend std::ostream& operator<<(std::ostream& out, const BoltBatch& state) {
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    for (uint32_t i = 0; i < state._batch_size; i++) {
      std::cout << "Vector: " << i << ":\n"
                << state._vector_states[i] << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    return out;
  }
};

}  // namespace thirdai::bolt