#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

struct BoltVector {
 public:
  uint32_t* active_neurons;
  float* activations;
  float* gradients;
  uint32_t len;

  BoltVector()
      : active_neurons(nullptr),
        activations(nullptr),
        gradients(nullptr),
        len(0) {}

  constexpr explicit BoltVector(uint32_t* an, float* a, float* g, uint32_t l)
      : active_neurons(an),
        activations(a),
        gradients(g),
        len(l),
        _owns_data(false) {
    std::cout << "Default constructor" << std::endl;
  }

  explicit BoltVector(uint32_t l, bool sparse, bool has_gradient = true)
      : len(l), _owns_data(true) {
    std::cout << "Allocating constructor" << std::endl;
    if (sparse) {
      active_neurons = new uint32_t[len];
    } else {
      active_neurons = nullptr;
    }
    activations = new float[len];
    if (has_gradient) {
      gradients = new float[len];
    } else {
      gradients = nullptr;
    }
  }

  static BoltVector makeSparseVector(const std::vector<uint32_t>& indices,
                                     const std::vector<float>& values) {
    assert(indices.size() == values.size());
    BoltVector vec(indices.size(), true, false);
    std::copy(indices.begin(), indices.end(), vec.active_neurons);
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  static BoltVector makeDenseVector(const std::vector<float>& values) {
    BoltVector vec(values.size(), false, false);
    std::cout << "makeDenseVector: " << vec.active_neurons << " "
              << vec.activations << " " << vec.gradients << std::endl;
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  BoltVector(const BoltVector&) = delete;

  BoltVector(BoltVector&& other) : len(other.len) {
    std::cout << "Move constructor" << std::endl;
    std::cout << "\t into ptr = " << this << std::endl;

    freeMemory();
    this->_owns_data = other._owns_data;
    this->active_neurons = other.active_neurons;
    this->activations = other.activations;
    this->gradients = other.gradients;

    other.active_neurons = nullptr;
    other.activations = nullptr;
    other.gradients = nullptr;
    other.len = 0;
  }

  BoltVector& operator=(const BoltVector&) = delete;

  BoltVector& operator=(BoltVector&& other) {
    std::cout << "Move assignment" << std::endl;

    this->len = other.len;
    freeMemory();

    this->_owns_data = other._owns_data;
    this->active_neurons = other.active_neurons;
    this->activations = other.activations;
    this->gradients = other.gradients;

    other.active_neurons = nullptr;
    other.activations = nullptr;
    other.gradients = nullptr;
    other.len = 0;

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const BoltVector& state) {
    bool dense = state.active_neurons == nullptr;
    for (uint32_t i = 0; i < state.len; i++) {
      printf("\t%u: \t%0.3f, \t%0.3f\n", dense ? i : state.active_neurons[i],
             state.activations[i],
             state.gradients != nullptr ? state.gradients[i] : 0.0);
    }
    return out;
  }

  constexpr bool isDense() const { return this->active_neurons == nullptr; }

  ~BoltVector() {
    std::cout << "Destructor called" << std::endl;
    freeMemory();
  }

 private:
  bool _owns_data;

  void freeMemory() {  // NOLINT clang tidy thinks this method should be const
    if (_owns_data) {
      delete[] this->active_neurons;
      delete[] this->activations;
      if (this->gradients == nullptr) {
        std::cout << "Gradient is null: " << this->gradients << std::endl;
      } else {
        std::cout << "Gradient is not null: " << this->gradients << std::endl;
      }
      delete[] this->gradients;
    }
  }
};

class BoltBatch {
 private:
  std::vector<BoltVector> _vector_states;
  uint32_t _batch_size;

 public:
  BoltBatch() : _batch_size(0) {}

  BoltBatch(const uint32_t dim, const uint32_t batch_size, bool is_dense)
      : _batch_size(batch_size) {
    for (uint32_t i = 0; i < _batch_size; i++) {
      _vector_states.push_back(BoltVector(dim, !is_dense));
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

  BoltBatch(BoltBatch&& other)
      : _vector_states(std::move(other._vector_states)),
        _batch_size(other._batch_size) {
    other._batch_size = 0;
  }

  BoltBatch& operator=(const BoltBatch& other) = delete;

  BoltBatch& operator=(BoltBatch&& other) {
    _vector_states = std::move(other._vector_states);
    _batch_size = other._batch_size;
    other._batch_size = 0;
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const BoltBatch& state) {
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    for (uint32_t i = 0; i < state._batch_size; i++) {
      std::cout << "Vector: " << i << ":\n"
                << state._vector_states.at(i) << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    return out;
  }
};

}  // namespace thirdai::bolt