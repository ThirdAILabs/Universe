#pragma once

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
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
        _owns_data(false) {}

  explicit BoltVector(uint32_t l, bool is_dense, bool has_gradient = true)
      : len(l), _owns_data(true) {
    if (!is_dense) {
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
    BoltVector vec(indices.size(), false, false);
    std::copy(indices.begin(), indices.end(), vec.active_neurons);
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  static BoltVector makeDenseVector(const std::vector<float>& values) {
    BoltVector vec(values.size(), true, false);
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  // TODO(nicholas): delete copy constructor/assignment and make load dataset
  // return smart pointer
  BoltVector(const BoltVector& other) : len(other.len), _owns_data(true) {
    if (other.active_neurons != nullptr) {
      active_neurons = new uint32_t[len];
      std::copy(other.active_neurons, other.active_neurons + len,
                active_neurons);
    } else {
      active_neurons = nullptr;
    }

    activations = new float[len];
    std::copy(other.activations, other.activations + len, activations);

    if (other.gradients != nullptr) {
      gradients = new float[len];
      std::copy(other.gradients, other.gradients + len, gradients);
    } else {
      gradients = nullptr;
    }
  }

  BoltVector(BoltVector&& other)
      : active_neurons(other.active_neurons),
        activations(other.activations),
        gradients(other.gradients),
        len(other.len),
        _owns_data(other._owns_data) {
    other.active_neurons = nullptr;
    other.activations = nullptr;
    other.gradients = nullptr;
    other.len = 0;
  }

  // TODO(nicholas): delete copy constructor/assignment and make load dataset
  // return smart pointer
  BoltVector& operator=(const BoltVector& other) {
    if (&other == this) {
      return *this;
    }
    freeMemory();

    this->len = other.len;
    this->_owns_data = true;

    if (other.active_neurons != nullptr) {
      active_neurons = new uint32_t[len];
      std::copy(other.active_neurons, other.active_neurons + len,
                active_neurons);
    } else {
      active_neurons = nullptr;
    }

    activations = new float[len];
    std::copy(other.activations, other.activations + len, activations);

    if (other.gradients != nullptr) {
      gradients = new float[len];
      std::copy(other.gradients, other.gradients + len, gradients);
    } else {
      gradients = nullptr;
    }

    return *this;
  }

  BoltVector& operator=(BoltVector&& other) {
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

  std::string toString() const {
    std::stringstream ss;
    ss << "[";

    if (isDense()) {
      for (size_t i = 0; i < len; i++) {
        ss << activations[i];
        if (i < len - 1) {
          ss << ", ";
        }
      }
    } else {
      for (size_t i = 0; i < len; i++) {
        ss << "(" << active_neurons[i] << ", " << activations[i] << ")";
        if (i < len - 1) {
          ss << ", ";
        }
      }
    }

    ss << "]";

    return ss.str();
  }

  ~BoltVector() { freeMemory(); }

 private:
  bool _owns_data;

  void freeMemory() {  // NOLINT clang tidy thinks this method should be const
    if (_owns_data) {
      delete[] this->active_neurons;
      delete[] this->activations;
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
      _vector_states.push_back(BoltVector(dim, is_dense));
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