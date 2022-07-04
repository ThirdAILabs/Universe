#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::bolt {

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPS = 0.0000001;

struct FoundActiveNeuron {
  std::optional<size_t> pos;
  float activation;
};

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
    BoltVector vec(indices.size(), /* is_dense = */ false,
                   /* has_gradient = */ false);
    std::copy(indices.begin(), indices.end(), vec.active_neurons);
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  static BoltVector makeDenseVector(const std::vector<float>& values) {
    BoltVector vec(values.size(), /* is_dense = */ true,
                   /* has_gradient = */ false);
    std::copy(values.begin(), values.end(), vec.activations);
    return vec;
  }

  static BoltVector makeSparseVectorWithGradients(
      const std::vector<uint32_t>& indices, const std::vector<float>& values) {
    auto vector = makeSparseVector(indices, values);
    vector.gradients = new float[values.size()];
    std::fill(vector.gradients, vector.gradients + vector.len, 0);
    return vector;
  }

  static BoltVector makeDenseVectorWithGradients(
      const std::vector<float>& values) {
    auto vector = makeDenseVector(values);
    vector.gradients = new float[vector.len];
    std::fill(vector.gradients, vector.gradients + vector.len, 0);
    return vector;
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

  /**
   * Finds the position and activation (value) of an active neuron.
   * Whether or not the vector is dense is templated because this is
   * likely used in loops where the density is known beforehand. This
   * allows us to reuse that information and prevent checking at every
   * iteration.
   */
  template <bool DENSE>
  FoundActiveNeuron findActiveNeuron(uint32_t active_neuron) const {
    if (DENSE) {
      return {active_neuron, activations[active_neuron]};
    }
    return findSparseActiveNeuron(active_neuron);
  }

  FoundActiveNeuron findActiveNeuronNoTemplate(uint32_t active_neuron) const {
    if (isDense()) {
      return {active_neuron, activations[active_neuron]};
    }
    return findSparseActiveNeuron(active_neuron);
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
  /**
   * Finds the position and activation (value) of an active neuron in
   * a sparse vector. Assumes that the vector is sparse; the active_neurons
   * array must not be nullptr.
   */
  FoundActiveNeuron findSparseActiveNeuron(uint32_t active_neuron) const {
    assert(active_neurons != nullptr);

    const uint32_t* start = active_neurons;
    const uint32_t* end = active_neurons + len;
    const uint32_t* itr = std::find(start, end, active_neuron);
    if (itr == end) {
      return {{}, 0.0};
    }
    uint32_t pos = std::distance(start, itr);
    return {pos, activations[pos]};
  }

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
  std::vector<BoltVector> _vectors;

 public:
  BoltBatch() {}

  BoltBatch(const uint32_t dim, const uint32_t batch_size, bool is_dense) {
    for (uint32_t i = 0; i < batch_size; i++) {
      _vectors.push_back(BoltVector(dim, is_dense));
    }
  }

  explicit BoltBatch(std::vector<BoltVector>&& vectors)
      : _vectors(std::move(vectors)) {}

  BoltVector& operator[](size_t i) {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  const BoltVector& operator[](size_t i) const {
    assert(i < _vectors.size());
    return _vectors[i];
  }

  uint32_t getBatchSize() const { return _vectors.size(); }
  
  BoltBatch(const BoltBatch& other) = delete;

  BoltBatch(BoltBatch&& other) = default;

  BoltBatch& operator=(const BoltBatch& other) = delete;

  BoltBatch& operator=(BoltBatch&& other) = default;

  friend std::ostream& operator<<(std::ostream& out, const BoltBatch& state) {
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    for (uint32_t i = 0; i < state._vectors.size(); i++) {
      std::cout << "Vector: " << i << ":\n"
                << state._vectors.at(i) << std::endl;
    }
    std::cout << "-------------------------------------------------------------"
              << std::endl;
    return out;
  }
};

}  // namespace thirdai::bolt