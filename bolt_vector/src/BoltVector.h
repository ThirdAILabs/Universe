#pragma once

#include <cereal/access.hpp>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thirdai {

using ValueIndexPair = std::pair<float, uint32_t>;

// This compares the first element in the pair, then the second element.
using TopKActivationsQueue =
    std::priority_queue<ValueIndexPair, std::vector<ValueIndexPair>,
                        std::greater<ValueIndexPair>>;

struct FoundActiveNeuron {
  // If pos is nullopt, then it means that we searched through a sparse
  // BoltVector and the neuron was not found (the activation in this case will
  // be 0).
  std::optional<size_t> pos;
  float activation;
};

struct BoltVector {
 public:
  uint32_t* active_neurons;
  float* activations;
  float* gradients;
  uint32_t len;

  BoltVector();

  BoltVector(uint32_t* an, float* a, float* g, uint32_t l);

  BoltVector(uint32_t l, bool is_dense, bool has_gradient = true);

  static BoltVector singleElementSparseVector(uint32_t active_neuron,
                                              float activation = 1.0);

  uint32_t getHighestActivationId() const;

  uint32_t getSecondHighestActivationId() const;

  void sortActiveNeurons();

  static BoltVector makeSparseVector(const std::vector<uint32_t>& indices,
                                     const std::vector<float>& values);

  static BoltVector makeSparseVector(
      const std::vector<std::pair<uint32_t, float>>& index_value_pairs);

  static BoltVector makeDenseVector(const std::vector<float>& values);

  static BoltVector makeSparseVectorWithGradients(
      const std::vector<uint32_t>& indices, const std::vector<float>& values);

  static BoltVector makeDenseVectorWithGradients(
      const std::vector<float>& values);

  std::vector<ValueIndexPair> indexValuePairs() const;

  BoltVector copy() const;

  BoltVector(const BoltVector& other);

  BoltVector(BoltVector&& other) noexcept;

  BoltVector& operator=(const BoltVector& other);

  BoltVector& operator=(BoltVector&& other) noexcept;

  template <bool DENSE>
  uint32_t activeNeuronAtIndex(uint32_t index) const;

  void zeroOutGradients();

  BoltVector viewChunk(size_t chunk_idx, size_t chunk_size) const;

  /**
   * Finds the position and activation (value) of an active neuron.
   * Whether or not the vector is dense is templated because this is
   * likely used in loops where the density is known beforehand. This
   * allows us to reuse that information and prevent checking at every
   * iteration.
   */
  template <bool DENSE>
  FoundActiveNeuron findActiveNeuron(uint32_t active_neuron) const;

  FoundActiveNeuron findActiveNeuronNoTemplate(uint32_t active_neuron) const;

  bool isDense() const;

  // Returns the active neuron ID's that are greater than activation_threshold.
  // Returns at most max_count_to_return (if number of neurons exceeds
  // max_count_to_return, returns those with highest activations). If
  // return_at_least_one is true, returns the neuron with the highest activation
  // even if no neurons otherwise exceeded activation_threshold.
  std::vector<uint32_t> getThresholdedNeurons(
      float activation_threshold, bool return_at_least_one,
      uint32_t max_count_to_return) const;

  TopKActivationsQueue findKLargestActivations(uint32_t k) const;

  bool hasGradients() const;

  bool ownsMemory() const { return _owns_data; }

  friend std::ostream& operator<<(std::ostream& out, const BoltVector& vec);

  std::string toString() const;

  ~BoltVector() noexcept;

 private:
  /**
   * Finds the position and activation (value) of an active neuron in
   * a sparse vector. Assumes that the vector is sparse; the active_neurons
   * array must not be nullptr.
   */
  FoundActiveNeuron findSparseActiveNeuron(uint32_t active_neuron) const;

  bool _owns_data;

  void freeMemory();

  friend class cereal::access;
  template <class Archive>
  void save(Archive& archive) const;

  template <class Archive>
  void load(Archive& archive);
};

class BoltBatch {
 private:
  std::vector<BoltVector> _vectors;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_vectors);
  }

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

  auto begin() const { return _vectors.begin(); }

  auto end() const { return _vectors.end(); }

  auto begin() { return _vectors.begin(); }

  auto end() { return _vectors.end(); }

  auto& vectors() { return _vectors; }

  uint32_t getBatchSize() const { return _vectors.size(); }

  /*
   * Throws an exception if the vector is not of the passed in
   * expected_dimension (for a sparse vector this just means none of the
   * active neurons are too large). "origin_string" should be a descriptive
   * string that tells the user where the error comes from if it is thrown, e.g.
   * something like "Passed in BoltVector too large for Input".
   */
  void verifyExpectedDimension(
      uint32_t expected_dimension,
      std::optional<std::pair<uint32_t, uint32_t>> num_nonzeros_range,
      const std::string& origin_string) const {
    for (const BoltVector& vec : _vectors) {
      if (vec.isDense()) {
        if (vec.len != expected_dimension) {
          throw std::invalid_argument(
              origin_string + ": Received dense BoltVector with dimension=" +
              std::to_string(vec.len) +
              ", but was supposed to have dimension=" +
              std::to_string(expected_dimension));
        }
      } else {
        for (uint32_t i = 0; i < vec.len; i++) {
          uint32_t active_neuron = vec.active_neurons[i];
          if (active_neuron >= expected_dimension) {
            throw std::invalid_argument(
                origin_string +
                ": Received sparse BoltVector with active_neuron=" +
                std::to_string(active_neuron) + " but was supposed to have=" +
                std::to_string(expected_dimension));
          }
        }
      }

      if (num_nonzeros_range && (vec.len > num_nonzeros_range.value().second ||
                                 vec.len < num_nonzeros_range.value().first)) {
        std::stringstream ss;
        ss << origin_string << ": Received BoltVector with len "
           << std::to_string(vec.len) + " but was expected to have between "
           << num_nonzeros_range.value().first << " and "
           << num_nonzeros_range.value().second << " nonzeros.";

        throw std::invalid_argument(ss.str());
      }
    }
  }
};

}  // namespace thirdai
