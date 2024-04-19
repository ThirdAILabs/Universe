
#include "BoltVector.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai {

BoltVector::BoltVector()
    : active_neurons(nullptr),
      activations(nullptr),
      gradients(nullptr),
      len(0),
      _owns_data(true) {}

BoltVector::BoltVector(uint32_t* an, float* a, float* g, uint32_t l)
    : active_neurons(an),
      activations(a),
      gradients(g),
      len(l),
      _owns_data(false) {}

BoltVector::BoltVector(uint32_t l, bool is_dense, bool has_gradient /* = true*/)
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

BoltVector BoltVector::singleElementSparseVector(uint32_t active_neuron,
                                                 float activation /*= 1.0*/) {
  BoltVector vec(/* l= */ 1, /* is_dense= */ false,
                 /* has_gradient= */ false);
  vec.active_neurons[0] = active_neuron;
  vec.activations[0] = activation;

  return vec;
}

uint32_t BoltVector::getHighestActivationId() const {
  float max_act = activations[0];
  uint32_t id = 0;
  for (uint32_t i = 1; i < len; i++) {
    if (activations[i] > max_act) {
      max_act = activations[i];
      id = i;
    }
  }
  if (isDense()) {
    return id;
  }
  return active_neurons[id];
}

uint32_t BoltVector::getSecondHighestActivationId() const {
  float largest_activation = std::numeric_limits<float>::min(),
        second_largest_activation = std::numeric_limits<float>::min();
  uint32_t max_id = 0, second_max_id = 0;
  if (len < 2) {
    throw std::invalid_argument(
        "The sparse output dimension should be at least 2 to call "
        "getSecondHighestActivationId.");
  }
  for (uint32_t i = 0; i < len; i++) {
    if (activations[i] > largest_activation) {
      second_largest_activation = largest_activation;
      second_max_id = max_id;
      largest_activation = activations[i];
      max_id = i;
    } else if (activations[i] > second_largest_activation) {
      second_largest_activation = activations[i];
      second_max_id = i;
    }
  }
  if (isDense()) {
    return second_max_id;
  }
  return active_neurons[second_max_id];
}

void BoltVector::sortActiveNeurons() {  // NOLINT: clang-tidy thinks this should
                                        // be const.
  assert(!isDense());

  std::vector<std::pair<uint32_t, float>> contents;
  contents.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    contents.emplace_back(active_neurons[i], activations[i]);
  }

  std::sort(contents.begin(), contents.end());

  for (uint32_t i = 0; i < len; i++) {
    active_neurons[i] = contents[i].first;
    activations[i] = contents[i].second;
  }
}

BoltVector BoltVector::makeSparseVector(const std::vector<uint32_t>& indices,
                                        const std::vector<float>& values) {
  assert(indices.size() == values.size());
  BoltVector vec(indices.size(), /* is_dense = */ false,
                 /* has_gradient = */ false);
  std::copy(indices.begin(), indices.end(), vec.active_neurons);
  std::copy(values.begin(), values.end(), vec.activations);
  return vec;
}

BoltVector BoltVector::makeSparseVector(
    const std::vector<std::pair<uint32_t, float>>& index_value_pairs) {
  BoltVector vec(index_value_pairs.size(), /* is_dense = */ false,
                 /* has_gradient = */ false);
  for (uint32_t i = 0; i < index_value_pairs.size(); i++) {
    vec.active_neurons[i] = index_value_pairs[i].first;
    vec.activations[i] = index_value_pairs[i].second;
  }
  return vec;
}

BoltVector BoltVector::makeDenseVector(const std::vector<float>& values) {
  BoltVector vec(values.size(), /* is_dense = */ true,
                 /* has_gradient = */ false);
  std::copy(values.begin(), values.end(), vec.activations);
  return vec;
}

BoltVector BoltVector::makeSparseVectorWithGradients(
    const std::vector<uint32_t>& indices, const std::vector<float>& values) {
  auto vector = makeSparseVector(indices, values);
  vector.gradients = new float[values.size()];
  std::fill(vector.gradients, vector.gradients + vector.len, 0);
  return vector;
}

BoltVector BoltVector::makeDenseVectorWithGradients(
    const std::vector<float>& values) {
  auto vector = makeDenseVector(values);
  vector.gradients = new float[vector.len];
  std::fill(vector.gradients, vector.gradients + vector.len, 0);
  return vector;
}

std::vector<ValueIndexPair> BoltVector::valueIndexPairs() const {
  std::vector<ValueIndexPair> pairs;
  if (isDense()) {
    for (uint32_t i = 0; i < len; i++) {
      pairs.emplace_back(activations[i], i);
    }
  } else {
    for (uint32_t i = 0; i < len; i++) {
      pairs.emplace_back(activations[i], active_neurons[i]);
    }
  }
  return pairs;
}

BoltVector BoltVector::copy() const {
  BoltVector vec;
  vec.len = this->len;

  // Since we are copying the data underlying the original vector to create,
  // this vector, this vector will always own its own data, even if the vector
  // it is copying from does not.
  vec._owns_data = true;

  vec.activations = new float[len];
  std::copy(this->activations, this->activations + len, vec.activations);

  if (this->active_neurons != nullptr) {
    vec.active_neurons = new uint32_t[len];
    std::copy(this->active_neurons, this->active_neurons + len,
              vec.active_neurons);
  }

  if (this->gradients != nullptr) {
    vec.gradients = new float[len];
    std::copy(this->gradients, this->gradients + len, vec.gradients);
  }

  return vec;
}

BoltVector::BoltVector(const BoltVector& other)
    : len(other.len), _owns_data(true) {
  if (other.active_neurons != nullptr) {
    active_neurons = new uint32_t[len];
    std::copy(other.active_neurons, other.active_neurons + len, active_neurons);
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

BoltVector::BoltVector(BoltVector&& other) noexcept
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

BoltVector& BoltVector::operator=(const BoltVector& other) {
  if (&other == this) {
    return *this;
  }
  freeMemory();

  this->len = other.len;
  this->_owns_data = true;

  if (other.active_neurons != nullptr) {
    active_neurons = new uint32_t[len];
    std::copy(other.active_neurons, other.active_neurons + len, active_neurons);
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

BoltVector& BoltVector::operator=(BoltVector&& other) noexcept {
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

template <bool DENSE>
uint32_t BoltVector::activeNeuronAtIndex(uint32_t index) const {
  if (DENSE) {
    return index;
  }
  return active_neurons[index];
}

void BoltVector::zeroOutGradients() {  // NOLINT clang-tidy thinks this should
                                       // be const.
  assert(hasGradients());
  std::fill_n(gradients, len, 0.0);
}

BoltVector BoltVector::viewChunk(size_t chunk_idx, size_t chunk_size) const {
  if ((chunk_idx + 1) * chunk_size > len) {
    throw std::invalid_argument(
        "Cannot access chunk " + std::to_string(chunk_idx) +
        " with chunk size " + std::to_string(chunk_size) +
        " in vector of length " + std::to_string(len) + ".");
  }
  uint32_t* chunk_active_neurons = nullptr;
  if (!isDense()) {
    chunk_active_neurons = active_neurons + chunk_idx * chunk_size;
  }
  float* chunk_activations = activations + chunk_idx * chunk_size;

  float* chunk_gradients = nullptr;
  if (hasGradients()) {
    chunk_gradients = gradients + chunk_idx * chunk_size;
  }

  return BoltVector(chunk_active_neurons, chunk_activations, chunk_gradients,
                    chunk_size);
}

/**
 * Finds the position and activation (value) of an active neuron.
 * Whether or not the vector is dense is templated because this is
 * likely used in loops where the density is known beforehand. This
 * allows us to reuse that information and prevent checking at every
 * iteration.
 */
template <bool DENSE>
FoundActiveNeuron BoltVector::findActiveNeuron(uint32_t active_neuron) const {
  if (DENSE) {
    return {active_neuron, activations[active_neuron]};
  }
  return findSparseActiveNeuron(active_neuron);
}

FoundActiveNeuron BoltVector::findActiveNeuronNoTemplate(
    uint32_t active_neuron) const {
  if (isDense()) {
    return {active_neuron, activations[active_neuron]};
  }
  return findSparseActiveNeuron(active_neuron);
}

bool BoltVector::isDense() const {
  return (this->active_neurons == nullptr) && (this->len > 0);
}

// Returns the active neuron ID's that are greater than activation_threshold.
// Returns at most max_count_to_return (if number of neurons exceeds
// max_count_to_return, returns those with highest activations). If
// return_at_least_one is true, returns the neuron with the highest activation
// even if no neurons otherwise exceeded activation_threshold.
std::vector<uint32_t> BoltVector::getThresholdedNeurons(
    float activation_threshold, bool return_at_least_one,
    uint32_t max_count_to_return) const {
  std::vector<uint32_t> thresholded;
  std::vector<uint32_t> ids(len);
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [this](uint32_t i1, uint32_t i2) {
    return activations[i1] > activations[i2];
  });

  for (unsigned int& id : ids) {
    if (activations[id] < activation_threshold) {
      break;
    }
    if (thresholded.size() == max_count_to_return) {
      return thresholded;
    }

    uint32_t neuron = this->isDense() ? id : active_neurons[id];
    thresholded.push_back(neuron);
  }

  if (return_at_least_one && thresholded.empty()) {
    uint32_t max_act_neuron = this->isDense() ? ids[0] : active_neurons[ids[0]];
    thresholded.push_back(max_act_neuron);
  }

  return thresholded;
}

TopKActivationsQueue BoltVector::topKNeurons(uint32_t k) const {
  TopKActivationsQueue top_k;
  if (k == 0){
    return top_k;
  }
  for (uint32_t pos = 0; pos < std::min(k, len); pos++) {
    uint32_t idx = isDense() ? pos : active_neurons[pos];
    top_k.push({activations[pos], idx});
  }
  for (uint32_t pos = k; pos < len; pos++) {
    uint32_t idx = isDense() ? pos : active_neurons[pos];
    ValueIndexPair val_idx_pair = {activations[pos], idx};
    // top_k.top() is minimum element.
    if (val_idx_pair > top_k.top()) {
      top_k.pop();
      top_k.push(val_idx_pair);
    }
  }
  return top_k;
}

std::vector<ValueIndexPair> BoltVector::topKNeuronsAsVector(uint32_t k) const {
  auto pq = topKNeurons(k);
  std::vector<ValueIndexPair> vec;
  while (!pq.empty()) {
    vec.push_back(pq.top());
    pq.pop();
  }
  return vec;
}

bool BoltVector::hasGradients() const { return gradients != nullptr; }

std::ostream& operator<<(std::ostream& out, const BoltVector& vec) {
  out << "[";
  if (vec.isDense()) {
    for (size_t i = 0; i < vec.len; i++) {
      out << vec.activations[i];
      if (i < vec.len - 1) {
        out << ", ";
      }
    }
  } else {
    for (size_t i = 0; i < vec.len; i++) {
      out << "(" << vec.active_neurons[i] << ", " << vec.activations[i] << ")";
      if (i < vec.len - 1) {
        out << ", ";
      }
    }
  }
  out << "]";

  return out;
}

std::string BoltVector::toString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

BoltVector::~BoltVector() noexcept { freeMemory(); }

FoundActiveNeuron BoltVector::findSparseActiveNeuron(
    uint32_t active_neuron) const {
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

void BoltVector::freeMemory() {  // NOLINT clang tidy thinks this method should
                                 // be const
  if (_owns_data) {
    delete[] this->active_neurons;
    delete[] this->activations;
    delete[] this->gradients;
  }
}

template <class Archive>
void BoltVector::save(Archive& archive) const {
  archive(len);
  bool is_sparse = !isDense();
  bool has_gradients = hasGradients();
  archive(is_sparse, has_gradients);

  if (is_sparse) {
    archive(cereal::binary_data(active_neurons, len * sizeof(uint32_t)));
  }

  archive(cereal::binary_data(activations, len * sizeof(float)));

  if (has_gradients) {
    archive(cereal::binary_data(gradients, len * sizeof(float)));
  }
}

template <class Archive>
void BoltVector::load(Archive& archive) {
  archive(len);

  bool is_sparse, has_gradients;
  archive(is_sparse, has_gradients);

  if (is_sparse) {
    active_neurons = new uint32_t[len];
    archive(cereal::binary_data(active_neurons, len * sizeof(uint32_t)));
  }

  activations = new float[len];
  archive(cereal::binary_data(activations, len * sizeof(float)));

  if (has_gradients) {
    gradients = new float[len];
    archive(cereal::binary_data(gradients, len * sizeof(float)));
  }
}

template uint32_t BoltVector::activeNeuronAtIndex<true>(uint32_t index) const;
template uint32_t BoltVector::activeNeuronAtIndex<false>(uint32_t index) const;

template FoundActiveNeuron thirdai::BoltVector::findActiveNeuron<true>(
    uint32_t index) const;
template FoundActiveNeuron thirdai::BoltVector::findActiveNeuron<false>(
    uint32_t index) const;

template void BoltVector::load(cereal::PortableBinaryInputArchive& archive);
template void BoltVector::load(cereal::BinaryInputArchive& archive);

template void BoltVector::save(
    cereal::PortableBinaryOutputArchive& archive) const;
template void BoltVector::save(cereal::BinaryOutputArchive& archive) const;

}  // namespace thirdai
