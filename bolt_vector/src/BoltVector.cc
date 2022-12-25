
#include "BoltVector.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>

namespace thirdai {

BoltVector::BoltVector()
    : neurons(nullptr), activations(nullptr), gradients(nullptr), len(0) {}

BoltVector::BoltVector(const uint32_t* neurons_src,
                       const float* activations_src, const float* gradients_src,
                       uint32_t length)
    : BoltVector(length, neurons_src == nullptr, gradients_src == nullptr) {
  if (neurons_src) {
    std::copy(neurons_src, neurons_src + length, neurons);
  }

  std::copy(activations_src, activations_src + length, activations);

  if (gradients_src) {
    std::copy(gradients_src, gradients_src + length, gradients);
  }
}

BoltVector::BoltVector(uint32_t l, bool is_dense, bool has_gradient /* = true*/)
    : len(l) {
  if (!is_dense) {
    neurons = new uint32_t[len];
  } else {
    neurons = nullptr;
  }
  activations = new float[len];
  if (has_gradient) {
    gradients = new float[len];
  } else {
    gradients = nullptr;
  }
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
  return neurons[id];
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
  return neurons[second_max_id];
}

void BoltVector::sortActiveNeurons() {  // NOLINT: clang-tidy thinks this should
                                        // be const.
  assert(!isDense());

  std::vector<std::pair<uint32_t, float>> contents;
  contents.reserve(len);
  for (uint32_t i = 0; i < len; i++) {
    contents.emplace_back(neurons[i], activations[i]);
  }

  std::sort(contents.begin(), contents.end());

  for (uint32_t i = 0; i < len; i++) {
    neurons[i] = contents[i].first;
    activations[i] = contents[i].second;
  }
}

BoltVector BoltVector::sparse(const std::vector<uint32_t>& indices,
                              const std::vector<float>& values,
                              bool has_gradient /*= true*/) {
  assert(indices.size() == values.size());
  BoltVector vector(indices.size(), /* is_dense = */ false, has_gradient);
  std::copy(indices.begin(), indices.end(), vector.neurons);
  std::copy(values.begin(), values.end(), vector.activations);
  if (has_gradient) {
    std::fill(vector.gradients, vector.gradients + vector.len, 0);
  }
  return vector;
}

BoltVector BoltVector::dense(const std::vector<float>& values,
                             bool has_gradient /*=true*/) {
  BoltVector vector(values.size(), /* is_dense = */ true, has_gradient);
  std::copy(values.begin(), values.end(), vector.activations);
  if (has_gradient) {
    std::fill(vector.gradients, vector.gradients + vector.len, 0);
  }
  return vector;
}

BoltVector BoltVector::copy() const {
  BoltVector vec;
  vec.len = this->len;

  vec.activations = new float[len];
  std::copy(this->activations, this->activations + len, vec.activations);

  if (this->neurons != nullptr) {
    vec.neurons = new uint32_t[len];
    std::copy(this->neurons, this->neurons + len, vec.neurons);
  }

  if (this->gradients != nullptr) {
    vec.gradients = new float[len];
    std::copy(this->gradients, this->gradients + len, vec.gradients);
  }

  return vec;
}

// TODO(Josh): Delete copy constructor and copy assignment (will help when
// we've moved to new Dataset and removed BoltBatches)
BoltVector::BoltVector(const BoltVector& other) : len(other.len) {
  if (other.neurons != nullptr) {
    neurons = new uint32_t[len];
    std::copy(other.neurons, other.neurons + len, neurons);
  } else {
    neurons = nullptr;
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
    : neurons(other.neurons),
      activations(other.activations),
      gradients(other.gradients),
      len(other.len) {
  other.neurons = nullptr;
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

  if (other.neurons != nullptr) {
    neurons = new uint32_t[len];
    std::copy(other.neurons, other.neurons + len, neurons);
  } else {
    neurons = nullptr;
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

  this->neurons = other.neurons;
  this->activations = other.activations;
  this->gradients = other.gradients;

  other.neurons = nullptr;
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
  return neurons[index];
}

void BoltVector::zeroOutGradients() {  // NOLINT clang-tidy thinks this should
                                       // be const.
  assert(hasGradients());
  std::fill_n(gradients, len, 0.0);
}

/**
 * Finds the position and activation (value) of an active neuron.
 * Whether or not the vector is dense is templated because this is
 * likely used in loops where the density is known beforehand. This
 * allows us to reuse that information and prevent checking at every
 * iteration.
 */

FoundActiveNeuron BoltVector::find(uint32_t active_neuron) const {
  if (isDense()) {
    return {active_neuron, activations[active_neuron]};
  }

  // Else not dense
  assert(neurons != nullptr);

  const uint32_t* start = neurons;
  const uint32_t* end = neurons + len;
  const uint32_t* itr = std::find(start, end, active_neuron);
  if (itr == end) {
    return {{}, 0.0};
  }
  uint32_t pos = std::distance(start, itr);
  return {pos, activations[pos]};
}

bool BoltVector::isDense() const { return this->neurons == nullptr; }

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

    uint32_t neuron = this->isDense() ? id : neurons[id];
    thresholded.push_back(neuron);
  }

  if (return_at_least_one && thresholded.empty()) {
    uint32_t max_act_neuron = this->isDense() ? ids[0] : neurons[ids[0]];
    thresholded.push_back(max_act_neuron);
  }

  return thresholded;
}

TopKActivationsQueue BoltVector::findKLargestActivations(uint32_t k) const {
  TopKActivationsQueue top_k;
  for (uint32_t pos = 0; pos < std::min(k, len); pos++) {
    uint32_t idx = isDense() ? pos : neurons[pos];
    top_k.push({activations[pos], idx});
  }
  for (uint32_t pos = k; pos < len; pos++) {
    uint32_t idx = isDense() ? pos : neurons[pos];
    ValueIndexPair val_idx_pair = {activations[pos], idx};
    // top_k.top() is minimum element.
    if (val_idx_pair > top_k.top()) {
      top_k.pop();
      top_k.push(val_idx_pair);
    }
  }
  return top_k;
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
      out << "(" << vec.neurons[i] << ", " << vec.activations[i] << ")";
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

void BoltVector::freeMemory() {  // NOLINT clang tidy thinks this method should
                                 // be const
  delete[] this->neurons;
  delete[] this->activations;
  delete[] this->gradients;
}

template <class Archive>
void BoltVector::save(Archive& archive) const {
  archive(len);
  bool is_sparse = !isDense();
  bool has_gradients = hasGradients();
  archive(is_sparse, has_gradients);

  if (is_sparse) {
    archive(cereal::binary_data(neurons, len * sizeof(uint32_t)));
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
    neurons = new uint32_t[len];
    archive(cereal::binary_data(neurons, len * sizeof(uint32_t)));
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

template void BoltVector::load(cereal::PortableBinaryInputArchive& archive);
template void BoltVector::load(cereal::BinaryInputArchive& archive);

template void BoltVector::save(
    cereal::PortableBinaryOutputArchive& archive) const;
template void BoltVector::save(cereal::BinaryOutputArchive& archive) const;

}  // namespace thirdai
