#include "RandomSampler.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <random>

namespace thirdai::bolt::nn {

RandomSampler::RandomSampler(uint32_t layer_dim, std::random_device& rd)
    : _rand_neurons(layer_dim), _layer_dim(layer_dim) {
  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
}

static void wrapAroundCopy(const uint32_t* const src, uint64_t src_len,
                           uint32_t* const dest, uint64_t copy_size,
                           uint64_t starting_offset) {
  assert(starting_offset < src_len);
  assert(copy_size <= src_len);

  uint64_t length_to_end = std::min(src_len - starting_offset, copy_size);
  uint64_t length_of_remainder =
      length_to_end < copy_size ? copy_size - length_to_end : 0;

  std::copy(src + starting_offset, src + starting_offset + length_to_end, dest);
  std::copy(src, src + length_of_remainder, dest + length_to_end);
}

void RandomSampler::query(const BoltVector& input, BoltVector& output,
                          const BoltVector* labels, uint32_t sparse_dim) const {
  uint32_t label_len = 0;
  if (labels) {
    label_len = std::min<uint64_t>(labels->len, sparse_dim);
    std::copy(labels->active_neurons, labels->active_neurons + label_len,
              output.active_neurons);
  }

  // Hack to intepret the float as an integer without doing a conversion.
  uint32_t seed = *reinterpret_cast<uint32_t*>(&input.activations[0]);

  // This is because rand() is not threadsafe and because we want to make the
  // output more deterministic.
  uint64_t random_offset = hashing::simpleIntegerHash(seed) % _layer_dim;

  uint64_t neurons_to_sample = sparse_dim - label_len;

  wrapAroundCopy(/* src= */ _rand_neurons.data(), /* src_len= */ _layer_dim,
                 /* dest= */ output.active_neurons + label_len,
                 /* copy_size= */ neurons_to_sample,
                 /* starting_offset= */ random_offset);
}

template void RandomSampler::serialize(cereal::BinaryInputArchive&);
template void RandomSampler::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RandomSampler::serialize(Archive& archive) {
  archive(cereal::base_class<NeuronIndex>(this), _rand_neurons, _layer_dim);
}

}  // namespace thirdai::bolt::nn

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::RandomSampler)