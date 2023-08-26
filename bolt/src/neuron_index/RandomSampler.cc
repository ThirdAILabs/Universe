#include "RandomSampler.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <proto/neuron_index.pb.h>
#include <utils/Random.h>
#include <random>

namespace thirdai::bolt {

RandomSampler::RandomSampler(uint32_t layer_dim) : _rand_neurons(layer_dim) {
  std::mt19937 rng(global_random::nextSeed());
  std::iota(_rand_neurons.begin(), _rand_neurons.end(), 0);
  std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rng);
}

RandomSampler::RandomSampler(const proto::bolt::RandomNeuronIndex& rand_proto)
    : _rand_neurons(rand_proto.random_neurons().begin(),
                    rand_proto.random_neurons().end()) {}

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
                          const BoltVector* labels) const {
  assert(!output.isDense());

  uint32_t label_len = 0;
  if (labels) {
    label_len = std::min<uint64_t>(labels->len, output.len);
    std::copy(labels->active_neurons, labels->active_neurons + label_len,
              output.active_neurons);
  }

  // Hack to intepret the float as an integer without doing a conversion.
  uint32_t seed = *reinterpret_cast<uint32_t*>(&input.activations[0]);

  // This is because rand() is not threadsafe and because we want to make the
  // output more deterministic.
  uint64_t random_offset =
      hashing::simpleIntegerHash(seed) % _rand_neurons.size();

  uint64_t neurons_to_sample = output.len - label_len;

  wrapAroundCopy(/* src= */ _rand_neurons.data(),
                 /* src_len= */ _rand_neurons.size(),
                 /* dest= */ output.active_neurons + label_len,
                 /* copy_size= */ neurons_to_sample,
                 /* starting_offset= */ random_offset);
}

proto::bolt::NeuronIndex* RandomSampler::toProto() const {
  proto::bolt::NeuronIndex* index = new proto::bolt::NeuronIndex();

  auto* rand_index = index->mutable_random();

  *rand_index->mutable_random_neurons() = {_rand_neurons.begin(),
                                           _rand_neurons.end()};

  return index;
}

template void RandomSampler::serialize(cereal::BinaryInputArchive&);
template void RandomSampler::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RandomSampler::serialize(Archive& archive) {
  archive(cereal::base_class<NeuronIndex>(this), _rand_neurons);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::RandomSampler,
                               "thirdai::bolt::nn::RandomSampler")