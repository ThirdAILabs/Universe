#include "MachNeuronIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <random>

namespace thirdai::bolt {

MachNeuronIndex::MachNeuronIndex(dataset::mach::MachIndexPtr mach_index)
    : _mach_index(std::move(mach_index)) {}

void MachNeuronIndex::query(const BoltVector& input, BoltVector& output,
                            const BoltVector* labels) const {
  (void)input;
  (void)labels;

  auto nonempty_buckets = _mach_index->nonemptyBuckets();

  assert(nonempty_buckets.size() <= output.len);

  if (nonempty_buckets.size() < output.len) {
    // Hack to intepret the float as an integer without doing a conversion.
    uint32_t seed = *reinterpret_cast<uint32_t*>(&input.activations[0]);
    std::mt19937 rng(seed);

    // Since the sparsity is set based off of the number of nonempty buckets, we
    // should never have more than one random neuron thus, we can avoid storing
    // a precomputed random neurons set like in the LSH neuron index.
    uint32_t n_buckets = _mach_index->numBuckets();

    for (uint32_t i = rng() % (n_buckets - output.len);
         i < n_buckets && nonempty_buckets.size() < output.len; i++) {
      nonempty_buckets.insert(i);
    }
  }

  uint32_t i = 0;
  for (uint32_t neuron : nonempty_buckets) {
    output.active_neurons[i++] = neuron;
    if (i == output.len) {
      break;
    }
  }
}

template void MachNeuronIndex::serialize(cereal::BinaryInputArchive&);
template void MachNeuronIndex::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachNeuronIndex::serialize(Archive& archive) {
  archive(cereal::base_class<NeuronIndex>(this), _mach_index);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE_WITH_NAME(thirdai::bolt::MachNeuronIndex,
                               "thirdai::bolt::nn::MachNeuronIndex")