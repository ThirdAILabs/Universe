#include "MachNeuronIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>

namespace thirdai::bolt::nn {

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

    // Since the sparsity is set based off of the number of nonempty buckets, we
    // should never have more than one random neuron thus, we can avoid storing
    // a precomputed random neurons set like in the LSH neuron index.
    while (nonempty_buckets.size() < output.len) {
      // This is because rand() is not threadsafe and because we want to make
      // the output more deterministic.
      uint64_t random_neuron =
          hashing::simpleIntegerHash(seed) % _mach_index->numBuckets();

      nonempty_buckets.insert(random_neuron);
      seed = random_neuron;
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

}  // namespace thirdai::bolt::nn