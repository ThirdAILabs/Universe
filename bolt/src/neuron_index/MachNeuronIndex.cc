#include "MachNeuronIndex.h"
#include <hashing/src/HashUtils.h>

namespace thirdai::bolt::nn {

void MachNeuronIndex::query(const BoltVector& input, BoltVector& output,
                            const BoltVector* labels,
                            uint32_t sparse_dim) const {
  (void)input;
  (void)labels;

  auto nonempty_buckets = _mach_index->nonemptyBuckets();

  if (nonempty_buckets.size() < sparse_dim) {
    // Hack to intepret the float as an integer without doing a conversion.
    uint32_t seed = *reinterpret_cast<uint32_t*>(&input.activations[0]);

    // This is because rand() is not threadsafe and because we want to make the
    // output more deterministic.
    uint64_t random_offset =
        hashing::simpleIntegerHash(seed) % _rand_neurons.size();

    while (nonempty_buckets.size() < sparse_dim) {
      nonempty_buckets.insert(_rand_neurons[random_offset]);
      random_offset = (random_offset + 1) % _rand_neurons.size();
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

}  // namespace thirdai::bolt::nn