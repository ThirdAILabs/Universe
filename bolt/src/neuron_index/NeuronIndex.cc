#include "NeuronIndex.h"
#include <bolt/src/neuron_index/LshIndex.h>
#include <bolt/src/neuron_index/RandomSampler.h>

namespace thirdai::bolt {

NeuronIndexPtr NeuronIndex::fromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == LshIndex::type()) {
    return LshIndex::fromArchive(archive);
  }

  if (type == RandomSampler::type()) {
    return RandomSampler::fromArchive(archive);
  }

  return nullptr;
}
}  // namespace thirdai::bolt