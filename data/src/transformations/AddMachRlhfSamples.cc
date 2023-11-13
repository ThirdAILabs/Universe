#include "AddMachRlhfSamples.h"
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <cstdint>

namespace thirdai::data {

AddMachRlhfSamples::AddMachRlhfSamples() {}

ColumnMap AddMachRlhfSamples::apply(ColumnMap columns, State& state) const {
  state.rlhfSampler().addSamples(columns);
  return columns;
}

template void AddMachRlhfSamples::serialize(cereal::BinaryInputArchive&);
template void AddMachRlhfSamples::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void AddMachRlhfSamples::serialize(Archive& archive) {
  archive(cereal::base_class<Transformation>(this));
}

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::AddMachRlhfSamples)