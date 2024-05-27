#include "AddMachMemorySamples.h"
#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <cstdint>

namespace thirdai::data {

AddMachMemorySamples::AddMachMemorySamples() {}

ColumnMap AddMachMemorySamples::apply(ColumnMap columns, State& state) const {
  state.machMemory().addSamples(columns);
  return columns;
}

ar::ConstArchivePtr AddMachMemorySamples::toArchive() const {
  auto map = ar::Map::make();
  map->set("type", ar::str(type()));
  return map;
}

}  // namespace thirdai::data
