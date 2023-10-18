#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <cstddef>
#include <iterator>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::data {

using RlhfSample = std::pair<std::string, uint32_t>;

class RLHFSampler final : public Transformation {
 public:
  RLHFSampler() {}

  static auto make() { return std::make_shared<RLHFSampler>(); }

  ColumnMap apply(ColumnMap columns, State& state) const final {
    state.labelwiseSamples()->addSamples(columns);
    return columns;
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive();
  }
};

using RLHFSamplerPtr = std::shared_ptr<RLHFSampler>;

}  // namespace thirdai::data

CEREAL_REGISTER_TYPE(thirdai::data::RLHFSampler)