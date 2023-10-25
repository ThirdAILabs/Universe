#include "Pipeline.h"
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <data/src/ColumnMap.h>

namespace thirdai::data {

void Pipeline::buildExplanationMap(const ColumnMap& input, State& state,
                                   ExplanationMap& explanations) const {
  ColumnMap last_input = input;

  for (const auto& transformation : _transformations) {
    // Apply the transformation first to make sure that the input is valid.
    ColumnMap next_input = transformation->apply(last_input, state);
    transformation->buildExplanationMap(last_input, state, explanations);
    last_input = std::move(next_input);
  }
}

}  // namespace thirdai::data
