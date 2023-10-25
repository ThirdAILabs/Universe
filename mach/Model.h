#pragma once

#include <bolt/src/nn/model/Model.h>

namespace thirdai::mach {

bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model);

auto modifyForMach(const bolt::Model& model);

auto inferLabelValueFill(const bolt::Model& model);

}  // namespace thirdai::mach
