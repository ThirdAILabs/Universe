#pragma once

#include <bolt/src/nn/model/Model.h>
#include <data/src/TensorConversion.h>

namespace thirdai::mach {

bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model);

bolt::ModelPtr modifyForMach(const bolt::Model& model);

data::ValueFillType inferLabelValueFill(const bolt::Model& model);

}  // namespace thirdai::mach
