#pragma once

#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <data/src/TensorConversion.h>
#include <sstream>
#include <string>

namespace thirdai::mach {

bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model);

auto modifyForMach(const bolt::Model& model);

auto inferLabelValueFill(const bolt::Model& model);

}  // namespace thirdai::mach
