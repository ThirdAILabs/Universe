#include "Model.h"
#include <bolt/src/nn/ops/Input.h>
#include <sstream>

namespace thirdai::mach {

bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
}

bolt::ModelPtr modifyForMach(const bolt::Model& model) {
  size_t n_inputs = model.inputs().size();
  size_t n_outputs = model.outputs().size();
  size_t n_labels = model.labels().size();
  size_t n_losses = model.losses().size();

  if (n_inputs != 1 || n_outputs != 1 || n_labels != 1 || n_losses != 1) {
    std::stringstream ss;
    ss << "Mach currently only supports models with one input, one output, one "
          "label, and one loss.\n"
       << "The given model has " << n_inputs << " inputs, " << n_outputs
       << " outputs, " << n_labels << " labels, and " << n_losses << " losses.";
    throw std::runtime_error(ss.str());
  }

  auto mach_bucket_label =
      bolt::Input::make(std::numeric_limits<uint32_t>::max());

  return bolt::Model::make(model.inputs(), model.outputs(), model.losses(),
                           /* additional_labels= */ {mach_bucket_label});
}

data::ValueFillType inferLabelValueFill(const bolt::Model& model) {
  return model.losses().front()->logitsSumToOne()
             ? data::ValueFillType::SumToOne
             : data::ValueFillType::Ones;
}

}  // namespace thirdai::mach