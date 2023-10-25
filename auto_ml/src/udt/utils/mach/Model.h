#pragma once

#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/TensorConversion.h>
#include <sstream>
#include <string>

namespace thirdai::automl::udt::utils::mach {

static bolt::ComputationPtr getEmbeddingComputation(const bolt::Model& model) {
  // This defines the embedding as the second to last computatation in the
  // computation graph.
  auto computations = model.computationOrder();
  return computations.at(computations.size() - 2);
}

static auto modifyForMach(const bolt::Model& model) {
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

static auto defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t num_buckets) {
  auto input = bolt::Input::make(input_dim);
  auto hidden =
      bolt::Embedding::make(hidden_dim, input_dim, "tanh", /* bias= */ false)
          ->apply(input);
  auto output =
      bolt::FullyConnected::make(
          num_buckets, hidden_dim, udt::utils::autotuneSparsity(num_buckets),
          "sigmoid", /* sampling= */ nullptr, /* use_bias= */ false)
          ->apply(hidden);
  auto labels = bolt::Input::make(num_buckets);
  auto loss = bolt::BinaryCrossEntropy::make(output, labels);
  return bolt::Model::make(/* inputs= */ {input}, /* outputs= */ {output},
                           /* losses= */ {loss});
}

static auto inferLabelValueFill(const bolt::Model& model) {
  return model.losses().front()->logitsSumToOne()
             ? data::ValueFillType::SumToOne
             : data::ValueFillType::Ones;
}

}  // namespace thirdai::automl::udt::utils::mach
