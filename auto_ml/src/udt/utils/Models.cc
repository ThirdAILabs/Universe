#include "Models.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/udt/Defaults.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::udt::utils {

ModelPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                    const config::ArgumentMap& args,
                    const std::optional<std::string>& model_config,
                    bool use_sigmoid_bce, bool mach) {
  if (model_config) {
    return utils::loadModel({input_dim}, output_dim, *model_config, mach);
  }
  uint32_t hidden_dim = args.get<uint32_t>("embedding_dimension", "integer",
                                           defaults::HIDDEN_DIM);
  bool use_tanh = args.get<bool>("use_tanh", "bool", defaults::USE_TANH);

  if (args.contains("use_bias")) {
    throw std::invalid_argument(
        "Option 'use_bias' has been depreciated. Please use 'hidden_bias' or "
        "'output_bias'.");
  }
  bool hidden_bias =
      args.get<bool>("hidden_bias", "bool", defaults::HIDDEN_BIAS);
  bool output_bias =
      args.get<bool>("output_bias", "bool", defaults::OUTPUT_BIAS);

  bool normalize_embeddings = args.get<bool>("normalize_embeddings", "bool",
                                             defaults::NORMALIZE_EMBEDDINGS);
  return utils::defaultModel(input_dim, hidden_dim, output_dim, use_sigmoid_bce,
                             use_tanh, /* hidden_bias= */ hidden_bias,
                             /* output_bias= */ output_bias, /* mach= */ mach,
                             /* normalize_embeddings= */ normalize_embeddings);
}

float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},    {900, 0.2},    {1800, 0.1},     {4000, 0.05},
      {10000, 0.02}, {20000, 0.01}, {1000000, 0.005}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return sparsity_values.back().second;
}

ModelPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                      uint32_t output_dim, bool use_sigmoid_bce, bool use_tanh,
                      bool hidden_bias, bool output_bias, bool mach,
                      bool normalize_embeddings) {
  auto input = bolt::Input::make(input_dim);

  const auto* hidden_activation = use_tanh ? "tanh" : "relu";

  auto hidden = bolt::Embedding::make(hidden_dim, input_dim, hidden_activation,
                                      /* bias= */ hidden_bias)
                    ->apply(input);

  if (normalize_embeddings) {
    hidden = bolt::LayerNorm::make()->apply(hidden);
  }

  // auto sparsity = autotuneSparsity(output_dim);
  auto sparsity = 1.0;
  const auto* activation = use_sigmoid_bce ? "sigmoid" : "softmax";
  auto output = bolt::FullyConnected::make(
                    output_dim, hidden->dim(), sparsity, activation,
                    /* sampling= */ nullptr, /* use_bias= */ output_bias)
                    ->apply(hidden);

  auto labels = bolt::Input::make(output_dim);

  bolt::LossPtr loss;
  if (use_sigmoid_bce) {
    loss = bolt::BinaryCrossEntropy::make(output, labels);
  } else {
    loss = bolt::CategoricalCrossEntropy::make(output, labels);
  }

  bolt::ComputationList additional_labels;
  if (mach) {
    // For mach we need the hash based labels for training, but the actual
    // document/class ids to compute metrics. Hence we add two labels to the
    // model.
    additional_labels.push_back(
        bolt::Input::make(std::numeric_limits<uint32_t>::max()));
  }

  auto model = bolt::Model::make({input}, {output}, {loss}, additional_labels);

  return model;
}

ModelPtr loadModel(const std::vector<uint32_t>& input_dims,
                   uint32_t specified_output_dim,
                   const std::string& config_path, bool mach) {
  config::ArgumentMap parameters;
  parameters.insert("output_dim", specified_output_dim);

  auto json_config = json::parse(config::loadConfig(config_path));

  auto model = config::buildModel(json_config, parameters, input_dims, mach);

  uint32_t actual_output_dim = model->outputs().at(0)->dim();
  if (actual_output_dim != specified_output_dim) {
    throw std::invalid_argument(
        "Expected model with output dim " +
        std::to_string(specified_output_dim) +
        ", but the model config yielded a model with output dim " +
        std::to_string(actual_output_dim) + ".");
  }

  return model;
}

void verifyCanSetModel(const ModelPtr& curr_model, const ModelPtr& new_model) {
  auto vec_eq = [](const auto& a, const auto& b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (uint32_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };

  if (!vec_eq(curr_model->inputDims(), new_model->inputDims())) {
    throw std::invalid_argument("Input dim mismatch in set_model.");
  }

  if (new_model->outputs().size() != 1 ||
      new_model->outputs().at(0)->dim() != curr_model->outputs().at(0)->dim()) {
    throw std::invalid_argument("Output dim mismatch in set_model.");
  }

  if (!vec_eq(curr_model->labelDims(), new_model->labelDims())) {
    throw std::invalid_argument("Label dim mismatch in set_model.");
  }
}

bool hasSoftmaxOutput(const ModelPtr& model) {
  auto outputs = model->outputs();
  if (outputs.size() > 1) {
    return false;  // TODO(Nicholas): Should this throw?
  }

  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
  return fc && (fc->kernel()->getActivationFunction() ==
                bolt::ActivationFunction::Softmax);
}

}  // namespace thirdai::automl::udt::utils