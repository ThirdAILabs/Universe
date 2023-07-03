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

  bool hidden_bias =
      args.get<bool>("hidden_bias", "bool", defaults::HIDDEN_BIAS);
  bool output_bias =
      args.get<bool>("output_bias", "bool", defaults::OUTPUT_BIAS);
  return utils::defaultModel(input_dim, hidden_dim, output_dim, use_sigmoid_bce,
                             use_tanh, /* hidden_bias= */ hidden_bias,
                             /* output_bias= */ output_bias, /* mach= */ mach);
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
                      bool hidden_bias, bool output_bias, bool mach) {
  auto input = bolt::nn::ops::Input::make(input_dim);

  const auto* hidden_activation = use_tanh ? "tanh" : "relu";

  auto hidden = bolt::nn::ops::Embedding::make(hidden_dim, input_dim,
                                               hidden_activation, hidden_bias)
                    ->apply(input);

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = use_sigmoid_bce ? "sigmoid" : "softmax";
  auto output = bolt::nn::ops::FullyConnected::make(
                    output_dim, hidden->dim(), sparsity, activation,
                    /* sampling= */ nullptr, /* use_bias= */ output_bias)
                    ->apply(hidden);

  auto labels = bolt::nn::ops::Input::make(output_dim);

  bolt::nn::loss::LossPtr loss;
  if (use_sigmoid_bce) {
    loss = bolt::nn::loss::BinaryCrossEntropy::make(output, labels);
  } else {
    loss = bolt::nn::loss::CategoricalCrossEntropy::make(output, labels);
  }

  bolt::nn::autograd::ComputationList additional_labels;
  if (mach) {
    // For mach we need the hash based labels for training, but the actual
    // document/class ids to compute metrics. Hence we add two labels to the
    // model.
    additional_labels.push_back(
        bolt::nn::ops::Input::make(std::numeric_limits<uint32_t>::max()));
  }

  auto model = bolt::nn::model::Model::make({input}, {output}, {loss},
                                            additional_labels);

  return model;
}

ModelPtr loadModel(const std::vector<uint32_t>& input_dims, uint32_t output_dim,
                   const std::string& config_path, bool mach) {
  config::ArgumentMap parameters;
  parameters.insert("output_dim", output_dim);

  auto json_config = json::parse(config::loadConfig(config_path));

  return config::buildModel(json_config, parameters, input_dims, mach);
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

  auto fc = bolt::nn::ops::FullyConnected::cast(outputs.at(0)->op());
  return fc && (fc->kernel()->getActivationFunction() ==
                bolt::ActivationFunction::Softmax);
}

}  // namespace thirdai::automl::udt::utils