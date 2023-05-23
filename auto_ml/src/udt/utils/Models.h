#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>

namespace thirdai::automl::udt::utils {

using bolt::nn::model::ModelPtr;

ModelPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                    const config::ArgumentMap& args,
                    const std::optional<std::string>& model_config,
                    bool use_sigmoid_bce = false);

ModelPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                      uint32_t output_dim, bool use_sigmoid_bce = false,
                      bool use_tanh = false, bool use_layer_norm = false,
                      bool use_bias = true);

ModelPtr loadModel(const std::vector<uint32_t>& input_dims, uint32_t output_dim,
                   const std::string& config_path);

void verifyCanSetModel(const ModelPtr& curr_model, const ModelPtr& new_model);

bool hasSoftmaxOutput(const ModelPtr& model);

}  // namespace thirdai::automl::udt::utils