#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>

namespace thirdai::automl::udt::utils {

bolt::nn::model::ModelPtr buildModel(
    uint32_t input_dim, uint32_t output_dim, const config::ArgumentMap& args,
    const std::optional<std::string>& model_config,
    bool use_sigmoid_bce = false);

bolt::nn::model::ModelPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                                       uint32_t output_dim,
                                       bool use_sigmoid_bce = false);

bolt::nn::model::ModelPtr loadModel(const std::vector<uint32_t>& input_dims,
                                    uint32_t output_dim,
                                    const std::string& config_path);

bool hasSoftmaxOutput(const bolt::nn::model::ModelPtr& model);

}  // namespace thirdai::automl::udt::utils