#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>

namespace thirdai::automl::udt::utils {

using bolt::ModelPtr;

ModelPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                    const config::ArgumentMap& args,
                    const std::optional<std::string>& model_config,
                    bool use_sigmoid_bce = false, bool mach = false, std::optional<size_t> splade_featurization_range = std::nullopt);

ModelPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                      uint32_t output_dim, bool use_sigmoid_bce = false,
                      bool use_tanh = false, bool hidden_bias = true,
                      bool output_bias = true, bool mach = false,
                      bool normalize_embeddings = false, std::optional<size_t> splade_featurization_range = std::nullopt);

float autotuneSparsity(uint32_t dim);

ModelPtr loadModel(const std::vector<uint32_t>& input_dims,
                   uint32_t specified_output_dim,
                   const std::string& config_path, bool mach = false);

void verifyCanSetModel(const ModelPtr& curr_model, const ModelPtr& new_model);

bool hasSoftmaxOutput(const ModelPtr& model);

}  // namespace thirdai::automl::udt::utils