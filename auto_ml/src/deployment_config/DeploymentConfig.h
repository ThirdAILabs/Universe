#pragma once

#include <cereal/access.hpp>
#include <cereal/specialize.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_set.hpp>
#include "DatasetConfig.h"
#include "ModelConfig.h"
#include "TrainEvalParameters.h"
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::automl::deployment_config {

class DeploymentConfig {
 public:
  DeploymentConfig(DatasetConfigPtr dataset_config, ModelConfigPtr model_config,
                   TrainEvalParameters train_test_parameters,
                   std::vector<std::string> available_options)
      : _dataset_config(std::move(dataset_config)),
        _model_config(std::move(model_config)),
        _train_test_parameters(std::move(train_test_parameters)),
        _available_options(available_options.begin(), available_options.end()) {
  }

  std::pair<DatasetLoaderFactoryPtr, bolt::BoltGraphPtr>
  createDataLoaderAndModel(
      const std::optional<std::string>& option,
      const UserInputMap& user_specified_parameters) const {
    if (!_available_options.empty() && !option) {
      throw std::invalid_argument(
          "Must specify an size option to instantiate this model "
          "configuration.");
    }

    if (option.has_value() && !_available_options.count(option.value())) {
      throw std::invalid_argument(
          "Option parameter '" + option.value() +
          "' was not found in list of available options.");
    }

    DatasetLoaderFactoryPtr dataset_state =
        _dataset_config->createDatasetState(option, user_specified_parameters);

    bolt::BoltGraphPtr model = _model_config->createModel(
        dataset_state->getInputNodes(), option, user_specified_parameters);

    return {std::move(dataset_state), model};
  }

  const TrainEvalParameters& parameters() const {
    return _train_test_parameters;
  }

  void save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::shared_ptr<DeploymentConfig> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::shared_ptr<DeploymentConfig> deserialize_into(new DeploymentConfig());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  DatasetConfigPtr _dataset_config;
  ModelConfigPtr _model_config;
  TrainEvalParameters _train_test_parameters;
  std::unordered_set<std::string> _available_options;

  // Private constructor for cereal
  DeploymentConfig() : _train_test_parameters({}, {}, {}, {}, {}) {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_dataset_config, _model_config, _train_test_parameters,
            _available_options);
  }
};

using DeploymentConfigPtr = std::shared_ptr<DeploymentConfig>;

}  // namespace thirdai::automl::deployment_config