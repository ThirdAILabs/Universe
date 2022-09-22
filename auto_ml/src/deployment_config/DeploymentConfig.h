#pragma once

#include "DatasetConfig.h"
#include "ModelConfig.h"
#include "TrainTestParameters.h"
#include <memory>

namespace thirdai::automl::deployment_config {

class DeploymentConfig {
 public:
  DeploymentConfig(DatasetConfigPtr dataset_config, ModelConfig model_config,
                   TrainTestParameters train_test_parameters)
      : _dataset_config(std::move(dataset_config)),
        _model_config(std::move(model_config)),
        _train_test_parameters(std::move(train_test_parameters)) {}

  std::pair<DatasetStatePtr, bolt::BoltGraphPtr> createDataLoaderAndModel(
      const std::string& option,
      const std::unordered_map<std::string, UserParameterInput>&
          user_specified_parameters) const {
    DatasetStatePtr dataset_state =
        _dataset_config->createDatasetState(option, user_specified_parameters);

    bolt::BoltGraphPtr model = _model_config.createModel(
        dataset_state->getInputNodes(), option, user_specified_parameters);

    return {std::move(dataset_state), model};
  }

  const TrainTestParameters& parameters() const {
    return _train_test_parameters;
  }

 private:
  DatasetConfigPtr _dataset_config;
  ModelConfig _model_config;
  TrainTestParameters _train_test_parameters;
};

}  // namespace thirdai::automl::deployment_config