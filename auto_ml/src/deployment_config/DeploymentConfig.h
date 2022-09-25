#pragma once

#include "DatasetConfig.h"
#include "ModelConfig.h"
#include "TrainEvalParameters.h"
#include <memory>

namespace thirdai::automl::deployment_config {

class DeploymentConfig {
 public:
  DeploymentConfig(DatasetConfigPtr dataset_config, ModelConfigPtr model_config,
                   TrainEvalParameters train_test_parameters)
      : _dataset_config(std::move(dataset_config)),
        _model_config(std::move(model_config)),
        _train_test_parameters(std::move(train_test_parameters)) {}

  std::pair<DatasetStatePtr, bolt::BoltGraphPtr> createDataLoaderAndModel(
      const std::string& option,
      const UserInputMap& user_specified_parameters) const {
    DatasetStatePtr dataset_state =
        _dataset_config->createDatasetState(option, user_specified_parameters);

    bolt::BoltGraphPtr model = _model_config->createModel(
        dataset_state->getInputNodes(), option, user_specified_parameters);

    return {std::move(dataset_state), model};
  }

  const TrainEvalParameters& parameters() const {
    return _train_test_parameters;
  }

 private:
  DatasetConfigPtr _dataset_config;
  ModelConfigPtr _model_config;
  TrainEvalParameters _train_test_parameters;
};

using DeploymentConfigPtr = std::shared_ptr<DeploymentConfig>;

}  // namespace thirdai::automl::deployment_config