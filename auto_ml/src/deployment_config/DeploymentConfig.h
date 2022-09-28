#pragma once

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
                   TrainEvalParameters train_test_parameters)
      : _dataset_config(std::move(dataset_config)),
        _model_config(std::move(model_config)),
        _train_test_parameters(std::move(train_test_parameters)) {}

  std::pair<DatasetLoaderFactoryPtr, bolt::BoltGraphPtr>
  createDataLoaderAndModel(
      const UserInputMap& user_specified_parameters) const {
    DatasetLoaderFactoryPtr dataset_state =
        _dataset_config->createDatasetState(user_specified_parameters);

    bolt::BoltGraphPtr model = _model_config->createModel(
        dataset_state->getInputNodes(), user_specified_parameters);

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