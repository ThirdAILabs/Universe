#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_set.hpp>
#include "DatasetConfig.h"
#include "ModelConfig.h"
#include "TrainEvalParameters.h"
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <memory>
#include <stdexcept>
#include <unordered_set>

namespace thirdai::automl::deployment {

/**
 * The DeploymentConfig acts as a meta config that internally allows us to
 * specify a model architecture and corresponding dataset loader. The important
 * feature is that these meta configs can either have the parameters explicitly
 * defined or left as user supplied parameters, but explicitly state how that
 * user specified parameter is to be used. This gives us as much control as we
 * want over types and sizes of models used, but allows us to build in degrees
 * of customization as needed.
 *
 * This config is organized into three parts:
 *
 *    - DatasetLoaderFactoryConfig: specifies how data should be processed and
 *      featurized for the model.
 *
 *    - ModelConfig: this specifies the architecure of the model.
 *
 *    - TrainEvalParameters: specifies other parameters that are important for
 *      training/evaluation but that we don't want to expose to the user. Unlike
 *      the other configs, these parameters must be supplied as constants, and
 *      cannot be user supplied. This is to hide complexity like rebuilding hash
 *      tables, but still allowing us to customize the config for different
 *      tasks.
 *
 * When the config is used to create a ModelPipeline, the hyperparameters in the
 * DatasetLoaderFactoryConfig and ModelConfig are resolved using a list of user
 * supplied parameters.
 */
class DeploymentConfig {
 public:
  DeploymentConfig(DatasetLoaderFactoryConfigPtr dataset_config,
                   ModelConfigPtr model_config,
                   TrainEvalParameters train_test_parameters)
      : _dataset_config(std::move(dataset_config)),
        _model_config(std::move(model_config)),
        _train_test_parameters(train_test_parameters) {}

  std::pair<DatasetLoaderFactoryPtr, bolt::BoltGraphPtr>
  createDataLoaderAndModel(UserInputMap user_specified_parameters) const {
    DatasetLoaderFactoryPtr dataset_factory =
        _dataset_config->createDatasetState(user_specified_parameters);

    user_specified_parameters.emplace(
        DatasetLabelDimensionParameter::PARAM_NAME,
        UserParameterInput(dataset_factory->getLabelDim()));

    bolt::BoltGraphPtr model = _model_config->createModel(
        dataset_factory->getInputNodes(), user_specified_parameters);

    return {std::move(dataset_factory), std::move(model)};
  }

  const TrainEvalParameters& train_eval_parameters() const {
    return _train_test_parameters;
  }

  void save(const std::string& filename) {
    std::stringstream output;
    cereal::PortableBinaryOutputArchive oarchive(output);
    oarchive(*this);

    // We are applying a simple block cipher here because cereal leaks some
    // class names for polymorphic classes in the binary archive and we want to
    // hide that information from customers.
    // TODO(Nicholas): also add a checksum for the serialized config to make
    // sure customers do not recieve a corrupted file.
    std::string output_str = output.str();
    applyBlockCipher(output_str);

    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);

    filestream.write(output_str.data(), output_str.size());
  }

  static std::shared_ptr<DeploymentConfig> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);

    std::stringstream encrypted_buffer;
    // Converting contents of file into string:
    // https://stackoverflow.com/questions/2602013/read-whole-ascii-file-into-c-stdstring
    encrypted_buffer << filestream.rdbuf();
    std::string input_str = encrypted_buffer.str();
    applyBlockCipher(input_str);

    std::stringstream decrypted_buffer;
    decrypted_buffer.write(input_str.data(), input_str.size());

    cereal::PortableBinaryInputArchive iarchive(decrypted_buffer);
    std::shared_ptr<DeploymentConfig> deserialize_into(new DeploymentConfig());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

  // For more information on what a block cipher is:
  // https://en.wikipedia.org/wiki/Block_cipher
  static void applyBlockCipher(std::string& data, char cipher = '#') {
    for (char& c : data) {
      c ^= cipher;
    }
  }

 private:
  DatasetLoaderFactoryConfigPtr _dataset_config;
  ModelConfigPtr _model_config;

  // These are static parameters that need to be configurable for different
  // models, but that the user cannot modify.
  TrainEvalParameters _train_test_parameters;

  // Private constructor for cereal
  DeploymentConfig() : _train_test_parameters({}, {}, {}, {}, {}) {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_dataset_config, _model_config, _train_test_parameters);
  }
};

using DeploymentConfigPtr = std::shared_ptr<DeploymentConfig>;

}  // namespace thirdai::automl::deployment