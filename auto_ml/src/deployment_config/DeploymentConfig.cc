#include "DeploymentConfig.h"
#include <cereal/archives/portable_binary.hpp>

namespace thirdai::automl::deployment {

std::pair<data::DatasetLoaderFactoryPtr, bolt::BoltGraphPtr>
DeploymentConfig::createDataSourceAndModel(
    UserInputMap user_specified_parameters) const {
  data::DatasetLoaderFactoryPtr dataset_factory =
      _dataset_config->createDatasetState(user_specified_parameters);

  if (user_specified_parameters.count(
          DatasetLabelDimensionParameter::PARAM_NAME)) {
    std::stringstream ss;
    ss << "User specified parameter has reserved parameter name '"
       << DatasetLabelDimensionParameter::PARAM_NAME << "'.";
    throw std::invalid_argument(ss.str());
  }
  user_specified_parameters.emplace(
      DatasetLabelDimensionParameter::PARAM_NAME,
      UserParameterInput(dataset_factory->getLabelDim()));

  auto input_dims = dataset_factory->getInputDims();
  bolt::BoltGraphPtr model =
      _model_config->createModel(input_dims, user_specified_parameters);

  return {std::move(dataset_factory), std::move(model)};
}

void DeploymentConfig::save(const std::string& filename) {
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

std::shared_ptr<DeploymentConfig> DeploymentConfig::load(
    const std::string& filename) {
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

}  // namespace thirdai::automl::deployment
