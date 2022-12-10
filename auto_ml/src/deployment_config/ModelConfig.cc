#include "ModelConfig.h"
#include <cereal/archives/portable_binary.hpp>
#include <dataset/src/utils/SafeFileIO.h>

namespace thirdai::automl::deployment {

bolt::BoltGraphPtr ModelConfig::createModel(
    std::vector<bolt::InputPtr> inputs,
    const UserInputMap& user_specified_parameters) const {
  if (_input_names.size() != inputs.size()) {
    throw std::invalid_argument(
        "Number of inputs in model config does not match number of inputs "
        "returned from data loader.");
  }

  PredecessorsMap predecessors;
  for (uint32_t i = 0; i < _input_names.size(); i++) {
    predecessors.insert(/* name= */ _input_names[i], /* node= */ inputs[i]);
  }

  for (uint32_t i = 0; i < _nodes.size() - 1; i++) {
    auto node = _nodes[i]->createNode(predecessors, user_specified_parameters);
    predecessors.insert(/* name= */ _nodes[i]->name(), /* node= */ node);
  }

  auto output =
      _nodes.back()->createNode(predecessors, user_specified_parameters);
  // This is to check that there is not another node with this name.
  predecessors.insert(/* name= */ _nodes.back()->name(), /* node= */ output);

  auto model = std::make_shared<bolt::BoltGraph>(inputs, output);

  model->compile(_loss);

  return model;
}

void ModelConfig::save(const std::string& filename) {
  auto output = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::PortableBinaryOutputArchive oarchive(output);
  oarchive(*this);
}

std::shared_ptr<ModelConfig> ModelConfig::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::PortableBinaryInputArchive iarchive(filestream);
  std::shared_ptr<ModelConfig> deserialize_into(new ModelConfig());
  iarchive(*deserialize_into);

  return deserialize_into;
}
}  // namespace thirdai::automl::deployment