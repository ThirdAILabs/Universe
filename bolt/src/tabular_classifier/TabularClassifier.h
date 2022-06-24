#pragma once

#include <cereal/archives/binary.hpp>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <memory>

namespace thirdai::bolt {

class TabularClassifier {
 public:
  TabularClassifier(const std::string& model_size, uint32_t n_classes);

  // void train(const std::string& filename, uint32_t epochs, float
  // learning_rate);

  // void predict(const std::string& filename,
  //              const std::optional<std::string>& output_filename);

  void save(const std::string& filename) {
    std::ofstream filestream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::unique_ptr<TabularClassifier> load(const std::string& filename) {
    std::ifstream filestream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<TabularClassifier> deserialize_into(
        new TabularClassifier());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  // Private constructor for cereal
  TabularClassifier() {}

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_model);
  }

  std::unique_ptr<FullyConnectedNetwork> _model;
};

}  // namespace thirdai::bolt