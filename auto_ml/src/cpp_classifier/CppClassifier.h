#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai {

namespace bolt {
class Model;
}  // namespace bolt

namespace automl {
class Featurizer;
}  // namespace automl

class CppClassifier {
 public:
  static std::shared_ptr<CppClassifier> load(const std::string& saved_model);

  uint32_t predict(const std::unordered_map<std::string, std::string>& input);

  template <class Archive>
  void serialize(Archive& archive);

 private:
  std::shared_ptr<automl::Featurizer> _featurizer;
  std::shared_ptr<bolt::Model> _model;
};

}  // namespace thirdai