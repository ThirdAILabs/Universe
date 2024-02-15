#pragma once

#include <memory>
#include <optional>
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
  CppClassifier(std::shared_ptr<automl::Featurizer> featurizer,
                std::shared_ptr<bolt::Model> model,
                std::optional<float> binary_prediction_threshold);

  CppClassifier();

#if !_WIN32
  __attribute__((visibility("default")))
#endif
  static std::shared_ptr<CppClassifier>
  load(const std::string& saved_model);

#if !_WIN32
  __attribute__((visibility("default")))
#endif
  uint32_t
  predict(const std::unordered_map<std::string, std::string>& input);

  template <class Archive>
  void serialize(Archive& archive);

 private:
  std::shared_ptr<automl::Featurizer> _featurizer;
  std::shared_ptr<bolt::Model> _model;
  std::optional<float> _binary_prediction_threshold;
};

}  // namespace thirdai