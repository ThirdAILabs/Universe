#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai {

namespace licensing {
void activate(std::string api_key);
}  // namespace licensing

class CppClassifier {
 public:
  static std::shared_ptr<CppClassifier> load(const std::string& saved_model);

  uint32_t predict(const std::unordered_map<std::string, std::string>& input);

  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai