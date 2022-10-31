#pragma once

#include <memory>
#include <string>

namespace thirdai::automl::deployment {

class UniversalDeepTransformerBase {
 public:
  UniversalDeepTransformerBase() {}
  virtual void save(const std::string& filename) = 0;
  static std::unique_ptr<UniversalDeepTransformerBase> load(
      const std::string& filename);

  virtual ~UniversalDeepTransformerBase() = default;
};

}  // namespace thirdai::automl::deployment