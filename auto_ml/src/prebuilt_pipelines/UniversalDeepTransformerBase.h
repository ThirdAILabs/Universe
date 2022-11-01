#pragma once

#include <memory>
#include <string>

namespace thirdai::automl::deployment {

class UniversalDeepTransformerBase {
 public:
  UniversalDeepTransformerBase() {}
  void save(const std::string& filename);
  static std::unique_ptr<UniversalDeepTransformerBase> load(
      const std::string& filename);

  virtual ~UniversalDeepTransformerBase() = default;

protected:
    static UniversalDeepTransformerBase buildUDT();
};

}  // namespace thirdai::automl::deployment