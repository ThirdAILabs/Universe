#pragma once

#include <memory>
#include <string>
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::automl::deployment {

class UniversalDeepTransformerBase {
 public:
  UniversalDeepTransformerBase() {}
  virtual void save(const std::string& filename) = 0;
  static std::unique_ptr<UniversalDeepTransformerBase> load(
      const std::string& filename);

  virtual ~UniversalDeepTransformerBase() = default;

 protected:
  //  explicit UniversalDeepTransformerBase()
  static UniversalDeepTransformerBase buildUDT();

private:

friend class cereal::access;
template <class Archive>
void serialize(Archive& archive) {
  archive()
}
};

}  // namespace thirdai::automl::deployment