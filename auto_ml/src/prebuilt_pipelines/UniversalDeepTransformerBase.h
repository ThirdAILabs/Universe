#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <memory>
#include <optional>
#include <string>

namespace thirdai::automl::deployment {

class UniversalDeepTransformerBase {
 public:
  virtual void save(const std::string& filename) = 0;

  virtual ~UniversalDeepTransformerBase() = default;

 private:
  std::optional<std::string> _name;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_name);
  }
};
using UniversalDeepTransformerBasePtr =
    std::shared_ptr<UniversalDeepTransformerBase>;

}  // namespace thirdai::automl::deployment