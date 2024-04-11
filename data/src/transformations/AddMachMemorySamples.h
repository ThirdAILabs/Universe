#pragma once

#include "Transformation.h"
#include <memory>

namespace thirdai::data {

class AddMachMemorySamples final : public Transformation {
 public:
  AddMachMemorySamples();

  explicit AddMachMemorySamples(const ar::Archive& archive) { (void)archive; }

  static auto make() { return std::make_shared<AddMachMemorySamples>(); }

  ColumnMap apply(ColumnMap columns, State& state) const final;

  static std::string type() { return "add_mach_memory_samples"; }

  ar::ConstArchivePtr toArchive() const final;
};

}  // namespace thirdai::data