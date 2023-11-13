#pragma once

#include <data/src/transformations/Transformation.h>
#include <memory>
namespace thirdai::data {

class AddMachRlhfSamples final : public Transformation {
 public:
  AddMachRlhfSamples();

  static auto make() { return std::make_shared<AddMachRlhfSamples>(); }

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data