#pragma once

#include "Input.h"
#include <tuple>

namespace thirdai::bolt {

class Input3D final : public Input {
 private:
  explicit Input3D(std::tuple<uint32_t, uint32_t, uint32_t> input_size_3d)
      : Input(std::get<0>(input_size_3d) * std::get<1>(input_size_3d) *
              std::get<2>(input_size_3d)),
        _input_size_3d(input_size_3d) {}

 public:
  static std::shared_ptr<Input3D> make(
      std::tuple<uint32_t, uint32_t, uint32_t> input_size_3d) {
    return std::shared_ptr<Input3D>(new Input3D(input_size_3d));
  }

  std::tuple<uint32_t, uint32_t, uint32_t> getOutputDim3D() const {
    return _input_size_3d;
  }

 private:
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Input>(this), _input_size_3d);
  }

  std::tuple<uint32_t, uint32_t, uint32_t> _input_size_3d;
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Input3D)