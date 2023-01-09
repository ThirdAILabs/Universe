#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include "Input.h"
#include <tuple>

namespace thirdai::bolt {

/**
 * The Input3D node functions is a shallow wrapper of the normal Input node that adds functionality for interpreting the single dimensional input vectors as 3D tensors.
 * the input dimension is specified with three values: width, height, and depth.
 * Input3D will then note these values and assume input BoltVectors of total
 * dimension = width * height * depth. This is useful to provide additional
 * information for treating input vectors as 3D (e.g. for image processing)
 * while still passing in typical 1D BoltVectors as input.
 *
 * The notion of sparsity is loosely defined in this format. You may use this
 * node however you wish, it is only a means to provide more information to
 * future nodes. Whether we should refactor Bolt to support a more natively
 * supported multi-dimensional representations is a different question and a
 * much harder sell overall.
 *
 * @param input_size_3d A tuple of (width, height, depth).
 */
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
  Input3D() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::tuple<uint32_t, uint32_t, uint32_t> _input_size_3d;
};

}  // namespace thirdai::bolt

// we need to force dynamic init for cereal to work, otherwise registration of
// the object will never take place. see here for more info:
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a01ebe0f840ac20c307f64622384e4dae
// we NOLINT since clang-tidy gets mad but cereal wants the class named this way
CEREAL_FORCE_DYNAMIC_INIT(Input3D)  // NOLINT