#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <optional>
#include <tuple>

namespace thirdai::bolt::seismic {

using Shape = std::tuple<size_t, size_t, size_t>;

class InputShapeData {
 public:
  InputShapeData(size_t subcube_dim, size_t patch_dim,
                 std::optional<size_t> max_pool_dim)
      : _subcube_shape(subcube_dim, subcube_dim, subcube_dim),
        _patch_shape(patch_dim, patch_dim, patch_dim) {
    if (max_pool_dim) {
      _max_pool = {*max_pool_dim, *max_pool_dim, *max_pool_dim};
    }

    checkNonzeroDims(_subcube_shape);
    checkNonzeroDims(_patch_shape);
    if (_max_pool) {
      checkNonzeroDims(*_max_pool);
    }

    if (!shapesAreMultiples(_subcube_shape, _patch_shape)) {
      throw std::invalid_argument(
          "Subcube shape must be a multiple of the patch shape.");
    }
    if (_max_pool && !shapesAreMultiples(_patch_shape, *_max_pool)) {
      throw std::invalid_argument(
          "Max pool shape must be a multiple of the patch shape.");
    }
  }

  InputShapeData() {}  // Should only be used for cerealization

  const Shape& subcubeShape() const { return _subcube_shape; }

  const Shape& patchShape() const { return _patch_shape; }

  const std::optional<Shape>& maxPool() const { return _max_pool; }

  size_t nPatches() const {
    auto [dim_x, dim_y, dim_z] = _subcube_shape;
    auto [patch_x, patch_y, patch_z] = _patch_shape;
    return (dim_x / patch_x) * (dim_y / patch_y) * (dim_z / patch_z);
  }

  size_t flattenedPatchDim() const {
    auto [patch_x, patch_y, patch_z] = _patch_shape;
    if (_max_pool) {
      auto [pool_x, pool_y, pool_z] = *_max_pool;
      return (patch_x / pool_x) * (patch_y / pool_y) * (patch_z / pool_z);
    }
    return patch_x * patch_y * patch_z;
  }

 private:
  static void checkNonzeroDims(const Shape& a) {
    auto [x, y, z] = a;
    if (x == 0 || y == 0 || z == 0) {
      throw std::invalid_argument(
          "Expected all dimensions of shape to be nonzero.");
    }
  }

  static bool shapesAreMultiples(const Shape& a, const Shape& b) {
    auto [a_x, a_y, a_z] = a;
    auto [b_x, b_y, b_z] = b;
    return (a_x % b_x == 0) && (a_y % b_y == 0) && (a_z % b_z == 0);
  }

  Shape _subcube_shape;
  Shape _patch_shape;
  std::optional<Shape> _max_pool;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_subcube_shape, _patch_shape, _max_pool);
  }
};

}  // namespace thirdai::bolt::seismic