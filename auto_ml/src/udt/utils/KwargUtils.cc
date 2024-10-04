#include "KwargUtils.h"
#include <pybind11/pytypes.h>
#include <optional>

namespace thirdai::automl {

template <typename CppType, typename PyType>
std::optional<CppType> getArg(const py::kwargs& kwargs,
                              const std::string& key) {
  if (!kwargs.contains(key)) {
    return std::nullopt;
  }

  const auto& value = kwargs[py::str(key)];

  if (!py::isinstance<PyType>(value)) {
    return std::nullopt;
  }

  return value.cast<CppType>();
}

std::optional<float> floatArg(const py::kwargs& kwargs,
                              const std::string& key) {
  return getArg<float, py::float_>(kwargs, key);
}

std::optional<bool> boolArg(const py::kwargs& kwargs, const std::string& key) {
  return getArg<bool, py::bool_>(kwargs, key);
}

}  // namespace thirdai::automl