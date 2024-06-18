#include "KwargUtils.h"
#include <pybind11/pytypes.h>
#include <optional>

namespace thirdai::automl {

std::optional<float> floatArg(const py::kwargs& kwargs,
                              const std::string& key) {
  if (!kwargs.contains(key)) {
    return std::nullopt;
  }

  const auto& value = kwargs[py::str(key)];

  if (!py::isinstance<py::float_>(value)) {
    return std::nullopt;
  }

  return value.cast<float>();
}

}  // namespace thirdai::automl