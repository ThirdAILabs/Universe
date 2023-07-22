#pragma once

#include <dataset/src/mach/MachIndex.h>
#include <stdexcept>

namespace thirdai::data {

using dataset::mach::MachIndexPtr;

class State {
 public:
  explicit State(MachIndexPtr mach_index)
      : _mach_index(std::move(mach_index)) {}

  State() {}

  const auto& machIndex() const {
    if (!_mach_index) {
      throw std::invalid_argument(
          "Transformation state does not contain MachIndex.");
    }
    return _mach_index;
  }

 private:
  MachIndexPtr _mach_index = nullptr;
};

using StatePtr = std::shared_ptr<State>;

}  // namespace thirdai::data