#pragma once

#include <cereal/types/polymorphic.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <serialization/src/Archive.h>
#include <stdexcept>
#include <variant>

namespace thirdai::ar {

/**
 * The motivation behind this class is that we want to be able to save and load
 * parameters without having to copy them into the archive. This class stores a
 * reference to a parameter that it serializes, and only allocates storage for a
 * parameter when it is loaded, and then provides a mechanism for ops to request
 * that loaded parameter without copying it.
 */
class ParameterReference final : public Archive {
 public:
  ParameterReference(const std::vector<float>& parameter, bolt::OpPtr op)
      : _state(SavableState{&parameter, std::move(op)}) {}

  static auto make(const std::vector<float>& parameter, bolt::OpPtr op) {
    return std::make_shared<ParameterReference>(parameter, std::move(op));
  }

  std::shared_ptr<std::vector<float>> loadedParameter() const {
    if (std::holds_alternative<LoadedState>(_state)) {
      return std::get<LoadedState>(_state).parameter;
    }
    throw std::runtime_error(
        "Cannot access the parameter in a ParameterReference before saving and "
        "loading.");
  }

  std::vector<float> takeLoadedParameter() const {
    auto parameter = loadedParameter();
    return std::move(*parameter);
  }

  std::string type() const final { return "ParameterReference"; }

 private:
  struct SavableState {
    const std::vector<float>* parameter;
    // A shared_ptr to the op is stored to ensure that the lifetime of the op is
    // at least that of this reference.
    bolt::OpPtr op;
  };

  struct LoadedState {
    std::shared_ptr<std::vector<float>> parameter;
  };

  std::variant<SavableState, LoadedState> _state;

  ParameterReference() {}

  friend class cereal::access;

  template <typename Ar>
  void save(Ar& archive) const;

  template <typename Ar>
  void load(Ar& archive);
};

}  // namespace thirdai::ar

// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_FORCE_DYNAMIC_INIT(ParameterReference)  // NOLINT