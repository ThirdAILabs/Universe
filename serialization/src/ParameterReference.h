#pragma once

#include <cereal/types/base_class.hpp>
#include <bolt/src/nn/ops/Op.h>
#include <serialization/src/Archive.h>
#include <stdexcept>
#include <variant>

namespace thirdai::serialization {

class ParameterReference final : public Archive {
 public:
  ParameterReference(const std::vector<float>& parameter, bolt::OpPtr op)
      : _state(SavableState{&parameter, std::move(op)}) {}

  std::shared_ptr<std::vector<float>> loadedParameter() const {
    if (std::holds_alternative<LoadedState>(_state)) {
      return std::get<LoadedState>(_state).parameter;
    }
    throw std::invalid_argument(
        "Parameter reference must be in loaded state to access the parameter.");
  }

  std::vector<float> takeLoadedParameter() const {
    auto parameter = loadedParameter();
    return std::move(*parameter);
  }

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

  template <typename Archive>
  void save(Archive& archive) const {
    if (!std::holds_alternative<SavableState>(_state)) {
      throw std::invalid_argument(
          "ParameterReference must be in saveable state to save.");
    }
    archive(cereal::base_class<Archive>(this),
            *std::get<SavableState>(_state).parameter);
  }

  template <typename Archive>
  void load(Archive& archive) {
    auto parameter = std::make_shared<std::vector<float>>();
    archive(cereal::base_class<Archive>(this), *parameter);
    _state = LoadedState{parameter};
  }
};

}  // namespace thirdai::serialization