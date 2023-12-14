#pragma once

#include <smx/src/autograd/functions/Activations.h>
#include <smx/src/modules/Module.h>
#include <stdexcept>

namespace thirdai::smx {

class Activation final : public UnaryModule {
 public:
  explicit Activation(const std::string& type) {
    if (type == "relu") {
      _type = Type::relu;
    } else if (type == "tanh") {
      _type = Type::tanh;
    } else if (type == "sigmoid") {
      _type = Type::sigmoid;
    } else if (type == "softmax") {
      _type = Type::softmax;
    } else {
      throw std::invalid_argument(
          "Invalid activation function '" + type +
          "'. Please use 'relu', 'tanh', 'sigmoid', or 'softmax'.");
    }
  }

  VariablePtr forward(const VariablePtr& x) final {
    switch (_type) {
      case Type::relu:
        return smx::relu(x);
      case Type::tanh:
        return smx::tanh(x);
      case Type::sigmoid:
        return smx::sigmoid(x);
      case Type::softmax:
        return smx::softmax(x);
      default:
        throw std::invalid_argument("Invalid activation in switch.");
    }
  }

  std::vector<VariablePtr> parameters() const final { return {}; }

 private:
  enum Type { relu, tanh, sigmoid, softmax };

  Type _type;
};

}  // namespace thirdai::smx