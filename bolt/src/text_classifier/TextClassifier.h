#pragma once

#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <memory>

namespace thirdai::bolt {

class TextClassifier {
 public:
  TextClassifier(const std::string& model_size, uint32_t output_dim,
                 uint32_t input_dim = 100000);

 private:
  std::unique_ptr<FullyConnectedNetwork> _model;
};

}  // namespace thirdai::bolt