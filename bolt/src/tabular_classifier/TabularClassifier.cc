#include "TabularClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

namespace thirdai::bolt {

TabularClassifier::TabularClassifier(const std::string& model_size,
                                     uint32_t n_classes) {
  // TODO(david) autotune these size depending on model_size and benchmark
  // results
  uint32_t input_dim = 100000;
  uint32_t hidden_layer_size = 1000;
  uint32_t hidden_layer_sparsity = 0.1;

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);
}

}  // namespace thirdai::bolt