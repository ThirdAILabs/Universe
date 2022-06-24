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
  (void)model_size;
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

void TabularClassifier::train(const std::string& filename, uint32_t epochs,
                              float learning_rate) {
  //   if metadata not initialized:
  //     train
  //   else
  //      undefined behavior/ what does this do?
  //      do we use the same metadata to process? or

  // training looks like this

  // read the csv into memory
  // read all the headers, ensure there is a "category" column
  // get the datatypes of all the headers
  //    if values in a column can be converted to floating point or integers,
  //    they will be numeric
  //        otherwise they are categorical
  //    record these datatypes in the metadata
  // for each column, if its a numeric datatype, apply the binning
  // afterwards, ensure the categories are unique
  // next, create capped pairgrams out of the categories
  // next, train the BOLT model
  //

  // metadata should have:
  //    column -> dtype, binning info if necessary
}

void TabularClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {}

}  // namespace thirdai::bolt