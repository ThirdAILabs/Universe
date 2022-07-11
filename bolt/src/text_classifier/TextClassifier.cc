#include "TextClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace thirdai::bolt {

TextClassifier::TextClassifier(const std::string& model_size,
                               uint32_t n_classes) {
  uint32_t input_dim = 100000;
  _model = AutoTuneUtils::createClassifierModel(/* input_dim */ input_dim,
                                                /* n_classes */ n_classes,
                                                model_size);
  _batch_processor =
      std::make_shared<dataset::TextClassificationProcessor>(input_dim);
}

void TextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  AutoTuneUtils::train(
      _model, filename,
      std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
          _batch_processor),
      /* epochs */ epochs,
      /* learning_rate */ learning_rate);
}

void TextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {
  AutoTuneUtils::predict(
      _model, filename,
      std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
          _batch_processor),
      output_filename, _batch_processor->getClassIdToNames());
}

}  // namespace thirdai::bolt