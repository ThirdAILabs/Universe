#include "TabularClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
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
  _model = AutoTuneUtils::createClassifierModel(/* input_dim */ 100000,
                                                /* n_classes */ n_classes,
                                                model_size);
  _metadata = nullptr;
}

void TabularClassifier::train(const std::string& filename,
                              std::vector<std::string>& column_datatypes,
                              uint32_t epochs, float learning_rate) {
  if (_metadata) {
    std::cout << "Note: Metadata from the training dataset is used for "
                 "predictions on future test data. Calling train(..) again "
                 "resets this metadata."
              << std::endl;
  }
  _metadata = setTabularMetadata(filename, column_datatypes);

  std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
      makeTabularBatchProcessor();

  AutoTuneUtils::train(
      _model, filename,
      std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
          batch_processor),
      /* epochs */ epochs,
      /* learning_rate */ learning_rate);
}

void TabularClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {
  if (!_metadata) {
    throw std::invalid_argument(
        "Cannot call predict(..) without calling train(..) first.");
  }

  std::shared_ptr<dataset::GenericBatchProcessor> batch_processor =
      makeTabularBatchProcessor();

  AutoTuneUtils::predict(
      _model, filename,
      std::static_pointer_cast<dataset::BatchProcessor<BoltBatch>>(
          batch_processor),
      output_filename, _metadata->getClassIdToNames());
}

}  // namespace thirdai::bolt