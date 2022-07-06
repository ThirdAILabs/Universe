#include "MultiLabelTextClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <_types/_uint32_t.h>
#include <sys/_types/_u_int32_t.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

namespace thirdai::bolt {

static bool canLoadDatasetInMemory(const std::string& filename);

MultiLabelTextClassifier::MultiLabelTextClassifier(uint32_t input_dim, uint32_t hidden_layer_dim, uint32_t n_classes) {

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_dim);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_dim, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Sigmoid)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);

  _batch_processor =
      std::make_shared<dataset::TextClassificationProcessor>(input_dim);

  _model->freezeHashTables();
}

void MultiLabelTextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  auto dataset = loadStreamingDataset(filename);

  BinaryCrossEntropyLoss loss;

  if (!canLoadDatasetInMemory(filename)) {
    for (uint32_t e = 0; e < epochs; e++) {
      // Train on streaming dataset
      _model->trainOnStream(dataset, loss, learning_rate);

      // Create new stream for next epoch with new data loader.
      dataset = loadStreamingDataset(filename);
    }

  } else {
    auto [train_data, train_labels] = dataset->loadInMemory();

    _model->train(train_data, train_labels, loss, learning_rate, 1);
    _model->freezeHashTables();
    _model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }
}

void MultiLabelTextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename,
    const float threshold = 0.8) {
  auto dataset = loadStreamingDataset(filename);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = dataset::SafeFileIO::ofstream(*output_filename);
  }

  auto print_predictions_callback = [&](const BoltBatch& outputs,
                                        uint32_t batch_size) {
    if (!output_file) {
      return;
    }
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      uint32_t pred = 0;
      for (uint32_t i = 0; i < outputs[batch_idx].len; i++) {
        if (outputs[batch_idx].activations[i] > threshold) {
          if (outputs[batch_idx].isDense()) {
            pred = i;
          } else {
            pred = outputs[batch_idx].active_neurons[i];
          }
          (*output_file) << _batch_processor->getClassName(pred) << std::endl;
        }
      }
    }
  };

  /*
    We are using predict with the stream directly because we only need a single
    pass through the dataset, so this is more memory efficient–—we don't
    have to worry about storing the activations in memory to compute the
    predictions, and can instead compute the predictions using the
    back_callback.
  */
  _model->predictOnStream(dataset, /* use_sparse_inference= */ true,
                          /* metric_names= */ {"categorical_accuracy"},
                          print_predictions_callback);

  if (output_file) {
    output_file->close();
  }
}


float getHiddenLayerSparsity(uint64_t layer_size) {
  if (layer_size < 1000) {
    return 0.2;
  }
  if (layer_size < 4000) {
    return 0.1;
  }
  if (layer_size < 10000) {
    return 0.05;
  }
  if (layer_size < 30000) {
    return 0.01;
  }
  return 0.005;
}

std::optional<uint64_t> getSystemRam() {
#if defined __linux__
  // https://stackoverflow.com/questions/349889/how-do-you-determine-the-amount-of-linux-system-ram-in-c
  struct sysinfo mem_info;
  if (sysinfo(&mem_info) == 0) {
    return mem_info.totalram;
  }
#elif defined __APPLE__
  // https://stackoverflow.com/questions/8782228/retrieve-ram-info-on-a-mac
  int mib[] = {CTL_HW, HW_MEMSIZE};
  uint64_t mem_size = 0;
  size_t length = sizeof(mem_size);

  if (sysctl(mib, 2, &mem_size, &length, NULL, 0) == 0) {
    return mem_size;
  }
#elif _WIN32
  // https://stackoverflow.com/questions/2513505/how-to-get-available-memory-c-g
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  GlobalMemoryStatusEx(&status);
  return status.ullTotalPhys;
#endif
  return std::nullopt;
}

bool canLoadDatasetInMemory(const std::string& filename) {
  auto total_ram = getSystemRam().value();

#if defined(__APPLE__) || defined(__linux__)
  // TODO(Nicholas): separate file size method for windows
  struct stat file_stats;

  if (!stat(filename.c_str(), &file_stats)) {
    uint64_t file_size = file_stats.st_size;
    return total_ram / 2 >= file_size;
  }
#elif _WIN32
  // https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/stat-functions?redirectedfrom=MSDN&view=msvc-170

  struct _stat64 file_stats;
  if (!_stat64(filename.c_str(), &file_stats)) {
    uint64_t file_size = file_stats.st_size;
    return total_ram / 2 >= file_size;
  }

#endif

  throw std::runtime_error("Unable to get filesize of '" + filename + "'");
}


}  // namespace thirdai::bolt