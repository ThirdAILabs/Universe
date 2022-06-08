#include "TextClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

#if defined __linux
#include <sys/sysinfo.h>
#endif

#if defined __APPLE__
#include <sys/sysctl.h>
#endif

#if _WIN32
#include <windows.h>
#endif

namespace thirdai::bolt {

// Autotuning helper functions
static uint32_t getHiddenLayerSize(const std::string& model_size,
                                   uint64_t n_classes, uint64_t input_dim);
static float getHiddenLayerSparsity(uint64_t layer_size);
static std::optional<uint64_t> getSystemRam();
static uint64_t getMemoryBudget(const std::string& model_size);

TextClassifier::TextClassifier(const std::string& model_size,
                               uint32_t n_classes) {
  uint32_t input_dim = 100000;
  uint32_t hidden_layer_size =
      getHiddenLayerSize(model_size, n_classes, input_dim);

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_size);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);

  _batch_processor =
      std::make_shared<dataset::TextClassificationProcessor>(input_dim);

  _model->enableSparseInference();
}

void TextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  std::shared_ptr<dataset::DataLoader> data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);

  auto dataset = std::make_shared<dataset::StreamingDataset<BoltBatch>>(
      data_loader, _batch_processor);

  std::shared_ptr<LossFunction> loss =
      std::make_shared<CategoricalCrossEntropyLoss>();

  if (epochs == 1) {
    _model->trainOnStream(dataset, loss, learning_rate);
  } else {
    auto [train_data, train_labels] = dataset->loadInMemory();

    _model->train(train_data, train_labels, loss, learning_rate, 1);
    _model->enableSparseInference();
    _model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }
}

void TextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {
  std::shared_ptr<dataset::DataLoader> data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);

  auto dataset = std::make_shared<dataset::StreamingDataset<BoltBatch>>(
      data_loader, _batch_processor);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = std::ofstream(*output_filename);
  }

  auto callback = [&](const BoltBatch& outputs, uint32_t batch_size) {
    if (!output_file) {
      return;
    }
    for (uint32_t batch_id = 0; batch_id < batch_size; batch_id++) {
      float max_act = 0.0;
      uint32_t pred = 0;
      for (uint32_t i = 0; i < outputs[batch_id].len; i++) {
        if (outputs[batch_id].activations[i] > max_act) {
          max_act = outputs[batch_id].activations[i];
          if (outputs[batch_id].isDense()) {
            pred = i;
          } else {
            pred = outputs[batch_id].active_neurons[i];
          }
        }
      }

      (*output_file) << _batch_processor->getClassName(pred) << std::endl;
    }
  };

  _model->predictOnStream(dataset, {"categorical_accuracy"}, callback);

  if (output_file) {
    output_file->close();
  }
}

uint32_t getHiddenLayerSize(const std::string& model_size, uint64_t n_classes,
                            uint64_t input_dim) {
  /*
    Estimated num parameters = (input_dim + n_classes) * hidden_dim

    Estimated memory = (Estimated num parameters) * 16

    16 comes from 4 bytes per parameter, and 4x the number of parameters from
    gradients, momentum, velocity.

    We can get the maximum layer size by setting this equation equal to the
    amount of ram.
  */
  // TODO(nicholas): what if getSystemRam() fails?
  uint64_t memory_budget = getMemoryBudget(model_size);

  uint64_t hidden_layer_size = memory_budget / (16 * (input_dim + n_classes));

  return hidden_layer_size;
}

static constexpr uint64_t ONE_GB = 1 << 30;

uint64_t getMemoryBudget(const std::string& model_size) {
  std::regex small_re("[Ss]mall");
  std::regex medium_re("[Mm]edium");
  std::regex large_re("[Ll]arge");
  std::regex gig_re("[1-9]\\d* ?Gb");

  std::optional<uint64_t> system_ram_opt = getSystemRam();

  if (!system_ram_opt) {
    std::cout << "WARNING: Unable to determine total RAM on your machine. "
                 "Using default of 8Gb."
              << std::endl;
  }

  // If unable to find system RAM assume max ram is 8Gb
  uint64_t system_ram = getSystemRam().value_or(8 * ONE_GB);

  // For small models we use either 1Gb of 1/16th of the total RAM, whichever
  // is smaller.
  if (std::regex_search(model_size, small_re)) {
    return std::min<uint64_t>(system_ram / 16, ONE_GB);
  }

  // For medium models we use either 2Gb of 1/8th of the total RAM, whichever
  // is smaller.
  if (std::regex_search(model_size, medium_re)) {
    return std::min<uint64_t>(system_ram / 8, 2 * ONE_GB);
  }

  // For large models we use either 4Gb of 1/4th of the total RAM, whichever
  // is smaller.
  if (std::regex_search(model_size, large_re)) {
    return std::min<uint64_t>(system_ram / 4, 4 * ONE_GB);
  }

  if (std::regex_search(model_size, gig_re)) {
    uint64_t requested_size = std::stoull(model_size) * ONE_GB;

    if (requested_size > (system_ram / 2)) {
      std::cout << "WARNING: You have requested " << model_size
                << " for your text classifier. This is over 1/2 of the total "
                   "RAM on your machine."
                << std::endl;
    }

    return std::min<uint64_t>(requested_size, system_ram);
  }

  throw std::invalid_argument(
      "'model_size' parameter must be either 'small', 'medium', 'large', or "
      "a "
      "gigabyte size of the model, i.e. 5Gb");
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

}  // namespace thirdai::bolt