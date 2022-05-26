#include "TextClassifier.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <cctype>
#include <fstream>
#include <memory>
#include <stdexcept>
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
static uint32_t getHiddenLayerSize(std::string model_size, uint64_t n_classes,
                                   uint64_t input_dim);
static float getHiddenLayerSparsity(uint64_t layer_size);
static std::optional<uint64_t> getSystemRam();
static uint32_t getNumCpus();

TextClassifier::TextClassifier(const std::string& model_size,
                               uint32_t n_classes, uint32_t input_dim) {
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
}

void TextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  std::shared_ptr<dataset::DataLoader> data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);

  dataset::StreamingDataset dataset(data_loader, _batch_processor);

  CategoricalCrossEntropyLoss loss;

  if (epochs == 1) {
    trainOnStreamingDataset(dataset, loss, learning_rate);
  } else {
    auto in_memory_dataset = dataset.loadInMemory();

    _model->train(in_memory_dataset.data, in_memory_dataset.labels, loss,
                  learning_rate, epochs);
  }
}

void TextClassifier::trainOnStreamingDataset(dataset::StreamingDataset& dataset,
                                             const LossFunction& loss,
                                             float learning_rate) {
  _model->initializeNetworkState(dataset.getMaxBatchSize(), false);

  BoltBatch outputs = _model->getOutputs(dataset.getMaxBatchSize(), false);

  MetricAggregator metrics({});

  uint32_t rehash_batch = 0, rebuild_batch = 0;
  while (auto batch = dataset.nextBatch()) {
    _model->processTrainingBatch(batch->first, outputs, batch->second, loss,
                                 learning_rate, rehash_batch, rebuild_batch,
                                 metrics);
  }
}

void TextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {
  std::shared_ptr<dataset::DataLoader> data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, 256);

  dataset::StreamingDataset dataset(data_loader, _batch_processor);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = std::ofstream(*output_filename);
  }

  MetricAggregator metrics({"categorical_accuracy"});

  _model->initializeNetworkState(dataset.getMaxBatchSize(),
                                 /* force_dense= */ false);
  BoltBatch outputs =
      _model->getOutputs(dataset.getMaxBatchSize(), /* force_dense= */ false);

  while (auto batch = dataset.nextBatch()) {
    _model->processTestBatch(batch->first, outputs, &batch->second,
                             /* output_active_neurons= */ nullptr,
                             /* output_activations = */ nullptr, metrics,
                             /* compute_metrics= */ true);

    for (uint32_t batch_id = 0; batch_id < batch->first.getBatchSize();
         batch_id++) {
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
      if (output_file) {
        (*output_file) << _batch_processor->getClassName(pred) << std::endl;
      }
    }
  }

  metrics.logAndReset();

  if (output_file) {
    output_file->close();
  }
}

uint32_t getHiddenLayerSize(std::string model_size, uint64_t n_classes,
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
  uint64_t max_hidden_layer_size =
      getSystemRam().value() / (input_dim + n_classes);

  for (char& c : model_size) {
    c = std::tolower(c);
  }

  // Choose fraction of memory to use based off of specified model size
  uint64_t hidden_layer_size;
  if (model_size == "small") {
    hidden_layer_size = max_hidden_layer_size / 32;
  } else if (model_size == "medium") {
    hidden_layer_size = max_hidden_layer_size / 16;
  } else if (model_size == "large") {
    hidden_layer_size = max_hidden_layer_size / 8;
  } else {
    throw std::invalid_argument("Invalid model size paramter '" + model_size +
                                "'. Please use 'small', 'medium', or 'large'.");
  }

  // Update model size based off of number of cpus.
  uint32_t ncpus = getNumCpus();

  /*
   (0,6) CPUs is a smaller laptop so the layer size is decreased.
   [6, 12) CPUs is a large laptop or a small server so the size is unchanged.
   [12, ...) CPUs is a large server so we increase the layer size.
  */

  if (ncpus < 6) {
    // Likely a smaller laptop - decrease layer size.
    hidden_layer_size /= 2;
  } else if (ncpus >= 12) {
    // Larger server - increase layer size
    hidden_layer_size *= 2;
  }

  if (hidden_layer_size > 100000) {
    std::cout << "Warning: text classifier autotune returned oversized layer: "
              << hidden_layer_size << std::endl;
    hidden_layer_size = 100000;
  }

  return hidden_layer_size;
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

uint32_t getNumCpus() { return std::thread::hardware_concurrency(); }

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