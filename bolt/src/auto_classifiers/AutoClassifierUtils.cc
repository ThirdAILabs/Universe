#include "AutoClassifierUtils.h"
#include <bolt/src/layers/BoltVector.h>

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

std::shared_ptr<FullyConnectedNetwork> AutoClassifierUtils::createNetwork(
    uint64_t input_dim, uint32_t n_classes, const std::string& model_size) {
  uint32_t hidden_layer_size =
      getHiddenLayerSize(model_size, n_classes, input_dim);

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_size);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};
  return std::make_shared<FullyConnectedNetwork>(std::move(configs), input_dim);
}

std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
AutoClassifierUtils::loadStreamingDataset(
    const std::string& filename,
    const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
        batch_processor,
    uint32_t batch_size) {
  std::shared_ptr<dataset::DataLoader> data_loader =
      std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

  auto dataset =
      std::make_shared<dataset::StreamingDataset<BoltBatch, BoltBatch>>(
          data_loader, batch_processor);
  return dataset;
}

void AutoClassifierUtils::train(
    std::shared_ptr<FullyConnectedNetwork>& model, const std::string& filename,
    const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
        batch_processor,
    uint32_t epochs, float learning_rate) {
  auto dataset = loadStreamingDataset(filename, batch_processor);

  CategoricalCrossEntropyLoss loss;

  if (!canLoadDatasetInMemory(filename)) {
    for (uint32_t e = 0; e < epochs; e++) {
      // Train on streaming dataset
      model->trainOnStream(dataset, loss, learning_rate);

      // Create new stream for next epoch with new data loader.
      dataset = loadStreamingDataset(filename, batch_processor);
    }

  } else {
    auto [train_data, train_labels] = dataset->loadInMemory();

    model->train(train_data, train_labels, loss, learning_rate, 1);
    model->freezeHashTables();
    model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }
}

void AutoClassifierUtils::predict(
    std::shared_ptr<FullyConnectedNetwork>& model, const std::string& filename,
    const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
        batch_processor,
    const std::optional<std::string>& output_filename,
    const std::vector<std::string>& class_id_to_class_name) {
  auto dataset = loadStreamingDataset(filename, batch_processor);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = dataset::SafeFileIO::ofstream(*output_filename);
  }

  auto print_predictions_callback = [&](const BoltBatch& outputs,
                                        uint32_t batch_size) {
    if (!output_file) {
      return;
    }
    for (uint32_t batch_id = 0; batch_id < batch_size; batch_id++) {
      uint32_t class_id = outputs[batch_id].getIdWithHighestActivation();
      (*output_file) << class_id_to_class_name[class_id] << std::endl;
    }
  };

  /*
    We are using predict with the stream directly because we only need a
    single pass through the dataset, so this is more memory efficient, and we
    don't have to worry about storing the activations in memory to compute the
    predictions, and can instead compute the predictions using the
    back_callback.
  */
  model->predictOnStream(dataset, /* use_sparse_inference= */ true,
                         /* metric_names= */ {"categorical_accuracy"},
                         print_predictions_callback);

  if (output_file) {
    output_file->close();
  }
}

uint32_t AutoClassifierUtils::getHiddenLayerSize(const std::string& model_size,
                                                 uint64_t n_classes,
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

float AutoClassifierUtils::getHiddenLayerSparsity(uint64_t layer_size) {
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

static constexpr uint64_t ONE_GB = 1 << 30;

uint64_t AutoClassifierUtils::getMemoryBudget(const std::string& model_size) {
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

std::optional<uint64_t> AutoClassifierUtils::getSystemRam() {
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

bool AutoClassifierUtils::canLoadDatasetInMemory(const std::string& filename) {
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