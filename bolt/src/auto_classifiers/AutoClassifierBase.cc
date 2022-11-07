#include "AutoClassifierBase.h"
#include <bolt_vector/src/BoltVector.h>

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

AutoClassifierBase::AutoClassifierBase(uint64_t input_dim, uint32_t n_classes,
                                       const std::string& model_size) {
  uint32_t hidden_layer_size =
      getHiddenLayerSize(model_size, n_classes, input_dim);

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_size);

  auto input_layer = Input::make(input_dim);

  auto hidden_layer = FullyConnectedNode::makeAutotuned(
      /* dim= */ hidden_layer_size, /* sparsity= */ hidden_layer_sparsity,
      /* activation= */ "relu");

  hidden_layer->addPredecessor(input_layer);

  auto output_layer = FullyConnectedNode::makeDense(
      /* dim= */ n_classes, /* activation= */ "softmax");

  output_layer->addPredecessor(hidden_layer);

  _model = std::make_shared<BoltGraph>(std::vector<InputPtr>{input_layer},
                                       output_layer);

  _model->compile(std::make_shared<CategoricalCrossEntropyLoss>());
}

void AutoClassifierBase::train(
    const std::string& filename,
    const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
        batch_processor,
    uint32_t epochs, float learning_rate) {
  auto dataset = loadStreamingDataset(filename, batch_processor);

  // The case has yet to come up where loading the dataset in
  // memory is a concern. Additionally, supporting streaming datasets in the DAG
  // api would require a lot of messy refactoring. When the problem does come up
  // we can prioritize that and come up with a good solution in another PR
  // (TODO(david, nick)). In the meantime, the user can simply implement this
  // chunked processing themselves by training on sections of the dataset.
  if (!canLoadDatasetInMemory(filename)) {
    throw std::invalid_argument("Cannot load dataset in memory.");
  }
  auto [train_data, train_labels] = dataset->loadInMemory();

  TrainConfig first_epoch_config =
      TrainConfig::makeConfig(/* learning_rate= */ learning_rate,
                              /* epochs= */ 1)
          .withMetrics({"categorical_accuracy"});

  // TODO(david) verify freezing hash tables is good for autoclassifier
  // training. The only way we can really test this is when we have a validation
  // based early stop callback already implemented.
  _model->train({train_data}, train_labels, first_epoch_config);
  _model->freezeHashTables(/* insert_labels_if_not_found */ true);

  TrainConfig remaining_epochs_config =
      TrainConfig::makeConfig(/* learning_rate= */
                              learning_rate,
                              /* epochs= */ epochs - 1)
          .withMetrics({"categorical_accuracy"});

  _model->train({train_data}, train_labels, remaining_epochs_config);
}

void AutoClassifierBase::predict(
    const std::string& filename,
    const std::shared_ptr<dataset::BatchProcessor<BoltBatch, BoltBatch>>&
        batch_processor,
    const std::optional<std::string>& output_filename,
    const std::vector<std::string>& class_id_to_class_name) {
  auto dataset = loadStreamingDataset(filename, batch_processor);

  // see comment above in train(..) about loading in memory
  if (!canLoadDatasetInMemory(filename)) {
    throw std::invalid_argument("Cannot load dataset in memory.");
  }
  auto [test_data, test_labels] = dataset->loadInMemory();

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = dataset::SafeFileIO::ofstream(*output_filename);
  }

  auto print_predictions_callback = [&](const BoltVector& output) {
    if (!output_file) {
      return;
    }
    uint32_t class_id = output.getHighestActivationId();
    (*output_file) << class_id_to_class_name[class_id] << std::endl;
  };

  EvalConfig config = EvalConfig::makeConfig()
                          .enableSparseInference()
                          .withMetrics({"categorical_accuracy"})
                          .withOutputCallback(print_predictions_callback)
                          .silence();

  _model->evaluate({test_data}, test_labels, config);

  if (output_file) {
    output_file->close();
  }
}

BoltVector AutoClassifierBase::predictSingle(
    std::vector<BoltVector>&& test_data, bool use_sparse_inference) {
  return _model->predictSingle(std::move(test_data), use_sparse_inference);
}

std::shared_ptr<dataset::StreamingDataset<BoltBatch, BoltBatch>>
AutoClassifierBase::loadStreamingDataset(
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

uint32_t AutoClassifierBase::getHiddenLayerSize(const std::string& model_size,
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

float AutoClassifierBase::getHiddenLayerSparsity(uint64_t layer_size) {
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

constexpr uint64_t ONE_GB = 1 << 30;

uint64_t AutoClassifierBase::getMemoryBudget(const std::string& model_size) {
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
      "a gigabyte size of the model, i.e. 5Gb");
}

std::optional<uint64_t> AutoClassifierBase::getSystemRam() {
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

bool AutoClassifierBase::canLoadDatasetInMemory(const std::string& filename) {
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