#include "MultiLabelTextClassifier.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/Graph.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
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

static uint32_t getHiddenLayerSize(const std::string& model_size,
                                   uint64_t n_classes, uint64_t input_dim);
float getHiddenLayerSparsity(uint64_t layer_size);
static std::optional<uint64_t> getSystemRam();
static uint64_t getMemoryBudget(const std::string& model_size);
static bool canLoadDatasetInMemory(const std::string& filename);

MultiLabelTextClassifier::MultiLabelTextClassifier(uint32_t n_classes) {

  uint32_t input_dim = 100000;
  uint32_t hidden_layer_dim = 1024;
  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_dim);
  
  auto input = std::make_shared<Input>(/* dim= */ input_dim);

  auto hidden = std::make_shared<FullyConnectedNode>(/* dim= */ hidden_layer_dim, /* sparsity= */ hidden_layer_sparsity, /* activation= */ "relu");
  hidden->addPredecessor(input);

  auto output = std::make_shared<FullyConnectedNode>(/* dim= */ n_classes, /* sparsity= */ 0.1, /* activation= */ "sigmoid");
  output->addPredecessor(hidden);

  _model = std::make_shared<BoltGraph>(/* inputs= */ std::vector<InputPtr>{input}, /* output= */ output);

  _model->compile(std::make_shared<BinaryCrossEntropyLoss>(), /* print_when_done= */ true);

  std::vector<std::shared_ptr<dataset::Block>> input_block = {
    std::make_shared<dataset::TextBlock>(/* col= */ 1,/* encoding= */std::make_shared<dataset::PairGram>(100000))
  };

  std::vector<std::shared_ptr<dataset::Block>> label_block = {
    std::make_shared<dataset::CategoricalBlock>(/* col= */ 0, /* encoding= */ std::make_shared<dataset::CategoricalMultiLabel>(n_classes))
  };

  _batch_processor =
      std::make_shared<dataset::GenericBatchProcessor>(std::move(input_block), std::move(label_block), /* has_header= */ false, /* delimiter= */ '\t');

}

void MultiLabelTextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  auto dataset = loadStreamingDataset(filename);

  BinaryCrossEntropyLoss loss;

  // Assume Wayfair's data can fit in memory
  auto [train_data, train_labels] = dataset->loadInMemory();

  auto train_cfg = TrainConfig::makeConfig(/* learning_rate= */ learning_rate, /* epochs= */ 1);

  _model->train(train_data, train_labels, train_cfg);
  //_model->freezeHashTables(/* insert_labels_if_not_found= */ true);

  train_cfg = TrainConfig::makeConfig(/* learning_rate= */ learning_rate, /* epochs= */ epochs-1);
  _model->train(train_data, train_labels, train_cfg);
  
}

void MultiLabelTextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename,
    const float threshold) {
  auto dataset = loadStreamingDataset(filename);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = dataset::SafeFileIO::ofstream(*output_filename);
  }

  std::stringstream metric;
  metric << "f_measure(" << threshold << ")";

  PredictConfig predict_cfg = PredictConfig::makeConfig().withMetrics({metric.str()}).returnActivations();

  auto [test_data, test_labels] = dataset->loadInMemory();

  auto [_, predict_output] = _model->predict(test_data, test_labels, predict_cfg);
  for (uint32_t vec_id = 0; vec_id < predict_output.getNumSavedVectors(); vec_id++) {
    BoltVector v = predict_output.getOutputVector(vec_id);
    auto predictions = v.getThresholdedNeurons(/* activation_threshold = */ threshold, /* return_at_least_one = */ true, /* max_count_to_return = */ 4);
    for (uint32_t i = 0; i < predictions.size(); i++) {
      (*output_file) << predictions.at(i);
      if (i != predictions.size() - 1) {
        (*output_file) << ",";
      }
    }
    (*output_file) << std::endl;
  }
  
  if (output_file) {
    output_file->close();
  }
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

  // For small models we use either 1Gb of 1/16th of the total RAM, whichever is
  // smaller.
  if (std::regex_search(model_size, small_re)) {
    return std::min<uint64_t>(system_ram / 16, ONE_GB);
  }

  // For medium models we use either 2Gb of 1/8th of the total RAM, whichever is
  // smaller.
  if (std::regex_search(model_size, medium_re)) {
    return std::min<uint64_t>(system_ram / 8, 2 * ONE_GB);
  }

  // For large models we use either 4Gb of 1/4th of the total RAM, whichever is
  // smaller.
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
      "'model_size' parameter must be either 'small', 'medium', 'large', or a "
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