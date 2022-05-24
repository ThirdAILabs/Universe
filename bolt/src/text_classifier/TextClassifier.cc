#include "TextClassifier.h"
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <cctype>
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
static uint32_t getHiddenLayerSize(std::string model_size, uint64_t output_dim,
                                   uint64_t input_dim);

static float getHiddenLayerSparsity(uint64_t layer_size);

static std::optional<uint64_t> getSystemRam();

static uint32_t getNumCpus();

TextClassifier::TextClassifier(const std::string& model_size,
                               uint32_t output_dim, uint32_t input_dim) {
  uint32_t hidden_layer_size =
      getHiddenLayerSize(model_size, output_dim, input_dim);

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_size);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(output_dim,
                                                  ActivationFunction::Softmax)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);
}

uint32_t getHiddenLayerSize(std::string model_size, uint64_t output_dim,
                            uint64_t input_dim) {
  /*
    Estimated num parameters = (input_dim + output_dim) * hidden_dim

    Estimated memory = (Estimated num parameters) * 16

    16 comes from 4 bytes per parameter, and 4x the number of parameters from
    gradients, momentum, velocity.

    We can get the maximum layer size by setting this equation equal to the
    amount of ram.
  */
  // TODO(nicholas): what if getSystemRam() fails?
  uint64_t max_hidden_layer_size =
      getSystemRam().value() / (input_dim + output_dim);

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