#include "Models.h"
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/Op.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/udt/Defaults.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::udt::utils {

ModelPtr buildModel(uint32_t input_dim, uint32_t output_dim,
                    const config::ArgumentMap& args,
                    const std::optional<std::string>& model_config,
                    bool use_sigmoid_bce, bool mach) {
  if (model_config) {
    return utils::loadModel({input_dim}, output_dim, *model_config, mach);
  }
  uint32_t hidden_dim = args.get<uint32_t>("embedding_dimension", "integer",
                                           defaults::HIDDEN_DIM);
  bool use_tanh = args.get<bool>("use_tanh", "bool", defaults::USE_TANH);

  if (args.contains("use_bias")) {
    throw std::invalid_argument(
        "Option 'use_bias' has been depreciated. Please use 'hidden_bias' or "
        "'output_bias'.");
  }
  bool hidden_bias =
      args.get<bool>("hidden_bias", "bool", defaults::HIDDEN_BIAS);
  bool output_bias =
      args.get<bool>("output_bias", "bool", defaults::OUTPUT_BIAS);

  bool normalize_embeddings = args.get<bool>("normalize_embeddings", "bool",
                                             defaults::NORMALIZE_EMBEDDINGS);
  return utils::defaultModel(input_dim, hidden_dim, output_dim, use_sigmoid_bce,
                             use_tanh, /* hidden_bias= */ hidden_bias,
                             /* output_bias= */ output_bias, /* mach= */ mach,
                             /* normalize_embeddings= */ normalize_embeddings);
}

float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},    {900, 0.2},    {1800, 0.1},     {4000, 0.05},
      {10000, 0.02}, {20000, 0.01}, {1000000, 0.005}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return sparsity_values.back().second;
}

ModelPtr defaultModel(uint32_t input_dim, uint32_t hidden_dim,
                      uint32_t output_dim, bool use_sigmoid_bce, bool use_tanh,
                      bool hidden_bias, bool output_bias, bool mach,
                      bool normalize_embeddings) {
  auto input = bolt::Input::make(input_dim);

  const auto* hidden_activation = use_tanh ? "tanh" : "relu";

  auto hidden = bolt::Embedding::make(hidden_dim, input_dim, hidden_activation,
                                      /* bias= */ hidden_bias)
                    ->apply(input);

  if (normalize_embeddings) {
    hidden = bolt::LayerNorm::make()->apply(hidden);
  }

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = use_sigmoid_bce ? "sigmoid" : "softmax";
  auto output = bolt::FullyConnected::make(
                    output_dim, hidden->dim(), sparsity, activation,
                    /* sampling= */ nullptr, /* use_bias= */ output_bias)
                    ->apply(hidden);

  auto labels = bolt::Input::make(output_dim);

  bolt::LossPtr loss;
  if (use_sigmoid_bce) {
    loss = bolt::BinaryCrossEntropy::make(output, labels);
  } else {
    loss = bolt::CategoricalCrossEntropy::make(output, labels);
  }

  bolt::ComputationList additional_labels;
  if (mach) {
    // For mach we need the hash based labels for training, but the actual
    // document/class ids to compute metrics. Hence we add two labels to the
    // model.
    additional_labels.push_back(
        bolt::Input::make(std::numeric_limits<uint32_t>::max()));
  }

  auto model = bolt::Model::make({input}, {output}, {loss}, additional_labels);

  return model;
}

ModelPtr loadModel(const std::vector<uint32_t>& input_dims,
                   uint32_t specified_output_dim,
                   const std::string& config_path, bool mach) {
  config::ArgumentMap parameters;
  parameters.insert("output_dim", specified_output_dim);

  auto json_config = json::parse(config::loadConfig(config_path));

  auto model = config::buildModel(json_config, parameters, input_dims, mach);

  uint32_t actual_output_dim = model->outputs().at(0)->dim();
  if (actual_output_dim != specified_output_dim) {
    throw std::invalid_argument(
        "Expected model with output dim " +
        std::to_string(specified_output_dim) +
        ", but the model config yielded a model with output dim " +
        std::to_string(actual_output_dim) + ".");
  }

  return model;
}

void verifyCanSetModel(const ModelPtr& curr_model, const ModelPtr& new_model) {
  auto vec_eq = [](const auto& a, const auto& b) -> bool {
    if (a.size() != b.size()) {
      return false;
    }
    for (uint32_t i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) {
        return false;
      }
    }
    return true;
  };

  if (!vec_eq(curr_model->inputDims(), new_model->inputDims())) {
    throw std::invalid_argument("Input dim mismatch in set_model.");
  }

  if (new_model->outputs().size() != 1 ||
      new_model->outputs().at(0)->dim() != curr_model->outputs().at(0)->dim()) {
    throw std::invalid_argument("Output dim mismatch in set_model.");
  }

  if (!vec_eq(curr_model->labelDims(), new_model->labelDims())) {
    throw std::invalid_argument("Label dim mismatch in set_model.");
  }
}

bool hasSoftmaxOutput(const ModelPtr& model) {
  auto outputs = model->outputs();
  if (outputs.size() > 1) {
    return false;  // TODO(Nicholas): Should this throw?
  }

  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
  return fc && (fc->kernel()->getActivationFunction() ==
                bolt::ActivationFunction::Softmax);
}

// Function to create directories recursively
bool createDirectories(const std::string& path) {
  try {
    std::filesystem::create_directories(path);
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to create directory: " << path << "\n";
    std::cerr << "Error: " << e.what() << "\n";
    return false;
  }
}

// Progress callback function to display download progress
static int progressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow,
                            curl_off_t /* ultotal */, curl_off_t /* ulnow */) {
  ProgressData* progressData = (ProgressData*)clientp;

  // Calculate progress percentage
  double progressPercent =
      (dltotal > 0)
          ? (static_cast<double>(dlnow) / static_cast<double>(dltotal)) * 100.0
          : 0.0;

  // Display progress information
  std::cout << "\rDownloading... " << dlnow << " / " << dltotal << " bytes "
            << "(" << progressPercent << "%) ";

  progressData->lastDownloadedBytes = dlnow;

  return 0;
}

// Function to download a file using libcurl
bool downloadFile(const std::string& url, const std::string& filePath) {
  FILE* file = fopen(filePath.c_str(), "wb");
  if (!file) {
    std::cerr << "Failed to open file for writing: " << filePath << std::endl;
    return false;
  }

  CURL* curl = curl_easy_init();
  if (curl) {
    // Set the URL to download
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

    // Set the file to write to
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, file);

    // Set progress callback function
    ProgressData progressData = {0};
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progressData);

    // Disable printing download progress to stdout
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(file);

    if (res != CURLE_OK) {
      std::cerr << "\nFailed to download file: " << url << " ("
                << curl_easy_strerror(res) << ")" << std::endl;
      std::remove(filePath.c_str());  // Remove incomplete file
      return false;
    }

    std::cout << "\nDownload complete: " << filePath << std::endl;
  } else {
    std::cerr << "Failed to initialize libcurl" << std::endl;
    fclose(file);
    return false;
  }

  return true;
}

data::SpladeConfig downloadSemanticEnhancementModel(
    const std::string& cacheDir, const std::string& modelName) {
  // Get the current working directory
  std::filesystem::path currentDir = std::filesystem::current_path();

  // Ensure cache directory exists (relative to the current directory)
  std::filesystem::path fullCacheDir = currentDir / cacheDir;
  if (!createDirectories(fullCacheDir.string())) {
    throw std::runtime_error("Failed to create cache directory: " +
                             fullCacheDir.string());
  }

  std::string semanticModelPath = (fullCacheDir / modelName).string();

  // Download the model only if it does not already exist
  if (access(semanticModelPath.c_str(), F_OK) == -1) {
    std::string url =
        "https://modelzoo-cdn.azureedge.net/test-models/" + modelName;
    if (!downloadFile(url, semanticModelPath)) {
      throw std::runtime_error("Failed to download semantic enhancement model");
    }
  }

  // Download BERT vocabulary if it does not exist
  std::string vocabPath = (fullCacheDir / "bert-base-uncased.vocab").string();
  if (access(vocabPath.c_str(), F_OK) == -1) {
    std::string vocabUrl =
        "https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt";
    if (!downloadFile(vocabUrl, vocabPath)) {
      throw std::runtime_error("Failed to download BERT vocabulary");
    }
  }

  return data::SpladeConfig(semanticModelPath, vocabPath, 100, std::nullopt);
}

}  // namespace thirdai::automl::udt::utils