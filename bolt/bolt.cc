#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/ConfigReader.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/Factory.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <toml.hpp>
#include <vector>

namespace bolt = thirdai::bolt;
namespace dataset = thirdai::dataset;

uint32_t getIntValue(toml::table const* table, const std::string& key,
                     bool use_default = false, uint32_t default_value = 0) {
  if (!table->contains(key) && use_default) {
    return default_value;
  }
  if (!table->contains(key)) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' in table."
              << std::endl;
    exit(1);
  } else if (!table->get(key)->is_integer()) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' to have type int."
              << std::endl;
    exit(1);
  }

  return table->get(key)->as_integer()->get();
}

float getFloatValue(toml::table const* table, const std::string& key,
                    bool use_default = false, float default_value = 0) {
  if (!table->contains(key) && use_default) {
    return default_value;
  }
  if (!table->contains(key)) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' in table."
              << std::endl;
    exit(1);
  } else if (!table->get(key)->is_floating_point()) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' to have type float."
              << std::endl;
    exit(1);
  }

  return table->get(key)->as_floating_point()->get();
}

std::string getStrValue(toml::table const* table, const std::string& key,
                        bool use_default = false,
                        std::string default_value = "") {
  if (!table->contains(key) && use_default) {
    return default_value;
  }
  if (!table->contains(key)) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' in table."
              << std::endl;
    exit(1);
  } else if (!table->get(key)->is_string()) {
    std::cerr << "Invalid config file format: expected key '" + key +
                     "' to have type string."
              << std::endl;
    exit(1);
  }

  return table->get(key)->as_string()->get();
}

std::vector<bolt::FullyConnectedLayerConfig> createFullyConnectedLayerConfigs(
    toml::node_view<toml::node> configs) {
  if (!configs.is_array_of_tables()) {
    std::cerr
        << "Invalid config file format: expected array of layer config tables."
        << std::endl;
    exit(1);
  }
  std::vector<bolt::FullyConnectedLayerConfig> layers;

  auto* array = configs.as_array();
  for (auto& config : *array) {
    if (!config.is_table()) {
      std::cerr
          << "Invalid config file format: expected table in layer config array."
          << std::endl;
      exit(1);
    }
    auto* table = config.as_table();

    uint32_t dim = getIntValue(table, "dim");
    uint32_t hashes_per_table = getIntValue(table, "hashes_per_table", true, 0);
    uint32_t num_tables = getIntValue(table, "num_tables", true, 0);
    uint32_t range_pow = getIntValue(table, "range_pow", true, 0);
    uint32_t reservoir_size = getIntValue(table, "reservoir_size", true, 0);
    std::string activation = getStrValue(table, "activation");
    float sparsity = getFloatValue(table, "sparsity", true, 1.0);

    layers.push_back(bolt::FullyConnectedLayerConfig(
        dim, sparsity, thirdai::bolt::getActivationFunction(activation),
        bolt::SamplingConfig(hashes_per_table, num_tables, range_pow,
                             reservoir_size)));
  }
  return layers;
}

bolt::EmbeddingLayerConfig createEmbeddingLayerConfig(toml::table& config) {
  if (!config.contains("embedding_layer") ||
      !config["embedding_layer"].is_table()) {
    std::cerr << "Invalid config file format: expected table for embedding "
                 "layer config."
              << std::endl;
    exit(1);
  }

  auto* embedding_config = config["embedding_layer"].as_table();

  uint32_t num_embedding_lookups =
      getIntValue(embedding_config, "num_embedding_lookups");
  uint32_t lookup_size = getIntValue(embedding_config, "lookup_size");
  uint32_t log_embedding_block_size =
      getIntValue(embedding_config, "log_embedding_block_size");

  return bolt::EmbeddingLayerConfig(num_embedding_lookups, lookup_size,
                                    log_embedding_block_size);
}

std::vector<std::string> getMetrics(toml::table const* config,
                                    const std::string& metric_name) {
  if (!config->contains(metric_name) || !config->get(metric_name)->is_array()) {
    std::cerr << "Invalid config file format: expected array for metrics."
              << std::endl;
    exit(1);
  }
  std::vector<std::string> metrics;

  const auto* array = config->get(metric_name)->as_array();
  for (const auto& m : *array) {
    if (!m.is_string()) {
      std::cerr << "Invalid config file format: expected metrics as strings."
                << std::endl;
      exit(1);
    }
    metrics.push_back(m.as_string()->get());
  }
  return metrics;
}

std::string findFullFilepath(const std::string& filename) {
  std::string full_file_path = __FILE__;
  std::string dataset_path_file =
      full_file_path.substr(0, full_file_path.size() - strlen("bolt/bolt.cc"));
  dataset_path_file += "dataset_paths.toml";

  try {
    toml::table table = toml::parse_file(dataset_path_file);

    if (!table.contains("prefixes") || !table["prefixes"].is_array()) {
      std::cerr << "Invalid config file format: expected array 'prefixes' in "
                   "'dataset_paths.toml'."
                << std::endl;
      exit(1);
    }
    auto* array = table["prefixes"].as_array();
    for (auto& prefix : *array) {
      if (!prefix.is_string()) {
        std::cerr
            << "Invalid config file format: expected prefix to be string in "
               "'dataset_paths.toml'."
            << std::endl;
        exit(1);
      }

      std::string candidate = prefix.as_string()->get() + filename;
      if (std::filesystem::exists(candidate)) {
        return candidate;
      }
    }

    std::cerr << "Could not find file '" + filename +
                     "' on any filepaths. Add correct path to "
                     "'Universe/dataset_paths.toml'"
              << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << "Expected 'dataset_paths.toml' at root of Universe"
              << std::endl;
    exit(1);
  }
}

void trainFCN(toml::table& config) {
  auto layers = createFullyConnectedLayerConfigs(config["layers"]);

  if (!config.contains("dataset") || !config["dataset"].is_table()) {
    std::cerr << "Invalid config file format: expected table for dataset info."
              << std::endl;
    exit(1);
  }
  auto* dataset_table = config["dataset"].as_table();
  uint32_t input_dim = getIntValue(dataset_table, "input_dim");
  std::string train_filename =
      findFullFilepath(getStrValue(dataset_table, "train_data"));
  std::string test_filename =
      findFullFilepath(getStrValue(dataset_table, "test_data"));
  std::string dataset_format = getStrValue(dataset_table, "format");
  uint32_t max_test_batches =
      getIntValue(dataset_table, "max_test_batches", true,
                  std::numeric_limits<uint32_t>::max());

  if (!config.contains("params") || !config["params"].is_table()) {
    std::cerr << "Invalid config file format: expected table for parameters."
              << std::endl;
    exit(1);
  }
  auto* param_table = config["params"].as_table();
  uint32_t batch_size = getIntValue(param_table, "batch_size");
  float learning_rate = getFloatValue(param_table, "learning_rate");
  uint32_t epochs = getIntValue(param_table, "epochs");
  uint32_t rehash = getIntValue(param_table, "rehash");
  uint32_t rebuild = getIntValue(param_table, "rebuild");

  auto train_metrics = getMetrics(param_table, "train_metrics");
  auto test_metrics = getMetrics(param_table, "test_metrics");

  auto loss_fn =
      thirdai::bolt::getLossFunction(getStrValue(param_table, "loss_fn"));

  uint32_t sparse_inference_epoch = 0;
  bool use_sparse_inference = param_table->contains("sparse_inference_epoch");
  if (use_sparse_inference) {
    sparse_inference_epoch = getIntValue(param_table, "sparse_inference_epoch");
  }

  bolt::FullyConnectedNetwork network(layers, input_dim);

  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> train_fac;
  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> test_fac;

  if (dataset_format == "svm") {
    train_fac = std::make_unique<dataset::BoltSvmBatchFactory>();
    test_fac = std::make_unique<dataset::BoltSvmBatchFactory>();
  } else if (dataset_format == "csv") {
    std::string delimiter = getStrValue(dataset_table, "delimter", true, ",");
    if (delimiter.size() != 1) {
      std::cerr << "Invalid config file format: csv delimiter should be single "
                   "character."
                << std::endl;
      exit(1);
    }

    train_fac = std::make_unique<dataset::BoltCsvBatchFactory>(delimiter[0]);
    test_fac = std::make_unique<dataset::BoltCsvBatchFactory>(delimiter[0]);
  } else {
    std::cerr << "Invalid dataset format '" << dataset_format
              << "'. Use 'svm' or 'csv'" << std::endl;
    exit(1);
  }

  dataset::InMemoryDataset<dataset::BoltInputBatch> train_data(
      train_filename, batch_size, std::move(*train_fac));

  dataset::InMemoryDataset<dataset::BoltInputBatch> test_data(
      test_filename, batch_size, std::move(*test_fac));

  for (uint32_t e = 0; e < epochs; e++) {
    network.train(train_data, loss_fn, learning_rate, 1, rehash, rebuild,
                  train_metrics);
    if (use_sparse_inference && e == sparse_inference_epoch) {
      network.enableSparseInference();
    }
    network.predict(test_data, nullptr, test_metrics, max_test_batches);
  }
}

using ClickThroughDataset =
    thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>;

ClickThroughDataset loadClickThorughDataset(const std::string& filename,
                                            uint32_t batch_size,
                                            uint32_t num_dense_features,
                                            uint32_t num_categorical_features,
                                            bool sparse_labels) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(
      num_dense_features, num_categorical_features, sparse_labels);
  ClickThroughDataset data(filename, batch_size, std::move(factory));
  auto end = std::chrono::high_resolution_clock::now();
  std::cout
      << "Read " << data.len() << " vectors in "
      << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
      << " seconds" << std::endl;
  return data;
}

void trainDLRM(toml::table& config) {
  auto embedding_layer = createEmbeddingLayerConfig(config);
  auto bottom_mlp =
      createFullyConnectedLayerConfigs(config["bottom_mlp_layers"]);
  auto top_mlp = createFullyConnectedLayerConfigs(config["top_mlp_layers"]);

  if (!config.contains("dataset") || !config["dataset"].is_table()) {
    std::cerr << "Invalid config file format: expected table for dataset info."
              << std::endl;
    exit(1);
  }
  auto* dataset_table = config["dataset"].as_table();
  uint32_t dense_features = getIntValue(dataset_table, "dense_features");
  uint32_t categorical_features =
      getIntValue(dataset_table, "categorical_features");
  std::string train_filename =
      findFullFilepath(getStrValue(dataset_table, "train_data"));
  std::string test_filename =
      findFullFilepath(getStrValue(dataset_table, "test_data"));

  if (!config.contains("params") || !config["params"].is_table()) {
    std::cerr << "Invalid config file format: expected table for parameters."
              << std::endl;
    exit(1);
  }
  auto* param_table = config["params"].as_table();
  uint32_t batch_size = getIntValue(param_table, "batch_size");
  float learning_rate = getFloatValue(param_table, "learning_rate");
  uint32_t epochs = getIntValue(param_table, "epochs");
  uint32_t rehash = getIntValue(param_table, "rehash");
  uint32_t rebuild = getIntValue(param_table, "rebuild");

  auto train_metrics = getMetrics(param_table, "train_metrics");
  auto test_metrics = getMetrics(param_table, "test_metrics");

  auto loss_fn =
      thirdai::bolt::getLossFunction(getStrValue(param_table, "loss_fn"));

  bolt::DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, dense_features);

  auto train_data =
      loadClickThorughDataset(train_filename, batch_size, dense_features,
                              categorical_features, top_mlp.back().dim > 1);
  auto test_data =
      loadClickThorughDataset(test_filename, batch_size, dense_features,
                              categorical_features, top_mlp.back().dim > 1);

  for (uint32_t e = 0; e < epochs; e++) {
    dlrm.train(train_data, loss_fn, learning_rate, 1, rehash, rebuild,
               train_metrics);
    dlrm.predict(test_data, nullptr, test_metrics);
  }
}

int main(int argc, const char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid args, usage: ./bolt <config file>" << std::endl;
    return 1;
  }

  try {
    toml::table table = toml::parse_file(argv[1]);

    if (table.contains("layers")) {
      trainFCN(table);
    } else if (table.contains("bottom_mlp_layers") &&
               table.contains("top_mlp_layers")) {
      trainDLRM(table);
    }
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
