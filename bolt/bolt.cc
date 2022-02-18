#include <bolt/src/networks/DLRM.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/ConfigReader.h>
#include <dataset/src/Dataset.h>
#include <chrono>
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
        dim, sparsity, activation,
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

void trainFCN(toml::table& config) {
  auto layers = createFullyConnectedLayerConfigs(config["layers"]);

  if (!config.contains("dataset") || !config["dataset"].is_table()) {
    std::cerr << "Invalid config file format: expected table for dataset info."
              << std::endl;
    exit(1);
  }
  auto* dataset_table = config["dataset"].as_table();
  uint32_t input_dim = getIntValue(dataset_table, "input_dim");
  std::string train_filename = getStrValue(dataset_table, "train_data");
  std::string test_filename = getStrValue(dataset_table, "test_data");
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

  bolt::FullyConnectedNetwork network(layers, input_dim);

  if (dataset_format == "svm") {
    dataset::InMemoryDataset<dataset::SparseBatch> train_data(
        train_filename, batch_size, dataset::SvmSparseBatchFactory{});

    dataset::InMemoryDataset<dataset::SparseBatch> test_data(
        test_filename, batch_size, dataset::SvmSparseBatchFactory{});

    for (uint32_t e = 0; e < epochs; e++) {
      network.train(train_data, learning_rate, 1, rehash, rebuild);
      network.predict(test_data, max_test_batches);
    }
  } else if (dataset_format == "csv") {
    std::string delimiter = getStrValue(dataset_table, "delimter", true, ",");
    if (delimiter.size() != 1) {
      std::cerr << "Invalid config file format: csv delimiter should be single "
                   "character."
                << std::endl;
      exit(1);
    }

    dataset::InMemoryDataset<dataset::DenseBatch> train_data(
        train_filename, batch_size,
        dataset::CsvDenseBatchFactory{delimiter[0]});

    dataset::InMemoryDataset<dataset::DenseBatch> test_data(
        test_filename, batch_size, dataset::CsvDenseBatchFactory{delimiter[0]});

    for (uint32_t e = 0; e < epochs; e++) {
      network.train(train_data, learning_rate, 1, rehash, rebuild);
      network.predict(test_data, max_test_batches);
    }
  }
}

using ClickThroughDataset =
    thirdai::dataset::InMemoryDataset<thirdai::dataset::ClickThroughBatch>;

ClickThroughDataset loadClickThorughDataset(const std::string& filename,
                                            uint32_t batch_size,
                                            uint32_t num_dense_features,
                                            uint32_t num_categorical_features) {
  auto start = std::chrono::high_resolution_clock::now();
  thirdai::dataset::ClickThroughBatchFactory factory(num_dense_features,
                                                     num_categorical_features);
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
  auto bottom_mlp = createFullyConnectedLayerConfigs(config["bottom_mlp"]);
  auto top_mlp = createFullyConnectedLayerConfigs(config["top_mlp"]);
  uint32_t output_dim = top_mlp.back().dim;

  if (!config.contains("dataset") || !config["dataset"].is_table()) {
    std::cerr << "Invalid config file format: expected table for dataset info."
              << std::endl;
    exit(1);
  }
  auto* dataset_table = config["dataset"].as_table();
  uint32_t dense_features = getIntValue(dataset_table, "dense_features");
  uint32_t categorical_features =
      getIntValue(dataset_table, "categorical_features");
  std::string train_filename = getStrValue(dataset_table, "train_data");
  std::string test_filename = getStrValue(dataset_table, "test_data");

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

  bolt::DLRM dlrm(embedding_layer, bottom_mlp, top_mlp, dense_features);

  auto train_data = loadClickThorughDataset(
      train_filename, batch_size, dense_features, categorical_features);
  auto test_data = loadClickThorughDataset(
      test_filename, batch_size, dense_features, categorical_features);

  std::vector<float> scores(test_data.len() * output_dim);
  for (uint32_t e = 0; e < epochs; e++) {
    dlrm.train(train_data, learning_rate, 1, rehash, rebuild);
    dlrm.predict(test_data, scores.data());
  }
}

int main(int argc, const char** argv) {
  if (argc != 2) {
    std::cerr << "Invalid args, usage: ./bolt <config file>" << std::endl;
    return 1;
  }

  toml::table table = toml::parse_file(argv[1]);

  if (table.contains("layers")) {
    trainFCN(table);
  } else if (table.contains("bottom_mlp_layers") &&
             table.contains("top_mlp_layers")) {
    trainDLRM(table);
  }

  return 0;
}

// int main(int argc, char** argv) {
//   if (argc != 2) {
//     std::cerr << "Invalid args, usage: ./bolt <config file>" << std::endl;
//     return 1;
//   }

//   bolt::ConfigReader config(argv[1]);

//   uint32_t num_layers = config.intVal("num_layers");
//   std::vector<bolt::FullyConnectedLayerConfig> layers;

//   for (uint32_t l = 0; l < num_layers; l++) {
//     bolt::ActivationFunc func = bolt::ActivationFunc::ReLU;
//     if (l == num_layers - 1) {
//       func = bolt::ActivationFunc::Softmax;
//     }
//     layers.emplace_back(
//         config.intVal("dims", l), config.floatVal("sparsity", l), func,
//         bolt::SamplingConfig(config.intVal("hashes_per_table", l),
//                              config.intVal("num_tables", l),
//                              config.intVal("range_pow", l),
//                              config.intVal("reservoir_size", l)));
//   }

//   bolt::FullyConnectedNetwork network(layers, config.intVal("input_dim"));

//   float learning_rate = config.floatVal("learning_rate");
//   uint32_t epochs = config.intVal("epochs");
//   uint32_t batch_size = config.intVal("batch_size");
//   uint32_t rehash = config.intVal("rehash");
//   uint32_t rebuild = config.intVal("rebuild");
//   uint32_t switch_inference_epoch = config.valExists("sparse_inf_at_epoch")
//                                         ?
//                                         config.intVal("sparse_inf_at_epoch")
//                                         : 0;

//   uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();
//   if (config.valExists("max_test_batches")) {
//     max_test_batches = config.intVal("max_test_batches");
//   }

//   if (!config.valExists("dataset_format")) {
//     std::cerr << "dataset_format is not specified in the config file."
//               << std::endl;
//     return 1;
//   }
//   if (config.strVal("dataset_format") == "svm") {
//     dataset::InMemoryDataset<dataset::SparseBatch> train_data(
//         config.strVal("train_data"), batch_size,
//         dataset::SvmSparseBatchFactory{});

//     dataset::InMemoryDataset<dataset::SparseBatch> test_data(
//         config.strVal("test_data"), batch_size,
//         dataset::SvmSparseBatchFactory{});

//     for (uint32_t e = 0; e < epochs; e++) {
//       if ((switch_inference_epoch > 0) && e == switch_inference_epoch) {
//         std::cout << "Freezing Selection for Sparse Inference" <<
//         std::endl; network.useSparseInference();
//       }

//       network.train<dataset::SparseBatch>(train_data, learning_rate, 1,
//       rehash,
//                                           rebuild);
//       network.predict<dataset::SparseBatch>(test_data, max_test_batches);
//     }
//     network.predict(test_data);
//   } else if (config.strVal("dataset_format") == "csv") {
//     if (!config.valExists("csv_delimiter")) {
//       std::cerr << "csv_delimiter is not specified in the config file."
//                 << std::endl;
//       return 1;
//     }
//     dataset::InMemoryDataset<dataset::DenseBatch> train_data(
//         config.strVal("train_data"), batch_size,
//         dataset::CsvDenseBatchFactory(config.strVal("csv_delimiter").at(0)));

//     dataset::InMemoryDataset<dataset::DenseBatch> test_data(
//         config.strVal("test_data"), batch_size,
//         dataset::CsvDenseBatchFactory(config.strVal("csv_delimiter").at(0)));

//     for (uint32_t e = 0; e < epochs; e++) {
//       if ((switch_inference_epoch > 0) && e == switch_inference_epoch) {
//         std::cout << "Freezing Selection for Sparse Inference" <<
//         std::endl; network.useSparseInference();
//       }

//       network.train<dataset::DenseBatch>(train_data, learning_rate, 1,
//       rehash,
//                                          rebuild);
//       network.predict<dataset::DenseBatch>(test_data, max_test_batches);
//     }
//     network.predict(test_data);
//   } else {
//     std::cerr << "The dataset format " << config.strVal("dataset_format")
//               << " is not supported." << std::endl;
//     return 1;
//   }

//   return 0;
// }
