#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Continuous.h>
#include <dataset/src/blocks/CountHistory.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/ShuffleBatchBuffer.h>
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

using Columns = std::unordered_map<std::string, size_t>;
using Schema = std::unordered_map<std::string, std::string>;
using Blocks = std::vector<std::shared_ptr<dataset::Block>>;

struct SequentialClassifierConfig {
  SequentialClassifierConfig(std::string task, size_t horizon, size_t n_items,
                             size_t n_users = 0, size_t n_item_categories = 0,
                             size_t n_target_classes = 0)
      : _n_users(n_users),
        _n_items(n_items),
        _n_item_categories(n_item_categories),
        _horizon(horizon),
        _n_target_classes(n_target_classes),
        _task(std::move(task)) {}

  size_t _n_users;
  size_t _n_items;
  size_t _n_item_categories;
  size_t _horizon;
  size_t _n_target_classes;
  std::string _task;
};

const size_t BATCH_SIZE = 2048;

class SequentialClassifier {
 public:
  explicit SequentialClassifier(
      std::unordered_map<std::string, std::string> schema,
      SequentialClassifierConfig config, char delimiter = ',')
      : _schema(std::move(schema)),
        _config(std::move(config)),
        _delimiter(delimiter) {}

  void train(std::string filename) {
    auto loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);
    auto header = loader->getHeader();
    if (!header) {
      throw std::invalid_argument(
          "[SequentialClassifier::train] The file has no header.");
    }
    std::unordered_map<std::string, size_t> columns =
        parseHeader(*header, _delimiter);
    auto input_blocks = buildInputBlocks(columns);
    auto label_blocks = buildLabelBlocks(columns);
    dataset::StreamingGenericDatasetLoader pipeline(
        loader, input_blocks, label_blocks, /* shuffle = */ true,
        dataset::ShuffleBufferConfig(),
        /* has_header = */ false, _delimiter);
    if (!_network) {
      _network = buildNetwork(pipeline);
    }
    auto data = pipeline.loadInMemory();
    MeanSquaredError loss;
    _network->train(data.first, data.second, loss, /* learning_rate = */ 0.0001,
                    /* epochs = */ 5, /* rehash = */ 6400,
                    /* rebuild = */ 128000);
  }

  void predict(std::string filename) {
    auto loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);
    auto header = loader->getHeader();
    if (!header) {
      throw std::invalid_argument(
          "[SequentialClassifier::predict] The file has no header.");
    }
    std::unordered_map<std::string, size_t> columns =
        parseHeader(*header, _delimiter);
    auto input_blocks = buildInputBlocks(columns);
    auto label_blocks = buildLabelBlocks(columns);
    dataset::StreamingGenericDatasetLoader pipeline(
        loader, input_blocks, label_blocks, /* shuffle = */ true,
        dataset::ShuffleBufferConfig(),
        /* has_header = */ false, _delimiter);
    if (!_network) {
      throw std::runtime_error(
          "[SequentialClassifier::predict] Predict method called before "
          "training the classifier.");
    }
    auto [data, label] = pipeline.loadInMemory();
    std::vector<std::string> metrics{"root_mean_squared_error"};
    _network->predict(data, label, /* output_active_neurons= */ nullptr,
                      /* output_activations = */ nullptr, 
                      /* use_sparse_inference = */ true,
                      metrics);
  }

 private:
  FullyConnectedNetwork buildNetwork(
      dataset::StreamingGenericDatasetLoader& pipeline) const {
    SequentialConfigList configs;
    // _dim, float _sparsity,
    // const std::string& _act_func,
    // SamplingConfig _config

    size_t output_dim =
        toLower(_config._task) == "regression" ? 1 : _config._n_target_classes;
    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        /* _dim = */ 5000, /* _sparsity = */ 0.02,
        /* _act_func = */ ActivationFunction::ReLU));
    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        /* _dim = */ output_dim, /* _sparsity = */ 0.02,
        /* _act_func = */ ActivationFunction::ReLU));
    FullyConnectedNetwork network(configs,
                                  /* input_dim = */ pipeline.getInputDim());
    return network;
  }

  static std::unordered_map<std::string, size_t> parseHeader(
      std::string& header, char delimiter) {
    std::unordered_map<std::string, size_t> columns;
    size_t col = 0;
    size_t start = 0;
    size_t end = 0;
    while (end != std::string::npos) {
      end = header.find(delimiter, start);
      size_t len =
          end == std::string::npos ? header.size() - start : end - start;
      columns[header.substr(start, len)] = col;
      col++;
      start = end + 1;
    }
    return columns;
  }

  Blocks buildLabelBlocks(Columns& columns) {
    auto task_lower = toLower(_config._task);
    if (task_lower == "regression") {
      return {std::make_shared<dataset::ContinuousBlock>(columns.at("target"))};
    }
    if (task_lower == "classification") {
      if (_config._n_target_classes == 0) {
        throw std::invalid_argument(
            "[SequentialClassifier] Task is classification but "
            "n_target_classes is set to 0 in config.");
      }
      return {std::make_shared<dataset::CategoricalBlock>(
          columns.at("target"), _config._n_target_classes)};
    }
    std::stringstream error_ss;
    error_ss
        << "[SequentialClassifier::buildLabelBlocks] Got invalid task name '"
        << _config._task << "'. Choose either 'regression' or 'classification'";
    throw std::invalid_argument(error_ss.str());
  }

  static std::string toLower(const std::string& str) {
    std::string lower;
    for (char c : str) {
      lower.push_back(std::tolower(c));
    }
    return lower;
  }

  Blocks buildInputBlocks(const Columns& columns) {
    checkValidSchema();
    std::vector<std::shared_ptr<dataset::Block>> blocks;
    addDateBlock(columns, blocks);
    addUserIdBlock(columns, blocks);
    addUserTimeSeriesBlock(columns, blocks);
    addItemIdBlock(columns, blocks);
    addItemTimeSeriesBlock(columns, blocks);
    addItemTextBlock(columns, blocks);
    addItemCategoricalBlock(columns, blocks);
    return blocks;
  }

  void checkValidSchema() {
    std::vector<std::string> valid_keys{"user", "item", "timestamp",
                                        "item_text", "item_categorical",
                                        // "user_text",
                                        // "user_categorical",
                                        "quantities", "target"};
    for (const auto& [key, _] : _schema) {
      if (std::count(valid_keys.begin(), valid_keys.end(), key) == 0) {
        std::stringstream ss;
        std::string delimiter;
        ss << "Found invalid key '" << key << "' in schema. Valid keys: ";
        for (const auto& valid_key : valid_keys) {
          ss << delimiter << "'" << valid_key << "'";
          delimiter = ", ";
        }
        ss << ".";
        throw std::invalid_argument(ss.str());
      }
    }
  }

  void addUserIdBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("user") == 0) {
      return;
    }
    if (_config._n_users == 0) {
      throw std::invalid_argument(
          "[SequentialClassifier] Found key 'user' in provided schema but "
          "n_users is set to 0 in config.");
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        columns.at("user"), _config._n_users));
  }

  void addItemIdBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'item' in schema.");
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        columns.at("item"), _config._n_items));
  }

  void addUserTimeSeriesBlock(const Columns& columns,

                              Blocks& blocks) {
    if (_schema.count("user") == 0) {
      return;
    }
    if (_schema.count("timestamp") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'timestamp' in schema.");
    }
    bool has_count_col = true;
    size_t count_col = 0;
    if (_schema.count("quantities") == 0) {
      has_count_col = false;
    } else {
      count_col = columns.at("quantities");
    }

    dataset::DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5,
                                        /* range_pow = */ 22);
    blocks.push_back(std::make_shared<dataset::TrendBlock>(
        has_count_col, columns.at("user"), columns.at("timestamp"), count_col,
        _config._horizon,
        /* lookback = */ std::max(_config._horizon, static_cast<size_t>(30)),
        config));
  }

  void addItemTimeSeriesBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'item' in schema.");
    }
    if (_schema.count("timestamp") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'timestamp' in schema.");
    }
    bool has_count_col = true;
    size_t count_col = 0;
    if (_schema.count("quantities") == 0) {
      has_count_col = false;
    } else {
      count_col = columns.at("quantities");
    }

    dataset::DynamicCountsConfig config(/* max_range = */ 1, /* n_rows = */ 5,
                                        /* range_pow = */ 22);

    if (_trend_block == nullptr) {
      _trend_block = std::make_shared<dataset::TrendBlock>(
          has_count_col, columns.at("item"), columns.at("timestamp"), count_col,
          _config._horizon,
          /* lookback = */ std::max(_config._horizon, static_cast<size_t>(30)),
          config);
    }
    blocks.push_back(_trend_block);
  }

  void addDateBlock(const Columns& columns,

                    Blocks& blocks) {
    if (_schema.count("timestamp") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'timestamp' in schema.");
    }
    blocks.push_back(std::make_shared<dataset::DateBlock>(
        /* col = */ columns.at("timestamp")));
  }

  void addItemTextBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item_text") == 0) {
      return;
    }
    blocks.push_back(std::make_shared<dataset::TextBlock>(
        columns.at("item_text"), /* dim = */ 100000));
  }

  void addItemCategoricalBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item_categorical") == 0) {
      return;
    }
    if (_config._n_item_categories == 0) {
      throw std::invalid_argument(
          "[SequentialClassifier] Found key 'item_categorical' in provided "
          "schema but n_item_categories is set to 0 in config.");
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        columns.at("item_categorical"), _config._n_item_categories));
  }

  Schema _schema;
  SequentialClassifierConfig _config;
  std::shared_ptr<dataset::TrendBlock> _trend_block;
  char _delimiter;
  std::optional<FullyConnectedNetwork> _network;
};

}  // namespace thirdai::bolt