#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/Continuous.h>
#include <dataset/src/blocks/Date.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Trend.h>
#include <dataset/src/bolt_datasets/DataLoader.h>
#include <dataset/src/bolt_datasets/ShuffleBatchBuffer.h>
#include <dataset/src/bolt_datasets/StreamingGenericDatasetLoader.h>
#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
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
  SequentialClassifierConfig(std::string model_size, std::string task,
                             size_t horizon, size_t n_items, size_t n_users = 0,
                             size_t n_item_categories = 0,
                             size_t n_target_classes = 0)
      : _n_users(n_users),
        _n_items(n_items),
        _n_item_categories(n_item_categories),
        _horizon(horizon),
        _n_target_classes(n_target_classes),
        _task(std::move(task)),
        _model_size(std::move(model_size)) {}

  size_t _n_users;
  size_t _n_items;
  size_t _n_item_categories;
  size_t _horizon;
  size_t _n_target_classes;
  std::string _task;
  std::string _model_size;
};

const size_t BATCH_SIZE = 2048;

class SequentialClassifierTests;

class SequentialClassifier {
  friend SequentialClassifierTests;

 public:
  explicit SequentialClassifier(
      std::unordered_map<std::string, std::string> schema,
      SequentialClassifierConfig config, char delimiter = ',',
      bool use_sequential_feats = true)
      : _schema(std::move(schema)),
        _config(std::move(config)),
        _delimiter(delimiter),
        _use_sequential_feats(use_sequential_feats) {}

  std::shared_ptr<dataset::StreamingGenericDatasetLoader> buildPipeline(
      std::string& filename, bool train, bool overwrite_index) {
    startBuildingPipeline();
    auto loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, BATCH_SIZE);
    auto header = loader->getHeader();
    if (!header) {
      throw std::invalid_argument(
          "[SequentialClassifier::train] The file has no header.");
    }
    std::unordered_map<std::string, size_t> columns =
        parseHeader(*header, _delimiter);
    auto input_blocks = buildInputBlocks(columns, overwrite_index);
    auto label_blocks = buildLabelBlocks(columns);
    auto buffer_size = autotuneShuffleBufferSize();

    return std::make_shared<dataset::StreamingGenericDatasetLoader>(
        loader, input_blocks, label_blocks, /* shuffle = */ train,
        dataset::ShuffleBufferConfig(buffer_size),
        /* has_header = */ false, _delimiter);
  }

  void train(std::string filename, uint32_t epochs, float learning_rate,
             bool overwrite_index = false) {
    auto pipeline =
        buildPipeline(filename, /* train = */ true, overwrite_index);
    if (!_network) {
      _network = buildNetwork(*pipeline);
    }
    MeanSquaredError loss;

    if (!AutoTuneUtils::canLoadDatasetInMemory(filename)) {
      for (uint32_t e = 0; e < epochs; e++) {
        // Train on streaming dataset
        _network->trainOnStream(pipeline, loss, learning_rate);

        /*
          Create new stream for next epoch with new data loader.
          overwrite_index always true in this case because we're
          rereading from the same file.
        */

        pipeline = buildPipeline(filename, /* train = */ true,
                                 /* overwrite_index = */ true);
      }

    } else {
      auto [train_data, train_labels] = pipeline->loadInMemory();

      _network->train(train_data, train_labels, loss, learning_rate, 1);
      _network->freezeHashTables();
      _network->train(train_data, train_labels, loss, learning_rate,
                      epochs - 1);
    }
  }

  float predict(std::string filename) {
    auto pipeline = buildPipeline(filename, /* train = */ false,
                                  /* overwrite_index = */ false);
    if (!_network) {
      throw std::runtime_error(
          "[SequentialClassifier::predict] Predict method called before "
          "training the classifier.");
    }
    std::vector<std::string> metrics{metricName()};
    auto res = _network->predictOnStream(
        pipeline, /* use_sparse_inference = */ true, metrics);
    return res[metricName()];
  }

 private:
  std::string metricName() const {
    return toLower(_config._task) == "regression" ? "root_mean_squared_error"
                                                  : "categorical_accuracy";
  }

  FullyConnectedNetwork buildNetwork(
      dataset::StreamingGenericDatasetLoader& pipeline) const {
    SequentialConfigList configs;

    auto hidden_dim = AutoTuneUtils::getHiddenLayerSize(
        _config._model_size, pipeline.getLabelDim(), pipeline.getInputDim());
    auto hidden_sparsity = AutoTuneUtils::getHiddenLayerSparsity(hidden_dim);

    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        hidden_dim, /* _sparsity = */ hidden_sparsity,
        /* _act_func = */ ActivationFunction::ReLU));
    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        /* _dim = */ pipeline.getLabelDim(),
        /* _act_func = */ toLower(_config._task) == "regression"
            ? ActivationFunction::Linear
            : ActivationFunction::Softmax));
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
      return {std::make_shared<dataset::ContinuousBlock>(
          columns.at(_schema.at("target")))};
    }
    if (task_lower == "classification") {
      if (_config._n_target_classes == 0) {
        throw std::invalid_argument(
            "[SequentialClassifier] Task is classification but "
            "n_target_classes is set to 0 in config.");
      }
      return {std::make_shared<dataset::CategoricalBlock>(
          columns.at(_schema.at("target")), std::make_shared<dataset::StringToUidMap>(_config._n_target_classes))};
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

  Blocks buildInputBlocks(const Columns& columns, bool overwrite_index) {
    checkValidSchema();
    checkColumns(columns);
    std::vector<std::shared_ptr<dataset::Block>> blocks;
    addDateBlock(columns, blocks);
    addUserIdBlock(columns, blocks);
    addItemIdBlock(columns, blocks);
    addItemTextBlock(columns, blocks);
    addItemCategoricalBlock(columns, blocks);
    if (_use_sequential_feats) {
      addUserTimeSeriesBlock(columns, blocks, overwrite_index);
      addItemTimeSeriesBlock(columns, blocks, overwrite_index);
    }
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

  void checkColumns(const Columns& columns) {
    for (const auto& [key, name] : _schema) {
      if (columns.count(name) == 0) {
        std::stringstream error_ss;
        error_ss << "[SequentialClassifier] Column name for key '" << key
                 << "' is set to '" << name
                 << "' but we could not find this column in the header.";
        throw std::invalid_argument(error_ss.str());
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
        columns.at(_schema.at("user")), std::make_shared<dataset::StringToUidMap>(_config._n_users)));
    addNonzeros(1);
  }

  void addItemIdBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'item' in schema.");
    }
    blocks.push_back(std::make_shared<dataset::CategoricalBlock>(
        columns.at(_schema.at("item")), std::make_shared<dataset::StringToUidMap>(_config._n_items)));
    addNonzeros(1);
  }

  void addUserTimeSeriesBlock(const Columns& columns,

                              Blocks& blocks, bool overwrite_index) {
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
      count_col = columns.at(_schema.at("quantities"));
    }

    if (_user_trend_block == nullptr || overwrite_index) {
      _user_trend_block = std::make_shared<dataset::TrendBlock>(
          has_count_col, columns.at(_schema.at("user")),
          columns.at(_schema.at("timestamp")), count_col, _config._horizon,
          /* lookback = */ std::max(_config._horizon, static_cast<size_t>(30)));
    }
    if (_user_trend_block == nullptr) {
      throw std::runtime_error("USER TREND BLOCK IS NULL");
    }
    blocks.push_back(_user_trend_block);
    addNonzeros(_user_trend_block->featureDim());
  }

  void addItemTimeSeriesBlock(const Columns& columns, Blocks& blocks,
                              bool overwrite_index) {
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
      count_col = columns.at(_schema.at("quantities"));
    }

    if (_item_trend_block == nullptr || overwrite_index) {
      _item_trend_block = std::make_shared<dataset::TrendBlock>(
          has_count_col, columns.at(_schema.at("item")),
          columns.at(_schema.at("timestamp")), count_col, _config._horizon,
          /* lookback = */ std::max(_config._horizon, static_cast<size_t>(30)));
    }
    if (_item_trend_block == nullptr) {
      throw std::runtime_error("ITEM TREND BLOCK IS NULL");
    }
    blocks.push_back(_item_trend_block);
    addNonzeros(_item_trend_block->featureDim());
  }

  void addDateBlock(const Columns& columns,

                    Blocks& blocks) {
    if (_schema.count("timestamp") == 0) {
      throw std::invalid_argument(
          "Could not find required key 'timestamp' in schema.");
    }
    blocks.push_back(std::make_shared<dataset::DateBlock>(
        /* col = */ columns.at(_schema.at("timestamp"))));
    addNonzeros(4);
  }

  void addItemTextBlock(const Columns& columns, Blocks& blocks) {
    if (_schema.count("item_text") == 0) {
      return;
    }
    blocks.push_back(std::make_shared<dataset::TextBlock>(
        columns.at(_schema.at("item_text")), /* dim = */ 100000));
    addNonzeros(100);
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
        columns.at(_schema.at("item_categorical")),
        std::make_shared<dataset::StringToUidMap>(_config._n_item_categories)));
    addNonzeros(100);
  }

  void startBuildingPipeline() { _est_nonzeros = 0; }

  void addNonzeros(size_t nonzeros) { _est_nonzeros += nonzeros; }

  size_t getNonzerosPerVector() const { return _est_nonzeros; }

  size_t autotuneShuffleBufferSize() const {
    auto batch_mem = BATCH_SIZE * _est_nonzeros *
                     8;  // 4 bytes for index, 4 bytes for value.
    if (auto ram = AutoTuneUtils::getSystemRam()) {
      auto mem_allowance = *ram / 2;
      return mem_allowance / batch_mem;
    }
    return 1000;
  }

  Schema _schema;
  SequentialClassifierConfig _config;
  std::shared_ptr<dataset::TrendBlock> _user_trend_block;
  std::shared_ptr<dataset::TrendBlock> _item_trend_block;
  char _delimiter;
  std::optional<FullyConnectedNetwork> _network;
  bool _use_sequential_feats;
  size_t _est_nonzeros;
};

}  // namespace thirdai::bolt
