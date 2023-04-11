#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <pybind11/pybind11.h>
#include <unordered_map>

namespace thirdai::automl::udt {

namespace py = pybind11;

class TextEmbeddingModel;
using TextEmbeddingModelPtr = std::shared_ptr<TextEmbeddingModel>;

/**
 * This class represents an embedding model that turns text in to a vector
 * representation. It can be pretrained (usually in an unsupervised fashion)
 * and then finetuned using supervised positive and negative pairs.
 */
class TextEmbeddingModel {
 public:
  /*
   * The embedding_op passed in can have pretrained weights that
   * come from a different model. We require that the input dimension of the
   * embedding_op equal options.feature_hash_range, since the embedding model
   * is a tabular featurizer followed by the fully connected embedding_op.
   * The data_type should specify the featurization of the input text.
   */
  explicit TextEmbeddingModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op,
      const data::TextDataTypePtr& text_data_type, data::TabularOptions options,
      float distance_cutoff);

  static TextEmbeddingModelPtr make(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op,
      const data::TextDataTypePtr& text_data_type, data::TabularOptions options,
      float distance_cutoff);

  py::object supervisedTrain(const dataset::DataSourcePtr& data_source,
                             const std::string& input_col_1,
                             const std::string& input_col_2,
                             const std::string& label_col, float learning_rate,
                             uint32_t epochs);

  bolt::nn::tensor::TensorPtr encode(const std::string& string);

  bolt::nn::tensor::TensorList encodeBatch(
      const std::vector<std::string>& strings);

  void save(const std::string& filename) const {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    save_stream(filestream);
  }

  void save_stream(std::ostream& output_stream) const {
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  static TextEmbeddingModelPtr load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    return load_stream(filestream);
  }

  static TextEmbeddingModelPtr load_stream(std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    TextEmbeddingModelPtr deserialize_into(new TextEmbeddingModel());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  TextEmbeddingModel() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(_embedding_factory, _embedding_model, _two_tower_model,
            _text_data_type, _options);
  }

  static bolt::nn::model::ModelPtr createEmbeddingModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim);

  // Also returns a loss metric for training the two tower model
  static std::pair<bolt::nn::model::ModelPtr, bolt::train::metrics::MetricPtr>
  createTwoTowerModel(const bolt::nn::ops::FullyConnectedPtr& embedding_op,
                      uint32_t input_dim, float distance_cutoffs);

  data::TabularDatasetFactoryPtr _embedding_factory;
  bolt::nn::model::ModelPtr _embedding_model, _two_tower_model;
  bolt::train::metrics::MetricPtr _two_tower_metric_ptr;
  data::TextDataTypePtr _text_data_type;
  data::TabularOptions _options;
};

}  // namespace thirdai::automl::udt