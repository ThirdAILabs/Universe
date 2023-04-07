#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/DataSource.h>
#include <pybind11/pybind11.h>
#include <unordered_map>

namespace thirdai::automl::udt {

namespace py = pybind11;

class StringEncoder {
 public:
  explicit StringEncoder(const float* non_owning_pretrained_fc_weights,
                         const float* non_owning_pretrained_fc_biases,
                         uint32_t fc_dim,
                         const data::TextDataTypePtr& data_type,
                         const data::TabularOptions& options);

  py::object supervisedTrain(const dataset::DataSourcePtr& data_source,
                             const std::string& input_col_1,
                             const std::string& input_col_2,
                             const std::string& label_col, float learning_rate,
                             uint32_t epochs,
                             const std::vector<std::string>& metrics);

  bolt::nn::tensor::TensorPtr encode(const std::string& string);

  bolt::nn::tensor::TensorList encodeBatch(
      const std::vector<std::string>& strings);

 private:
  static bolt::nn::model::ModelPtr createEmbeddingModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim);

  static bolt::nn::model::ModelPtr createTwoTowerModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim);

  data::TabularDatasetFactoryPtr _embedding_factory;
  bolt::nn::model::ModelPtr _embedding_model, _two_tower_model;
  data::TextDataTypePtr _data_type;
  data::TabularOptions _options;
};

using StringEncoderPtr = std::shared_ptr<StringEncoder>;

}  // namespace thirdai::automl::udt