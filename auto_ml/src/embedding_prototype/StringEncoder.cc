#include "StringEncoder.h"
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/Validation.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>

namespace thirdai::automl::udt {

StringEncoder::StringEncoder(const std::string& activation_func,
                             const float* non_owning_pretrained_fc_weights,
                             const float* non_owning_pretrained_fc_biases,
                             uint32_t fc_dim,
                             const data::TextDataTypePtr& data_type,
                             const data::TabularOptions& options)
    : _data_type(data_type), _options(options) {
  uint32_t input_dim = options.feature_hash_range;

  auto fc_op = bolt::nn::ops::FullyConnected::make(
      /* dim= */ fc_dim,
      /* input_dim= */ input_dim, /* sparsity= */ 1.0,
      /* activation= */ activation_func,
      /* sampling=*/nullptr);

  fc_op->setWeightsAndBiases(non_owning_pretrained_fc_weights,
                             non_owning_pretrained_fc_biases);

  _embedding_model = createEmbeddingModel(fc_op, input_dim);

  _two_tower_model = createTwoTowerModel(fc_op, input_dim);

  const data::ColumnDataTypes& input_data_types = {{"text", data_type}};
  _embedding_factory = data::TabularDatasetFactory::make(
      input_data_types,
      /* temporal_tracking_relationships = */ {},
      /* label_blocks = */ {},
      /* label_col_names = */ {},
      /* options = */ options,
      /* force_parallel = */ false);
}

py::object StringEncoder::supervisedTrain(
    const dataset::DataSourcePtr& data_source, const std::string& input_col_1,
    const std::string& input_col_2, const std::string& label_col,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& metrics) {
  const data::ColumnDataTypes& input_data_types = {{input_col_1, _data_type},
                                                   {input_col_2, _data_type}};
  auto label_block = dataset::NumericalCategoricalBlock::make(
      /* col= */ label_col,
      /* n_classes= */ 2);
  auto supervised_factory = data::TabularDatasetFactory::make(
      input_data_types,
      /* temporal_tracking_relationships = */ {},
      /* label_blocks = */ {label_block},
      /* label_col_names = */ {},
      /* tabular_options = */ _options,
      /* force_parallel = */ false);

  auto train_dataset_loader =
      supervised_factory->getDatasetLoader(data_source, /* shuffle = */ true);

  auto old_data = train_dataset_loader->loadAll(defaults::BATCH_SIZE);

  bolt::train::Trainer trainer(_embedding_model);

  auto labels = bolt::train::convertDataset(old_data.back(), 1);
  old_data.pop_back();
  auto input_data =
      bolt::train::convertDatasets(old_data, _two_tower_model->inputDims());

  return py::cast(trainer.train_with_metric_names(
      {input_data, labels}, learning_rate, epochs, metrics));
}

bolt::nn::model::ModelPtr StringEncoder::createEmbeddingModel(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim) {
  auto input = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto output = embedding_op->apply(input);

  // TODO(Josh/Nick): This label and loss is only to get the model to compile.
  // Remove once it is possible to create a model without a loss.
  auto label = bolt::nn::ops::Input::make(/* dim= */ embedding_op->dim());
  auto loss = bolt::nn::loss::CategoricalCrossEntropy::make(output, label);

  return bolt::nn::model::Model::make({input}, {output}, {loss});
}

bolt::nn::tensor::TensorList StringEncoder::encodeBatch(
    const std::vector<std::string>& strings) {
  MapInputBatch map_input;
  for (const auto& string : strings) {
    map_input.push_back({{"text", string}});
  }
  auto input_vectors = _embedding_factory->featurizeInputBatch(map_input).at(0);
  auto input_tensors = bolt::nn::tensor::Tensor::convert(
      input_vectors, _embedding_model->inputDims().at(0));
  return _embedding_model->forward({input_tensors}, /* use_sparsity = */ false);
}

bolt::nn::tensor::TensorPtr StringEncoder::encode(const std::string& string) {
  return encodeBatch({string}).at(0);
}

bolt::nn::model::ModelPtr StringEncoder::createTwoTowerModel(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim) {
  auto input_1 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto input_2 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto output_1 = embedding_op->apply(input_1);

  auto output_2 = embedding_op->apply(input_2);

  auto label = bolt::nn::ops::Input::make(/* dim= */ 1);

  auto loss =
      bolt::nn::loss::EuclideanContrastive::make(output_1, output_2, label, 1);

  return bolt::nn::model::Model::make({input_1, input_2}, {output_1, output_2},
                                      {loss});
}
}  // namespace thirdai::automl::udt