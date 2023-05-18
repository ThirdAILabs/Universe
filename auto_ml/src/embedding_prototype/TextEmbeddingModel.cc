#include "TextEmbeddingModel.h"
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/EuclideanContrastive.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/Validation.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Numerical.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace thirdai::automl::udt {

TextEmbeddingModel::TextEmbeddingModel(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op,
    const data::TextDataTypePtr& text_data_type, data::TabularOptions options,
    float distance_cutoff)
    : _text_data_type(text_data_type), _options(std::move(options)) {
  uint32_t input_dim = embedding_op->inputDim();
  if (_options.feature_hash_range != input_dim) {
    throw std::invalid_argument(
        "The feature hash range of the tabular options passed in must equal "
        "the input dimension.");
  }

  _embedding_model = createEmbeddingModel(embedding_op, input_dim);

  _two_tower_model =
      createTwoTowerModel(embedding_op, input_dim, distance_cutoff);

  const data::ColumnDataTypes& input_data_types = {{"text", text_data_type}};
  _embedding_factory = data::TabularDatasetFactory::make(
      input_data_types,
      /* provided_temporal_relationships = */ {},
      /* label_blocks = */ {},
      /* label_col_names = */ {},
      /* options = */ _options,
      /* force_parallel = */ false);
}

TextEmbeddingModelPtr TextEmbeddingModel::make(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op,
    const data::TextDataTypePtr& text_data_type, data::TabularOptions options,
    float distance_cutoff) {
  return std::make_shared<TextEmbeddingModel>(
      embedding_op, text_data_type, std::move(options), distance_cutoff);
}

py::object TextEmbeddingModel::supervisedTrain(
    const dataset::DataSourcePtr& data_source, const std::string& input_col_1,
    const std::string& input_col_2, const std::string& label_col,
    float learning_rate, uint32_t epochs) {
  const data::ColumnDataTypes& input_data_types_1 = {
      {input_col_1, _text_data_type}};
  const data::ColumnDataTypes& input_data_types_2 = {
      {input_col_2, _text_data_type}};
  auto label_block = dataset::NumericalBlock::make(
      /* col= */ label_col);

  auto supervised_factory_1 = data::TabularDatasetFactory::make(
      input_data_types_1,
      /* provided_temporal_relationships = */ {},
      /* label_blocks = */ {label_block},
      /* label_col_names = */ {},
      /* options = */ _options,
      /* force_parallel = */ false);
  auto supervised_factory_2 = data::TabularDatasetFactory::make(
      input_data_types_2,
      /* provided_temporal_relationships = */ {},
      /* label_blocks = */ {label_block},
      /* label_col_names = */ {},
      /* options = */ _options,
      /* force_parallel = */ false);

  auto train_dataset_loader_1 =
      supervised_factory_1->getDatasetLoader(data_source, /* shuffle = */ true);
  auto boltv1_data_1 = train_dataset_loader_1->loadAll(defaults::BATCH_SIZE);

  data_source->restart();
  auto train_dataset_loader_2 =
      supervised_factory_2->getDatasetLoader(data_source, /* shuffle = */ true);
  auto boltv1_data_2 = train_dataset_loader_2->loadAll(defaults::BATCH_SIZE);

  auto boltv1_data_x = {boltv1_data_1.at(0), boltv1_data_2.at(0)};
  auto boltv1_data_y = boltv1_data_1.back();

  bolt::train::Dataset tensor_data_x = bolt::train::convertDatasets(
      boltv1_data_x, _two_tower_model->inputDims());
  bolt::train::Dataset tensor_data_y =
      bolt::train::convertDataset(boltv1_data_y, 1);

  bolt::train::LabeledDataset tensor_data = {tensor_data_x, tensor_data_y};

  bolt::train::Trainer trainer(_two_tower_model);
  return py::cast(trainer.train(tensor_data, learning_rate, epochs));
}

py::object TextEmbeddingModel::encodeBatch(
    const std::vector<std::string>& strings) {
  MapInputBatch map_input;
  for (const auto& string : strings) {
    map_input.push_back({{"text", string}});
  }
  auto input_tensors = _embedding_factory->featurizeInputBatch(map_input);
  auto output =
      _embedding_model->forward(input_tensors, /* use_sparsity = */ false);

  return bolt::nn::python::tensorToNumpy(output.at(0));
}

py::object TextEmbeddingModel::encode(const std::string& string) {
  return encodeBatch({string});
}

bolt::nn::model::ModelPtr TextEmbeddingModel::createEmbeddingModel(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim) {
  auto input = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto output = embedding_op->apply(input);

  // TODO(Nick): This label and loss is only to get the model to compile.
  // Remove once it is possible to create a model without a loss.
  auto label = bolt::nn::ops::Input::make(/* dim= */ embedding_op->dim());
  auto loss = bolt::nn::loss::CategoricalCrossEntropy::make(output, label);

  return bolt::nn::model::Model::make({input}, {output}, {loss});
}

bolt::nn::model::ModelPtr TextEmbeddingModel::createTwoTowerModel(
    const bolt::nn::ops::FullyConnectedPtr& embedding_op, uint32_t input_dim,
    float distance_cutoff) {
  auto input_1 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto input_2 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

  auto output_1 = embedding_op->apply(input_1);

  auto output_2 = embedding_op->apply(input_2);

  auto label = bolt::nn::ops::Input::make(/* dim= */ 1);

  auto loss = bolt::nn::loss::EuclideanContrastive::make(
      output_1, output_2, label, distance_cutoff);

  return bolt::nn::model::Model::make({input_1, input_2}, {output_1, output_2},
                                      {loss});
}

TextEmbeddingModelPtr createTextEmbeddingModel(
    const bolt::nn::model::ModelPtr& model,
    const data::TabularDatasetFactoryPtr& dataset_factory,
    float distance_cutoff) {
  auto data_types = dataset_factory->inputDataTypes();
  if (data_types.size() != 1) {
    throw std::runtime_error(
        "Creating a text embedding model is only supported for UDT "
        "instantiations with a single input text column and a target column, "
        "but "
        "there was not exactly one input data type (found " +
        std::to_string(data_types.size()) + ")");
  }
  data::TextDataTypePtr text_type = data::asText(data_types.begin()->second);
  if (!text_type) {
    throw std::runtime_error(
        "Creating a text embedding model is only supported for UDT "
        "instantiations with a single text column and a target column, but "
        "we did not find a text column.");
  }

  auto fc_op =
      bolt::nn::ops::FullyConnected::cast(model->opExecutionOrder().at(0));

  auto tabular_options = dataset_factory->tabularOptions();

  return TextEmbeddingModel::make(fc_op, text_type, tabular_options,
                                  distance_cutoff);
}

}  // namespace thirdai::automl::udt