#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/Validation.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/BlockList.h>
#include <dataset/src/blocks/Categorical.h>
#include <pybind11/pybind11.h>
#include <memory>

// #include <bolt/src/nn/loss/EuclideanContrastive.h>

namespace thirdai::automl::udt {

namespace py = pybind11;

class StringEncoder {
 public:
  explicit StringEncoder(uint64_t embedding_dim,
                         const data::TextDataTypePtr& data_type,
                         const data::TabularOptions& options)
      : _data_type(data_type), _options(options) {
    (void)data_type;

    uint32_t input_dim = options.text_pairgrams_word_limit;

    auto embedding_op = bolt::nn::ops::FullyConnected::make(
        /* dim= */ embedding_dim,
        /* input_dim= */ defaults::FEATURE_HASH_RANGE, /* sparsity= */ 1.0,
        /* activation= */ "relu",
        /* sampling=*/nullptr);

    _embedding_model = createEmbeddingModel(embedding_op, input_dim);

    _two_tower_model = createTwoTowerModel(embedding_op, input_dim);

    const data::ColumnDataTypes& input_data_types = {{"text", data_type}};
    _embedding_factory = std::make_shared<data::TabularDatasetFactory>(
        input_data_types,
        /* temporal_tracking_relationships = */ data::TemporalRelationships(),
        /* label_blocks = */ dataset::BlockList(),
        /* label_col_names = */ std::vector<std::string>(), options,
        /* force_parallel = */ false);
  }

  py::object supervisedTrain(const dataset::DataSourcePtr& data_source,
                             const std::string& input_col_1,
                             const std::string& input_col_2,
                             const std::string& label_col, float learning_rate,
                             uint32_t epochs,
                             const std::vector<std::string>& metrics,
                             const std::vector<bolt::CallbackPtr>& callbacks) {
    const data::ColumnDataTypes& input_data_types = {{input_col_1, _data_type},
                                                     {input_col_2, _data_type}};
    auto label_block = dataset::NumericalCategoricalBlock::make(
        /* col= */ label_col,
        /* n_classes= */ 2);
    auto supervised_factory = std::make_shared<data::TabularDatasetFactory>(
        input_data_types,
        /* temporal_tracking_relationships = */ data::TemporalRelationships(),
        /* label_blocks = */ std::vector<dataset::BlockPtr>{label_block},
        /* label_col_names = */ std::vector<std::string>(), _options,
        /* force_parallel = */ false);

    auto train_dataset_loader =
        supervised_factory->getDatasetLoader(data_source, /* shuffle = */ true);

    auto old_data = train_dataset_loader->loadAll(defaults::BATCH_SIZE);

    bolt::train::Trainer trainer(_embedding_model);

    auto labels = bolt::train::convertDataset(old_data.back(), 1);
    old_data.pop_back();
    auto input_data =
        bolt::train::convertDatasets(old_data, _two_tower_model->inputDims());

    return py::cast(trainer.train({input_data, labels}, learning_rate, epochs,
                  /* input_metrics = */ {}, /* validation_data = */ {},
                  /* validation_metrics = */ {},
                  /* steps_per_validation = */ {},
                  /* use_sparsity_in_validation = */ false,
                  /* callbacks = */ callbacks, /* metrics = */ metrics));
  }

  //   py::object encode(const std::string& string) {}

 private:
  static bolt::nn::model::ModelPtr createEmbeddingModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op,
      uint32_t input_dim) {
    auto input = bolt::nn::ops::Input::make(/* dim= */ input_dim);

    auto output = embedding_op->apply(input);

    // TODO(Josh/Nick): This label and loss is only to get the model to compile.
    // Remove once it is possible to create a model without a loss.
    auto label = bolt::nn::ops::Input::make(/* dim= */ embedding_op->dim());
    auto loss = bolt::nn::loss::CategoricalCrossEntropy::make(output, label);

    return bolt::nn::model::Model::make({input}, {output}, {loss});
  }

  static bolt::nn::model::ModelPtr createTwoTowerModel(
      const bolt::nn::ops::FullyConnectedPtr& embedding_op,
      uint32_t input_dim) {
    auto input_1 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

    auto input_2 = bolt::nn::ops::Input::make(/* dim= */ input_dim);

    auto output_1 = embedding_op->apply(input_1);

    auto output_2 = embedding_op->apply(input_2);

    auto label = bolt::nn::ops::Input::make(/* dim= */ 1);

    auto loss =
        bolt::nn::loss::EuclideanContrastive::make(output_1, output_2, label, 1)

            return bolt::nn::model::Model::make({input}, {output}, {loss});
  }

  data::TabularDatasetFactoryPtr _embedding_factory;
  bolt::nn::model::ModelPtr _embedding_model, _two_tower_model;
  data::TextDataTypePtr _data_type;
  data::TabularOptions _options;
};

using StringEncoderPtr = std::shared_ptr<StringEncoder>;

}  // namespace thirdai::automl::udt