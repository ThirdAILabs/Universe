#include "UDTGraphClassifier.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>

namespace thirdai::automl::udt {

UDTGraphClassifier::UDTGraphClassifier(
    const data::ColumnDataTypes& data_types, const std::string& target_col,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  if (!integer_target) {
    throw exceptions::NotImplemented(
        "We do not yet support non integer classes on graphs.");
  }

  _dataset_manager = std::make_shared<data::GraphDatasetManager>(
      data_types, target_col, n_target_classes, options);

  // TODO(Any): Add customization/autotuning like in UDTClassifier
  auto model = createGNN(
      /* input_dims = */ _dataset_manager->getInputDims(),
      /* output_dim = */ _dataset_manager->getLabelDim(),
      /* model_config = */ model_config,
      /* user_args = */ user_args);

  _classifier =
      utils::Classifier::make(model, /* freeze_hash_tables = */ false);
}

py::object UDTGraphClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto train_dataset_loader = _dataset_manager->indexAndGetLabeledDatasetLoader(
      data, /* shuffle = */ true, /* shuffle_config= */ options.shuffle_config);

  dataset::DatasetLoaderPtr val_dataset_loader;
  if (val_data) {
    val_dataset_loader = _dataset_manager->indexAndGetLabeledDatasetLoader(
        val_data, /* shuffle = */ false);
  }

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            train_metrics, val_dataset_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTGraphClassifier::evaluate(const dataset::DataSourcePtr& data,
                                        const std::vector<std::string>& metrics,
                                        bool sparse_inference, bool verbose,
                                        std::optional<uint32_t> top_k) {
  (void)top_k;

  auto eval_dataset_loader = _dataset_manager->indexAndGetLabeledDatasetLoader(
      data, /* shuffle = */ false);

  return _classifier->evaluate(eval_dataset_loader, metrics, sparse_inference,
                               verbose);
}

template void UDTGraphClassifier::serialize(cereal::BinaryInputArchive&,
                                            const uint32_t version);
template void UDTGraphClassifier::serialize(cereal::BinaryOutputArchive&,
                                            const uint32_t version);

template <class Archive>
void UDTGraphClassifier::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_GRAPH_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_GRAPH_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_GRAPH_CLASSIFIER_VERSION after
  // serialization changes
  archive(cereal::base_class<UDTBackend>(this), _classifier, _dataset_manager);
}

ModelPtr UDTGraphClassifier::createGNN(
    std::vector<uint32_t> input_dims, uint32_t output_dim,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  assert(input_dims.size() == 2);

  if (model_config) {
    return utils::loadModel(input_dims, output_dim, *model_config,
                            /*mach = */ false);
  }

  auto node_features_input = bolt::Input::make(input_dims.at(0));
  auto neighbor_token_input = bolt::Input::make(input_dims.at(1));

  uint64_t num_embedding_lookups =
      user_args.get<uint64_t>("num_embedding_lookups", "integer",
                              /* default num_embedding_lookups = */ 4);
  uint64_t lookup_size = user_args.get<uint64_t>(
      "lookup_size", "integer", /* default lookup_size = */ 128);
  uint64_t log_embedding_block_size =
      user_args.get<uint64_t>("log_embedding_block_size", "integer",
                              /* default log_embedding_block_size = */ 20);
  const std::string reduction = user_args.get<std::string>(
      "reduction", "string", /* default reduction = */ "average");

  auto embedding_1 = bolt::RobeZ::make(num_embedding_lookups, lookup_size,
                                       log_embedding_block_size, reduction)
                         ->apply(neighbor_token_input);

  auto hidden_1 =
      bolt::FullyConnected::make(
          /* dim = */ 256,
          /* input_dim= */ node_features_input->dim(), /* sparsity = */ 1.0,
          /* activation = */ "relu")
          ->apply(node_features_input);

  auto concat_node = bolt::Concatenate::make()->apply({hidden_1, embedding_1});

  auto hidden_3 =
      bolt::FullyConnected::make(
          /* dim = */ 256, /* input_dim= */ concat_node->dim(),
          /* sparsity = */ 0.5, /* activation = */ "relu",
          /* sampling = */ std::make_shared<bolt::RandomSamplingConfig>())
          ->apply(concat_node);

  auto output = bolt::FullyConnected::make(
                    /* dim = */ output_dim, /* input_dim= */ hidden_3->dim(),
                    /* sparsity = */ 1, /* activation =*/"softmax")
                    ->apply(hidden_3);

  auto labels = bolt::Input::make(output_dim);

  auto loss = bolt::CategoricalCrossEntropy::make(output, labels);

  auto model = bolt::Model::make({node_features_input, neighbor_token_input},
                                 {output}, {loss});

  return model;
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTGraphClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTGraphClassifier,
                     thirdai::versions::UDT_GRAPH_CLASSIFIER_VERSION)