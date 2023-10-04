#include "UDTGraphClassifier.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <data/src/Loader.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <limits>

namespace thirdai::automl::udt {

UDTGraphClassifier::UDTGraphClassifier(const ColumnDataTypes& data_types,
                                       const std::string& target_col,
                                       uint32_t n_target_classes,
                                       bool integer_target,
                                       const TabularOptions& options) {
  if (!integer_target) {
    throw exceptions::NotImplemented(
        "We do not yet support non integer classes on graphs.");
  }

  _featurizer = std::make_shared<GraphFeaturizer>(data_types, target_col,
                                                  n_target_classes, options);

  // TODO(Any): Add customization/autotuning like in UDTClassifier
  auto model = createGNN(n_target_classes);

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
  auto train_data_loader = _featurizer->indexAndGetDataLoader(
      data, options.batchSize(), /* shuffle = */ true, options.verbose,
      options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader = _featurizer->indexAndGetDataLoader(
        val_data, defaults::BATCH_SIZE, /* shuffle = */ false, options.verbose);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            train_metrics, val_data_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTGraphClassifier::evaluate(const dataset::DataSourcePtr& data,
                                        const std::vector<std::string>& metrics,
                                        bool sparse_inference, bool verbose,
                                        std::optional<uint32_t> top_k) {
  (void)top_k;

  auto eval_dataset_loader = _featurizer->indexAndGetDataLoader(
      data, defaults::BATCH_SIZE, /* shuffle = */ false, verbose);

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
  archive(cereal::base_class<UDTBackend>(this), _classifier, _featurizer);
}

ModelPtr UDTGraphClassifier::createGNN(uint32_t output_dim) {
  auto node_features_input = bolt::Input::make(defaults::FEATURE_HASH_RANGE);
  auto neighbor_token_input =
      bolt::Input::make(std::numeric_limits<uint32_t>::max());

  auto embedding_1 =
      bolt::RobeZ::make(
          /* num_embedding_lookups = */ 4, /* lookup_size = */ 128,
          /* log_embedding_block_size = */ 20, /* reduction = */ "average")
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