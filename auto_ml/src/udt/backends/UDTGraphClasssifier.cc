#include "UDTGraphClassifier.h"
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <auto_ml/src/udt/utils/Train.h>

namespace thirdai::automl::udt {

bolt::BoltGraphPtr createGNN(std::vector<uint32_t> input_dims,
                             uint32_t output_dim) {
  auto node_features_input = bolt::Input::make(input_dims.at(0));

  auto neighbor_token_input = bolt::Input::makeTokenInput(
      /* expected_dim = */ input_dims.at(1),
      /* num_tokens_range = */ {0, std::numeric_limits<uint32_t>::max()});

  auto embedding_1 = bolt::EmbeddingNode::make(
      /* num_embedding_lookups = */ 4, /* lookup_size = */ 128,
      /* log_embedding_block_size = */ 20, /* reduction = */ "average");

  auto hidden_1 = bolt::FullyConnectedNode::makeAutotuned(
      /* dim = */ 256, /* sparsity = */ 1, /* activation = */ "relu");

  hidden_1->addPredecessor(node_features_input);

  embedding_1->addInput(neighbor_token_input);

  auto concat_node = bolt::ConcatenateNode::make();

  concat_node->setConcatenatedNodes(/* nodes = */ {hidden_1, embedding_1});

  auto hidden_3 = bolt::FullyConnectedNode::make(
      /* dim = */ 256, /* sparsity = */ 0.5, /* activation = */ "relu",
      /* sampling_config = */ std::make_shared<bolt::RandomSamplingConfig>());

  hidden_3->addPredecessor(concat_node);

  auto output = bolt::FullyConnectedNode::makeAutotuned(
      /* dim = */ output_dim, /* sparsity = */ 1, /* activation =*/"softmax");

  output->addPredecessor(hidden_3);

  std::vector<bolt::InputPtr> inputs = {node_features_input,
                                        neighbor_token_input};

  auto graph = std::make_shared<bolt::BoltGraph>(inputs, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

UDTGraphClassifier::UDTGraphClassifier(const data::ColumnDataTypes& data_types,
                                       const std::string& target_col,
                                       uint32_t n_target_classes,
                                       bool integer_target, char delimiter,
                                       const data::TabularOptions& options) {
  // TODO(Josh): Add other constructor params and throw exception?
  if (!integer_target) {
    throw exceptions::NotImplemented(
        "We do not yet support non integer classes on graphs.");
  }

  _dataset_manager = std::make_shared<data::GraphDatasetManager>(
      data_types, target_col, n_target_classes, delimiter, options);

  // TODO(Josh): Add customization/autotuning like in UDTClassifier
  _model = createGNN(
      /* input_dims = */ _dataset_manager->getInputDims(),
      /* output_dim = */ _dataset_manager->getLabelDim());
}

void UDTGraphClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  utils::DataSourceToDatasetLoader source_to_loader_func =
      [this](const dataset::DataSourcePtr& source) {
        return _dataset_manager->indexAndGetDatasetLoader(source);
      };

  utils::trainClassifier(data, learning_rate, epochs, validation,
                         batch_size_opt, max_in_memory_batches, metrics,
                         callbacks, verbose, logging_interval,
                         source_to_loader_func, _model);
}

py::object UDTGraphClassifier::evaluate(const dataset::DataSourcePtr& data,
                                        const std::vector<std::string>& metrics,
                                        bool sparse_inference,
                                        bool return_predicted_class,
                                        bool verbose, bool return_metrics) {
  auto dataset_loader = _dataset_manager->indexAndGetDatasetLoader(data);
  return utils::evaluate(metrics, sparse_inference, return_predicted_class,
                         verbose, return_metrics, _model, dataset_loader,
                         _binary_prediction_threshold);
}

template void UDTGraphClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTGraphClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTGraphClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _model, _dataset_manager,
          _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTGraphClassifier)