#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <_types/_uint32_t.h>
#include <auto_ml/src/dataset_factories/udt/GraphConfig.h>
#include <auto_ml/src/dataset_factories/udt/GraphDatasetFactory.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <utility>
namespace thirdai::automl::models {

class GraphUDT {
 public:
  explicit GraphUDT(bolt::BoltGraphPtr model,
                    data::GraphDatasetFactoryPtr graph_dataset_factory)
      : _model(std::move(model)),
        _graph_dataset_factory(std::move(graph_dataset_factory)) {}

  static GraphUDT buildGraphUDT(
      const data::ColumnDataTypes& data_types,
      const std::string& graph_file_name, const std::string& source,
      const std::string& target,
      const std::vector<std::string>& relationship_columns,
      uint32_t n_target_classes, bool neighbourhood_context = false,
      bool label_context = false, uint32_t kth_neighbourhood = 0,
      char delimeter = ',') {
    auto dataset_config = std::make_shared<data::GraphConfig>(
        data_types, graph_file_name, source, target, relationship_columns,
        n_target_classes, neighbourhood_context, label_context,
        kth_neighbourhood, delimeter);

    auto graph_dataset_factory =
        std::make_shared<data::GraphDatasetFactory>(dataset_config);

    bolt::BoltGraphPtr model;
    model = buildGraphBoltGraph(
        /* input_dims= */ graph_dataset_factory->getInputDim(),
        /* output_dim= */ graph_dataset_factory->getLabelDim(),
        /* hidden_layer_dim= */ 1024);

    return GraphUDT(model, graph_dataset_factory);
  }

  void train(const std::string& file_name, uint32_t epochs, float learning_rate,
             uint32_t batch_size) {
    auto train_config = bolt::TrainConfig::makeConfig(learning_rate, epochs);

    auto data_source =
        dataset::SimpleFileDataSource::make(file_name, batch_size);

    auto dataset_loader = std::make_unique<dataset::DatasetLoader>(
        data_source, _graph_dataset_factory->getBatchProcessor(),
        /* shuffle= */ true);

    auto loaded_data = dataset_loader->loadInMemory();

    auto [train_data, train_labels] = std::move(loaded_data);

    _model->train(train_data, train_labels, train_config);
  }

 private:
  static bolt::BoltGraphPtr buildGraphBoltGraph(
      const std::vector<uint32_t>& input_dims, uint32_t output_dim,
      uint32_t hidden_layer_dim) {
    std::vector<bolt::InputPtr> inputs;
    inputs.reserve(input_dims.size());
    for (uint32_t input_dim : input_dims) {
      inputs.push_back(bolt::Input::make(input_dim));
    }

    auto hidden = bolt::FullyConnectedNode::makeDense(hidden_layer_dim,
                                                      /* activation= */ "relu");
    hidden->addPredecessor(inputs[0]);

    auto sparsity =
        deployment::AutotunedSparsityParameter::autotuneSparsity(output_dim);
    const auto* activation = "softmax";
    auto output = bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity,
                                                          activation);
    output->addPredecessor(hidden);

    auto graph = std::make_shared<bolt::BoltGraph>(
        /* inputs= */ inputs, output);

    graph->compile(
        bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

    return graph;
  }

  bolt::BoltGraphPtr _model;
  data::GraphDatasetFactoryPtr _graph_dataset_factory;
};

}  // namespace thirdai::automl::models