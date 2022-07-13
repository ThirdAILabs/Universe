#include "BoltGraphPython.h"
#include "ConversionUtils.h"
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule) {
  auto graph_submodule = bolt_submodule.def_submodule("graph");

  py::class_<Node, NodePtr>(graph_submodule, "Node");  // NOLINT

  py::class_<FullyConnectedNode, std::shared_ptr<FullyConnectedNode>, Node>(
      graph_submodule, "FullyConnected")
      .def(py::init<uint64_t, ActivationFunction>(), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, std::string>(), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, float, ActivationFunction>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"),
           "Constructs a sparse FullyConnectedLayer object with sampling "
           "parameters autotuned.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, float, std::string>(), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"),
           "Constructs a sparse FullyConnectedLayer object with sampling "
           "parameters autotuned.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<uint64_t, float, ActivationFunction, SamplingConfig>(),
           py::arg("dim"), py::arg("sparsity"), py::arg("activation"),
           py::arg("sampling_config"),
           "Constructs a sparse FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: Enum specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config (SamplingConfig) - Sampling config object to "
           "initialize hash tables/functions.")
      .def(py::init<uint64_t, float, std::string, SamplingConfig>(),
           py::arg("dim"), py::arg("sparsity"), py::arg("activation"),
           py::arg("sampling_config"),
           "Constructs a sparse FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * sparsity: Float - What fraction of nuerons to activate during "
           "training and sparse inference.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n"
           " * sampling_config (SamplingConfig) - Sampling config object to "
           "initialize hash tables/functions.")
#endif
      .def("__call__", &FullyConnectedNode::addPredecessor,
           py::arg("prev_layer"),
           "Tells the graph which layer should act as input to this fully "
           "connected layer.")
      .def("save_parameters", &FullyConnectedNode::saveParameters,
           py::arg("filename"))
      .def("load_parameters", &FullyConnectedNode::loadParameters,
           py::arg("filename"))
      .def("get_sparsity", &FullyConnectedNode::getSparsity)
      .def("set_sparsity", &FullyConnectedNode::setNodeSparsity,
           py::arg("sparsity"));
  py::class_<ConcatenateNode, std::shared_ptr<ConcatenateNode>, Node>(
      graph_submodule, "Concatenate")
      .def(
          py::init<>(),
          "A layer that concatenates an arbitrary number of layers together.\n")
      .def("__call__", &ConcatenateNode::setConcatenatedNodes,
           py::arg("input_layers"),
           "Tells the graph which layers will be concatenated. Must be at "
           "least one node (although this is just an identity function, so "
           "really should be at least two).");

  py::class_<Input, InputPtr, Node>(graph_submodule, "Input")
      .def(py::init<uint32_t>(), py::arg("dim"),
           "Constructs an input layer node for the graph.");

  py::class_<TrainConfig>(graph_submodule, "TrainConfig")
      .def_static("make", &TrainConfig::makeConfig, py::arg("learning_rate"),
                  py::arg("epochs"))
      .def("with_metrics", &TrainConfig::withMetrics, py::arg("metrics"))
      .def("silence", &TrainConfig::silence)
      .def("with_rebuild_hash_tables", &TrainConfig::withRebuildHashTables,
           py::arg("rebuild_hash_tables"))
      .def("with_reconstruct_hash_functions",
           &TrainConfig::withReconstructHashFunctions,
           py::arg("reconstruct_hash_functions"));

  py::class_<PredictConfig>(graph_submodule, "PredictConfig")
      .def_static("make", &PredictConfig::makeConfig)
      .def("enable_sparse_inference", &PredictConfig::enableSparseInference)
      .def("with_metrics", &PredictConfig::withMetrics, py::arg("metrics"))
      .def("silence", &PredictConfig::silence);

  py::class_<BoltGraph>(graph_submodule, "Model")
      .def(py::init<std::vector<InputPtr>, NodePtr>(), py::arg("inputs"),
           py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[Node]) - The input nodes to the graph. Note that "
           "inputs are mapped to input layers by their index.\n"
           " * output (Node) - The output node of the graph.")
      .def("compile", &BoltGraph::compile, py::arg("loss"),
           py::arg("print_when_done") = true,
           "Compiles the graph for the given loss function. In this step the "
           "order in which to compute the layers is determined and various "
           "checks are preformed to ensure the model architecture is correct.")
      .def("train", &BoltGraph::train<BoltBatch>, py::arg("train_data"),
           py::arg("train_labels"), py::arg("train_config"),
           "Trains the network on the given training data.\n"
           "Arguments:\n"
           " * train_data: BoltDataset - Training data. This is a BoltDataset "
           "as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * train_labels: BoltDataset - Training labels. This is a "
           "BoltDataset as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * train_config: TrainConfig - the additional training parameters. "
           "See the TrainConfig documentation above.\n\n"

           "Returns a mapping from metric names to an array their values for "
           "every epoch.")
      .def("predict", &BoltGraph::predict<BoltBatch>, py::arg("test_data"),
           py::arg("test_labels"), py::arg("predict_config"),
           "Predicts the output given the input vectors and evaluates the "
           "predictions based on the given metrics.\n"
           "Arguments:\n"
           " * test_data: BoltDataset - Test data. This is a BoltDataset as "
           "loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * test_labels: BoltDataset - Test labels. This is a BoltDataset "
           "as loaded by thirdai.dataset.load_bolt_svm_dataset or "
           "thirdai.dataset.load_bolt_csv_dataset.\n"
           " * predict_config: PredictConfig - the additional prediction "
           "parameters. See the PredictConfig documentation above.\n\n"

           "Returns a  a mapping from metric names to their values.")
      .def("save", &BoltGraph::save, py::arg("filename"))
      .def_static("load", &BoltGraph::load, py::arg("filename"))
      .def("__str__",
           [](const BoltGraph& model) {
             return model.summarize(/* print = */ false,
                                    /* detailed = */ false);
           })
      .def("freeze_hash_tables", &BoltGraph::freezeHashTables,
           py::arg("insert_labels_if_not_found") = true,
           "Prevents updates to hash tables in the model. If you plan to use "
           "sparse inference, you may get a significant performance "
           "improvement if you call this one or two epochs before you finish "
           "training. Otherwise you should not call this method. If "
           "insert_labels_if_not_found is true then if the output layer is "
           "sparse it will insert the training labels into the hash hash "
           "tables if they are not found for a given input.")
      .def(
          "summary", &BoltGraph::summarize, py::arg("print") = true,
          py::arg("detailed") = false,
          "Returns a summary of the network.\n"
          "Arguments:\n"
          " * print: boolean. Optional, default True. When specified to "
          "\"True\", "
          "summary will print the network summary in addition to returning it. "
          "* detailed: boolean. Optional, default False. When specified to "
          "\"True\", summary will additionally return/print sampling config "
          "details for each layer in the network.")
      // TODO(josh/nick): These are temporary until we have a better story
      // for converting numpy to BoltGraphs
      .def(
          "train_np",
          [](BoltGraph& model, const py::object& train_data_numpy,
             const py::object& train_labels_numpy,
             const TrainConfig& train_config, uint32_t batch_size) {
            auto train_data = convertPyObjectToBoltDataset(train_data_numpy,
                                                           batch_size, false);

            auto train_labels = convertPyObjectToBoltDataset(train_labels_numpy,
                                                             batch_size, true);

            return model.train(train_data.dataset, train_labels.dataset,
                               train_config);
          },
          py::arg("train_data"), py::arg("train_labels"),
          py::arg("train_config"), py::arg("batch_size"))

      .def(
          "predict_np",
          [](BoltGraph& model, const py::object& test_data_numpy,
             const py::object& test_labels_numpy,
             const PredictConfig& predict_config, uint32_t batch_size) {
            auto test_data = convertPyObjectToBoltDataset(test_data_numpy,
                                                          batch_size, false);

            auto test_labels = convertPyObjectToBoltDataset(test_labels_numpy,
                                                            batch_size, true);

            return model.predict(test_data.dataset, test_labels.dataset,
                                 predict_config);
          },
          py::arg("test_data"), py::arg("test_labels"),
          py::arg("predict_config"), py::arg("batch_size") = 256)
      .def("get_layer", &BoltGraph::getNodeByName, py::arg("layer_name"),
           "Looks up a layer (node) of the network by using the layer's "
           "assigned name. As such, must be called after compile. You can "
           "determine which layer is which by printing a graph summary. "
           "Possible operations to perform on the returned object include "
           "setting layer sparsity, freezing weights, or saving to a file");
}

}  // namespace thirdai::bolt::python