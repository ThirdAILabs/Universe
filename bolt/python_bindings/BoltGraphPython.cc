#include "BoltGraphPython.h"
#include "ConversionUtils.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/TokenInput.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/batch_types/BoltTokenBatch.h>
#include <optional>

namespace thirdai::bolt::python {

void createBoltGraphSubmodule(py::module_& bolt_submodule) {
  auto graph_submodule = bolt_submodule.def_submodule("graph");

  py::class_<Node, NodePtr>(graph_submodule, "Node");  // NOLINT

  // Needed so python can know that InferenceOutput objects can own memory
  py::class_<InferenceOutputTracker>(graph_submodule,  // NOLINT
                                     "InferenceOutput");

  py::class_<FullyConnectedNode, FullyConnectedNodePtr, Node>(graph_submodule,
                                                              "FullyConnected")
      .def(py::init<uint64_t, const std::string&>(), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init<uint64_t, float, const std::string&>(), py::arg("dim"),
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
      .def(py::init<uint64_t, float, const std::string&, SamplingConfigPtr>(),
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
      .def("get_sparsity", &FullyConnectedNode::getNodeSparsity)
      .def("set_sparsity", &FullyConnectedNode::setNodeSparsity,
           py::arg("sparsity"))
      .def("get_dim", &FullyConnectedNode::outputDim);

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

  py::class_<EmbeddingNode, EmbeddingNodePtr, Node>(graph_submodule,
                                                    "Embedding")
      .def(py::init<uint32_t, uint32_t, uint32_t>(),
           py::arg("num_embedding_lookups"), py::arg("lookup_size"),
           py::arg("log_embedding_block_size"),
           "Constructs an Embedding layer that can be used in the graph.\n"
           "Arguments:\n"
           " * num_embedding_lookups: Int (positive) - The number of embedding "
           "lookups to perform for each token.\n"
           " * lookup_size: Int (positive) - How many consutive values to "
           "select as part of the embedding for each embedding lookup.\n"
           " * log_embedding_block_size: Int (positive) The log base 2 of the "
           "total size of the embedding block.\n")
      .def("__call__", &EmbeddingNode::addInput, py::arg("token_input_layer"),
           "Tells the graph which token input to use for this Embedding Node.");

  py::class_<Input, InputPtr, Node>(graph_submodule, "Input")
      .def(py::init<uint32_t>(), py::arg("dim"),
           "Constructs an input layer node for the graph.");

  py::class_<TokenInput, TokenInputPtr, Node>(graph_submodule, "TokenInput")
      .def(py::init<>(), "Constructs a token input layer node for the graph.");

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
      .def("silence", &PredictConfig::silence)
      .def("return_activations", &PredictConfig::returnActivations);

  py::class_<BoltGraph>(graph_submodule, "Model")
      .def(py::init<std::vector<InputPtr>, NodePtr>(), py::arg("inputs"),
           py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[Node]) - The input nodes to the graph. Note that "
           "inputs are mapped to input layers by their index.\n"
           " * output (Node) - The output node of the graph.")
      .def(py::init<std::vector<InputPtr>, std::vector<TokenInputPtr>,
                    NodePtr>(),
           py::arg("inputs"), py::arg("token_inputs"), py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[InputNode]) - The input nodes to the graph. Note "
           "that "
           "inputs are mapped to input layers by their index.\n"
           " * inputs (List[TokenInput]) - The token input nodes to the graph. "
           "Note that "
           "token inputs are mapped to token input layers by their index.\n"
           " * output (Node) - The output node of the graph.")
      .def("compile", &BoltGraph::compile, py::arg("loss"),
           py::arg("print_when_done") = true,
           "Compiles the graph for the given loss function. In this step the "
           "order in which to compute the layers is determined and various "
           "checks are preformed to ensure the model architecture is correct.")
      // Helper method that covers the common case of training based off of a
      // single BoltBatch dataset
      .def(
          "train",
          [](BoltGraph& model, const dataset::BoltDatasetPtr& data,
             const dataset::BoltDatasetPtr& labels,
             const TrainConfig& train_config) {
            return model.train({data}, /* train_tokens = */ {}, labels,
                               train_config);
          },
          py::arg("train_data"), py::arg("train_labels"),
          py::arg("train_config"))
      .def(
          "train", &BoltGraph::train, py::arg("train_data"),
          py::arg("train_tokens"), py::arg("train_labels"),
          py::arg("train_config"),
          "Trains the network on the given training data.\n"
          "Arguments:\n"
          " * train_data: PyObject - Training data. This can be one of "
          "three things. First, it can be a BoltDataset as loaded by "
          "thirdai.dataset.load_bolt_svm_dataset or "
          "thirdai.dataset.load_bolt_csv_dataset. Second, it can be a dense "
          "numpy array of float32 where each row in the array is interpreted "
          "as a vector. Thid, it can be a sparse dataset represented by a "
          " tuple of three numpy arrays (indices, values, offsets), where "
          "indices and offsets are uint32 and values are float32. In this case "
          "indices is a 1D array of all the nonzero indices concatenated, "
          "values is a 1D array of all the nonzero values concatenated, and "
          "offsets are the start positions in the indices and values array of "
          "each vector plus one extra element at the end of the array "
          "representing the total number of nonzeros. This is so that "
          "indices[offsets[i], offsets[i + 1]] contains the indices of the ith "
          "vector and values[offsets[i], offsets[i+1] contains the values of "
          "the ith vector. For example, if we have the vectors "
          "{0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, 0.0, 4.0}, then the indices "
          "array is {1, 3, 3}, the values array is {1.5, 9.0, 4.0} and the "
          "offsets array is {0, 2, 3}.\n"
          " * train_labels: PyObject - Training labels. This can be one of "
          "three things. First it can be a BoltDataset as loaded by "
          "thirdai.dataset.load_bolt_svm_dataset or "
          "thirdai.dataset.load_bolt_csv_dataset. Second, it can be a dense "
          "numpy array of float32 where each row in the array is interpreted "
          "as a label vector. Thid, it can be a set of sparse vectors (each "
          "vector is a label vector) represented as three numpy arrays "
          "(indices, values, offsets) where indices and offsets are uint32 "
          "and values are float32. In this case indices is a 1D array of all "
          "the nonzero indices concatenated, values is a 1D array of all the "
          "nonzero values concatenated, and offsets are the start positions "
          "in the indices and values array of each vector plus one extra "
          "element at the end of the array representing the total number of "
          "nonzeros. This is so that indices[offsets[i], offsets[i + 1]] "
          "contains the indices of the ith vector and values[offsets[i], "
          "offsets[i+1] contains the values of the ith vector. For example, if "
          "we have the vectors {0.0, 1.5, 0.0, 9.0} and {0.0, 0.0, 0.0, 4.0}, "
          "then the indices array is {1, 3, 3}, the values array is {1.5, "
          "9.0, 4.0}, and the offsets array is {0, 2, 3}.\n"
          " * train_config: TrainConfig - the additional training parameters. "
          "See the TrainConfig documentation above.\n\n"
          "Returns a mapping from metric names to an array of their values for "
          "every epoch.")
      .def(
          "get_input_gradients",
          [](BoltGraph& model, const dataset::BoltDatasetPtr& input_data,
             bool best_index = true,
             const std::vector<uint32_t>& required_labels =
                 std::vector<uint32_t>()) {
            auto gradients = model.getInputGradients(
                input_data, /* train_tokens = */ nullptr, best_index,
                required_labels);
            return gradients;
          },
          py::arg("input_data"), py::arg("best_index") = true,
          py::arg("required_labels") = std::vector<uint32_t>())
      .def("get_input_gradients", &BoltGraph::getInputGradients,
           py::arg("input_data"), py::arg("input_tokens"),
           py::arg("best_index") = true,
           py::arg("required_labels") = std::vector<uint32_t>())
      // Helper method that covers the common case of inference based off of a
      // single BoltBatch dataset
      .def(
          "predict",
          [](BoltGraph& model, const dataset::BoltDatasetPtr& data,
             const dataset::BoltDatasetPtr& labels,
             const PredictConfig& predict_config) {
            return dagPredictPythonWrapper(model, {data}, /* tokens = */ {},
                                           labels, predict_config);
          },
          py::arg("test_data"), py::arg("test_labels"),
          py::arg("predict_config"))
      .def(
          "predict", &dagPredictPythonWrapper, py::arg("test_data"),
          py::arg("test_tokens"), py::arg("test_labels"),
          py::arg("predict_config"),
          "Predicts the output given the input vectors and evaluates the "
          "predictions based on the given metrics.\n"
          "Arguments:\n"
          " * test_data: PyObject - Test data, in the same one of 3 formats as "
          "accepted by the train method (a BoltDataset, a dense numpy array, "
          "or a tuple of dense numpy arrays representing a sparse dataset)\n"
          " * test_labels: PyObject - Test labels, in the same format as "
          "test_data. This can also additionally be passed as None, in which "
          "case no metrics can be computed.\n"
          " * predict_config: PredictConfig - the additional prediction "
          "parameters. See the PredictConfig documentation above.\n\n"
          "Returns a tuple, where the first element is a mapping from metric "
          "names to their values. The second element, the output activation "
          "matrix, is only present if dont_return_activations was not called. "
          "The third element, the active neuron matrix, is only present if "
          "we are returning activations AND the ouptut is sparse.")
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
      .def("get_layer", &BoltGraph::getNodeByName, py::arg("layer_name"),
           "Looks up a layer (node) of the network by using the layer's "
           "assigned name. As such, must be called after compile. You can "
           "determine which layer is which by printing a graph summary. "
           "Possible operations to perform on the returned object include "
           "setting layer sparsity, freezing weights, or saving to a file");
}

py::tuple dagPredictPythonWrapper(BoltGraph& model,
                                  const dataset::BoltDatasetList& data,
                                  const dataset::BoltTokenDatasetList& tokens,
                                  const dataset::BoltDatasetPtr& labels,
                                  const PredictConfig& predict_config) {
  auto [metrics, output] = model.predict(data, tokens, labels, predict_config);

  // We need to get these now because we are about to std::move output
  const float* activation_pointer = output.getNonowningActivationPointer();
  const uint32_t* active_neuron_pointer =
      output.getNonowningActiveNeuronPointer();
  uint32_t num_nonzeros = output.numNonzerosInOutput();
  uint64_t dataset_len = output.numSamples();

  // At first, the InferenceOutput object owns the memory for the
  // activation and active_neuron vectors. We want to use it as the
  // owning object when we build the numpy array, but to do that we
  // need to cast it to a py::object. Importantly, we need to use
  // std::move to ensure that we are casting output itself to a python
  // object, not a copy of it. See return_value_policy::automatic here
  // https://pybind11.readthedocs.io/en/stable/advanced/functions.html#return-value-policies
  py::object output_handle = py::cast(std::move(output));

  return constructPythonInferenceTuple(
      py::cast(metrics), dataset_len, num_nonzeros,
      /* activations = */ activation_pointer,
      /* active_neurons = */ active_neuron_pointer,
      /* activation_handle = */ output_handle,
      /* active_neuron_handle = */ output_handle);
}

}  // namespace thirdai::bolt::python