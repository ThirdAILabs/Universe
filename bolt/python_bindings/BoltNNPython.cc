#include "BoltNNPython.h"
#include "PybindUtils.h"
#include <bolt/src/graph/DistributedTrainingWrapper.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/Conv.h>
#include <bolt/src/graph/nodes/DlrmAttention.h>
#include <bolt/src/graph/nodes/DotProduct.h>
#include <bolt/src/graph/nodes/Embedding.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/Input3D.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/graph/nodes/Switch.h>
#include <dataset/src/Datasets.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pytypes.h>
#include <optional>
#include <string>

namespace thirdai::bolt::python {

void createBoltNNSubmodule(py::module_& bolt_submodule) {
  auto nn_submodule = bolt_submodule.def_submodule("nn");

  using ParameterArray =
      py::array_t<float, py::array::c_style | py::array::forcecast>;

  using SerializedCompressedVector = py::array_t<char, py::array::c_style>;

  py::class_<ParameterReference>(nn_submodule, "ParameterReference")
      .def("copy", &ParameterReference::copy,
           "Returns a copy of the parameters held in the ParameterReference as "
           "a numpy array.")
      .def("get", &ParameterReference::get,
           /**
            * This means that the lifetime of the returned object is tied to
            * the lifetime of the object this method is called on, such that
            * the parent object cannot be garbage collected will this returned
            * object is still alive.
            */
           py::return_value_policy::reference_internal,
           "Returns a numpy array which shadows the parameters held in the "
           "ParameterReference and acts as a reference to them, modifying this "
           "array will modify the parameters.")

      .def("compress", &ParameterReference::compress,
           py::arg("compression_scheme"), py::arg("compression_density"),
           py::arg("seed_for_hashing"), py::arg("sample_population_size"),
           "Returns a char array representing a compressed vector. "
           "sample_population_size is the number of random samples you take "
           "for estimating a threshold for dragon compression or the number of "
           "sketches needed for count_sketch")
      /*
       * NOTE: The order of set functions is important for correct parameter
       * overloading Pybind will first try the first set method, which will only
       * work with an array of chars (a serialized compressed vector), since
       * SerializedCompressedVector does not specify py::array::forcecast.
       * Pybind will then try the second set method, which will work with any
       * pybind array that can be converted to floats, keeping the normal
       * behavior of setting parameters the same. See
       * https://pybind11.readthedocs.io/en/stable/advanced/functions.html#overload-resolution-order
       */
      .def("set",
           py::overload_cast<SerializedCompressedVector&>(
               &ParameterReference::set),
           py::arg("new_params"),
           "Takes a char array as input that represents a compressed vector "
           "and decompresses and copies into the ParameterReference object.")
      .def("set", py::overload_cast<ParameterArray&>(&ParameterReference::set),
           py::arg("new_params"),
           "Takes a numpy array of floats as input and copies its contents "
           "into the parameters held in the parameter reference object.")
      /*
       * TODO(Shubh):We should make a Compressed vector module at python
       * end to deal with concat function. Since, compressed vectors have an
       * underlying parameter reference, I think that until we have a seperate
       * copmression module, concat can be planted in ParameterReference.
       * Concatenating compressed vectors with the same underlying parameter
       * reference is the same as "concatenating" the parameter references.
       */
      .def_static(
          "concat", &ParameterReference::concat, py::arg("compressed_vectors"),
          "Takes in a list of compressed vector objects and returns a "
          "concatenated compressed vector object. Concatenation is non-lossy "
          "in nature but comes at the cost of a higher memory footprint. "
          "Note: Only concatenate compressed vectors of the same type with "
          "the same hyperparamters.");

  py::class_<GradientReference>(nn_submodule, "GradientReference")
      .def("get_gradients", &GradientReference::getGradients,
           "Returns flattened gradients for the model")
      .def("set_gradients", &GradientReference::setGradients,
           py::arg("flattened_gradients"),
           "Set the gradients for the model with flattened_gradients provided");

  // Needed so python can know that InferenceOutput objects can own memory
  py::class_<InferenceOutputTracker>(nn_submodule,  // NOLINT
                                     "InferenceOutput");

#if THIRDAI_EXPOSE_ALL
#pragma message("THIRDAI_EXPOSE_ALL is defined")                 // NOLINT
  py::class_<thirdai::bolt::SamplingConfig, SamplingConfigPtr>(  // NOLINT
      nn_submodule, "SamplingConfig");

  py::class_<thirdai::bolt::DWTASamplingConfig,
             std::shared_ptr<DWTASamplingConfig>, SamplingConfig>(
      nn_submodule, "DWTASamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("reservoir_size"));

  py::class_<thirdai::bolt::FastSRPSamplingConfig,
             std::shared_ptr<FastSRPSamplingConfig>, SamplingConfig>(
      nn_submodule, "FastSRPSamplingConfig")
      .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("num_tables"),
           py::arg("hashes_per_table"), py::arg("reservoir_size"));

  py::class_<RandomSamplingConfig, std::shared_ptr<RandomSamplingConfig>,
             SamplingConfig>(nn_submodule, "RandomSamplingConfig")
      .def(py::init<>());
#endif

  py::class_<Node, NodePtr>(nn_submodule, "Node")
      .def_property_readonly("name", [](Node& node) { return node.name(); })
      .def("disable_sparse_parameter_updates",
           &Node::disableSparseParameterUpdates,
           "Forces the node to use dense parameter updates.");

  py::class_<FullyConnectedNode, FullyConnectedNodePtr, Node>(nn_submodule,
                                                              "FullyConnected")
      .def(py::init(&FullyConnectedNode::makeDense), py::arg("dim"),
           py::arg("activation"),
           "Constructs a dense FullyConnectedLayer object.\n"
           "Arguments:\n"
           " * dim: Int (positive) - The dimension of the layer.\n"
           " * activation: String specifying the activation function "
           "to use, no restrictions on case - We support five activation "
           "functions: ReLU, Softmax, Tanh, Sigmoid, and Linear.\n")
      .def(py::init(&FullyConnectedNode::makeAutotuned), py::arg("dim"),
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
      .def(py::init(&FullyConnectedNode::make), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"),
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
      // TODO(Nick, Josh): sparsity can be def_property
      .def("get_sparsity", &FullyConnectedNode::getSparsity)
      .def("set_sparsity", &FullyConnectedNode::setSparsity,
           py::arg("sparsity"))
      .def("get_dim", &FullyConnectedNode::outputDim)
      .def_property_readonly(
          "weights",
          [](FullyConnectedNode& node) {
            uint32_t dim = node.outputDim();
            uint32_t prev_node_dim = node.getPredecessors()[0]->outputDim();
            const std::vector<uint32_t> dimensions = {dim, prev_node_dim};
            return ParameterReference(node.getWeightsPtr(), dimensions);
          },
          py::return_value_policy::reference_internal,
          "Returns a ParameterReference object to the weight matrix.")
      .def_property_readonly(
          "biases",
          [](FullyConnectedNode& node) {
            uint32_t dim = node.outputDim();
            return ParameterReference(node.getBiasesPtr(), {dim});
          },
          py::return_value_policy::reference,
          "Returns a ParameterReference object to the bias vector.")
      .def_property_readonly(
          "weight_gradients",
          [](FullyConnectedNode& node) {
            uint32_t dim = node.outputDim();
            uint32_t prev_node_dim = node.getPredecessors()[0]->outputDim();
            const std::vector<uint32_t> dimensions = {dim, prev_node_dim};
            return ParameterReference(node.getWeightGradientsPtr(), dimensions);
          },
          py::return_value_policy::reference_internal,
          "Returns a ParameterReference object to the weight gradients matrix.")
      .def_property_readonly(
          "bias_gradients",
          [](FullyConnectedNode& node) {
            uint32_t dim = node.outputDim();
            return ParameterReference(node.getBiasGradientsPtr(), {dim});
          },
          py::return_value_policy::reference,
          "Returns a ParameterReference object to the bias gradients vector.");

  py::class_<ConvNode, ConvNodePtr, Node>(nn_submodule, "Conv")
      .def(py::init(&ConvNode::makeDense), py::arg("num_filters"),
           py::arg("activation"), py::arg("kernel_size"),
           py::arg("next_kernel_size"))
      .def(py::init(&ConvNode::makeAutotuned), py::arg("num_filters"),
           py::arg("sparsity"), py::arg("activation"), py::arg("kernel_size"),
           py::arg("next_kernel_size"))
#if THIRDAI_EXPOSE_ALL
      .def(py::init(&ConvNode::make), py::arg("num_filters"),
           py::arg("sparsity"), py::arg("activation"), py::arg("kernel_size"),
           py::arg("next_kernel_size"), py::arg("sampling_config"))
#endif
      .def("__call__", &ConvNode::addPredecessor, py::arg("prev_layer"),
           "Tells the graph which layer should act as input to this conv "
           "layer.");

  py::class_<LayerNormNode, std::shared_ptr<LayerNormNode>, Node>(
      nn_submodule, "LayerNormalization")
      .def(py::init(&LayerNormNode::make),
           "Constructs a normalization layer object.")
      .def(py::init(&LayerNormNode::makeWithConfig),
           py::arg("layer_norm_config"),
           "Constructs a normalization layer object"
           "Arguments:\n"
           " * layer_norm_config: NormalizationLayerConfig - configuration "
           "parameters required for normalizing the input. \n")
      .def("__call__", &LayerNormNode::addPredecessor, py::arg("prev_layer"),
           "Tells the graph which layer should act as input to this "
           "normalization layer.");

  py::class_<ConcatenateNode, std::shared_ptr<ConcatenateNode>, Node>(
      nn_submodule, "Concatenate")
      .def(
          py::init(&ConcatenateNode::make),
          "A layer that concatenates an arbitrary number of layers together.\n")
      .def("__call__", &ConcatenateNode::setConcatenatedNodes,
           py::arg("input_layers"),
           "Tells the graph which layers will be concatenated. Must be at "
           "least one node (although this is just an identity function, so "
           "really should be at least two).");

#if THIRDAI_EXPOSE_ALL
  py::class_<SwitchNode, std::shared_ptr<SwitchNode>, Node>(nn_submodule,
                                                            "Switch")
      .def(py::init(&SwitchNode::makeDense), py::arg("dim"),
           py::arg("activation"), py::arg("n_layers"))
      .def(py::init(&SwitchNode::makeAutotuned), py::arg("dim"),
           py::arg("sparsity"), py::arg("activation"), py::arg("n_layers"))
      .def("__call__", &SwitchNode::addPredecessors, py::arg("prev_layer"),
           py::arg("token_input"));
#endif

  py::class_<EmbeddingNode, EmbeddingNodePtr, Node>(nn_submodule, "Embedding")
      .def(py::init(&EmbeddingNode::make), py::arg("num_embedding_lookups"),
           py::arg("lookup_size"), py::arg("log_embedding_block_size"),
           py::arg("reduction"), py::arg("num_tokens_per_input") = std::nullopt,
           "Constructs an Embedding layer that can be used in the graph.\n"
           "Arguments:\n"
           " * num_embedding_lookups: Int (positive) - The number of embedding "
           "lookups to perform for each token.\n"
           " * lookup_size: Int (positive) - How many consutive values to "
           "select as part of the embedding for each embedding lookup.\n"
           " * log_embedding_block_size: Int (positive) The log base 2 of the "
           "total size of the embedding block.\n")
      .def("__call__", &EmbeddingNode::addInput, py::arg("token_input_layer"),
           "Tells the graph which token input to use for this Embedding Node.")
      .def_property_readonly(
          "weights",
          [](EmbeddingNode& node) {
            std::vector<float>& raw_embedding_block =
                node.getRawEmbeddingBlock();
            return ParameterReference(
                raw_embedding_block.data(),
                {static_cast<uint32_t>(raw_embedding_block.size())});
          },
          py::return_value_policy::reference_internal,
          "Returns a ParameterReference object to the weight matrix.")
      .def_property_readonly(
          "weight_gradients",
          [](EmbeddingNode& node) {
            std::vector<float>& raw_embedding_block_gradient =
                node.getRawEmbeddingBlockGradient();
            return ParameterReference(
                raw_embedding_block_gradient.data(),
                {static_cast<uint32_t>(raw_embedding_block_gradient.size())});
          },
          py::return_value_policy::reference_internal,
          "Returns a ParameterReference object to the weight gradients "
          "matrix.");

  py::class_<DotProductNode, DotProductNodePtr, Node>(nn_submodule,
                                                      "DotProduct")
      .def(py::init(&DotProductNode::make))
      .def("__call__", &DotProductNode::setPredecessors, py::arg("lhs"),
           py::arg("rhs"));

  nn_submodule.def("TokenInput", &Input::makeTokenInput, py::arg("dim"),
                   py::arg("num_tokens_range"));

  py::class_<Input, InputPtr, Node>(nn_submodule, "Input")
      .def(py::init(&Input::make), py::arg("dim"),
           "Constructs an input layer node for the graph.");

  py::class_<Input3D, std::shared_ptr<Input3D>, Input>(nn_submodule, "Input3D")
      .def(py::init(&Input3D::make), py::arg("dim"),
           "Constructs a 3D input layer node for the graph.");

  py::class_<NormalizationLayerConfig>(nn_submodule, "LayerNormConfig")
      .def_static("make", &NormalizationLayerConfig::makeConfig)
      .def("center", &NormalizationLayerConfig::setCenteringFactor,
           py::arg("beta_regularizer"),
           "Sets the centering factor for the normalization configuration")
      .def("scale", &NormalizationLayerConfig::setScalingFactor,
           py::arg("gamma_regularizer"),
           "Sets the scaling factor the the normalization configuration.");

  py::class_<DlrmAttentionNode, DlrmAttentionNodePtr, Node>(nn_submodule,
                                                            "DlrmAttention")
      .def(py::init())
      .def("__call__", &DlrmAttentionNode::setPredecessors, py::arg("fc_layer"),
           py::arg("embedding_layer"));

  py::class_<BoltGraph, BoltGraphPtr>(nn_submodule, "Model")
      .def(py::init<std::vector<InputPtr>, NodePtr>(), py::arg("inputs"),
           py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[Node]) - The input nodes to the graph. Note that "
           "inputs are mapped to input layers by their index.\n"
           " * output (Node) - The output node of the graph.",
           bolt::python::OutputRedirect())
      .def(py::init<std::vector<InputPtr>, NodePtr>(), py::arg("inputs"),
           py::arg("output"),
           "Constructs a bolt model from a layer graph.\n"
           "Arguments:\n"
           " * inputs (List[InputNode]) - The input nodes to the graph. Note "
           "that "
           "inputs are mapped to input layers by their index.\n"
           " * inputs (List[TokenInput]) - The token input nodes to the graph. "
           "Note that "
           "token inputs are mapped to token input layers by their index.\n"
           " * output (Node) - The output node of the graph.",
           bolt::python::OutputRedirect())
      .def("compile", &BoltGraph::compile, py::arg("loss"),
           py::arg("print_when_done") = true,
           "Compiles the graph for the given loss function. In this step the "
           "order in which to compute the layers is determined and various "
           "checks are preformed to ensure the model architecture is correct.",
           bolt::python::OutputRedirect())
      // Helper method that covers the common case of training based off of a
      // single BoltBatch dataset
      .def(
          "train",
          [](BoltGraph& model, const dataset::BoltDatasetPtr& data,
             const dataset::BoltDatasetPtr& labels,
             const TrainConfig& train_config) {
            return model.train({data}, labels, train_config);
          },
          py::arg("train_data"), py::arg("train_labels"),
          py::arg("train_config"), bolt::python::OutputRedirect())
      .def("train", &BoltGraph::train, py::arg("train_data"),
           py::arg("train_labels"), py::arg("train_config"),
           R"pbdoc(  
Trains the network on the given training data and labels with the given training
config.

Args:
    train_data (List[BoltDataset] or BoltDataset): The data to train the model with. 
        There should be exactly one BoltDataset for each Input node in the Bolt
        model, and each BoltDataset should have the same total number of 
        vectors and the same batch size. The batch size for training is the 
        batch size of the passed in BoltDatasets (you can specify this batch 
        size when loading or creating a BoltDataset).
    train_labels (BoltDataset): The labels to use as ground truth during 
        training. There should be the same number of total vectors and the
        same batch size in this BoltDataset as in the train_data list.
    train_config (TrainConfig): The object describing all other training
        configuration details. See the TrainConfig documentation for more
        information as to possible options. This includes the number of epochs
        to train for, the verbosity of the training, the learning rate, and so
        much more!

Returns:
    Dict[Str, List[float]]:
    A dictionary from metric name to a list of the value of that metric 
    for each epoch (this also always includes an entry for 'epoch_times'). The 
    metrics that are returned are the metrics requested in the TrainConfig.

Notes:
    Sparse bolt training was originally based off of SLIDE. See [1]_ for more details

References:
    .. [1] "SLIDE : In Defense of Smart Algorithms over Hardware Acceleration for Large-Scale Deep Learning Systems" 
            https://arxiv.org/pdf/1903.03129.pdf.

Examples:
    >>> train_config = (
            bolt.TrainConfig(learning_rate=0.001, epochs=3)
            .with_metrics(["categorical_accuracy"])
        )
    >>> metrics = model.train(
            train_data=train_data, train_labels=train_labels, train_config=train_config
        )
    >>> print(metrics)
    {'epoch_times': [1.7, 3.4, 5.2], 'categorical_accuracy': [0.4665, 0.887, 0.9685]}

That's all for now, folks! More docs coming soon :)

)pbdoc",
           bolt::python::OutputRedirect())
#if THIRDAI_EXPOSE_ALL
      .def(
          "get_input_gradients_single",
          [](BoltGraph& model, std::vector<BoltVector>&& input_data,
             bool explain_prediction_using_highest_activation,
             std::optional<uint32_t> neuron_to_explain) {
            auto gradients = model.getInputGradientSingle(
                std::move(input_data),
                explain_prediction_using_highest_activation, neuron_to_explain);
            return dagGetInputGradientSingleWrapper(gradients);
          },
          py::arg("input_data"),
          py::arg("explain_prediction_using_highest_activation") = true,
          py::arg("neuron_to_explain") = std::nullopt,
          "Get the values of input gradients when back propagate "
          "label with the highest activation or second highest "
          "activation or with the required label."
          "Arguments:\n"
          " * input_data: The input is single input sample."
          " * explain_prediction_using_highest_activation: Boolean, if set to "
          "True, gives gradients "
          "correspond to "
          "highest activation, Otherwise gives gradients corresponds to "
          "second highest activation."
          " * neuron_to_explain: Optional, expected label for input vector, if "
          "it is provided then model gives gradients corresponds to that label."
          " Returns a tuple consists of (0) optional, it only returns the "
          "corresponding indices for sparse inputs."
          " and (1) list of gradients "
          "corresponds to the input vector.")
#endif
      .def(
          "explain_prediction",
          [](BoltGraph& model, std::vector<BoltVector>&& input_data) {
            auto gradients =
                model.getInputGradientSingle(std::move(input_data));
            return dagGetInputGradientSingleWrapper(gradients);
          },
          py::arg("input_data"),
          "explains why this model predicted the output with respect to input "
          "vector features."
          "Arguments:\n"
          " * input_data: A single input sample."
          " Returns a tuple consists of (0) optional, it only returns the "
          "corresponding indices for sparse inputs."
          " and (1) list of values which"
          "corresponds to the features in input vector.")
      .def(
          "get_input_confidence",
          [](BoltGraph& model, std::vector<BoltVector>&& input_data) {
            auto gradients = model.getInputGradientSingle(
                std::move(input_data),
                /*explain_prediction_using_highest_activation= */ false);
            return dagGetInputGradientSingleWrapper(gradients);
          },
          py::arg("input_data"),
          "gets some confidence values for the input vector, high the "
          "confidence more important the input feature for the model to "
          "predict."
          "Arguments:\n"
          " * input_data: A single input sample."
          " Returns a tuple consists of (0) optional, it only returns the "
          "corresponding indices for sparse inputs."
          " and (1) list of confidence values which "
          "tells how much important each feature in the input vector.")
      .def(
          "explain_required_label",
          [](BoltGraph& model, std::vector<BoltVector>&& input_data,
             uint32_t neuron_to_explain) {
            auto gradients = model.getInputGradientSingle(
                std::move(input_data),
                /*explain_prediction_using_highest_activation= */ false,
                neuron_to_explain);
            return dagGetInputGradientSingleWrapper(gradients);
          },
          py::arg("input_data"), py::arg("neuron_to_explain"),
          "explains how the input vector values should change "
          "so that model predicts the desired label."
          "Arguments:\n"
          " * input_data: A single input sample."
          " * neuron_to_explain: desired label user wants model to predict."
          " Returns a tuple consists of (0) optional, it only returns the "
          "corresponding indices for sparse inputs."
          " and (1) list of values which"
          "tells how each feature should change in the input vector.")
      // Helper method that covers the common case of inference based off of a
      // single BoltBatch dataset
      .def(
          "evaluate",
          [](BoltGraph& model, const dataset::BoltDatasetPtr& data,
             const dataset::BoltDatasetPtr& labels,
             const EvalConfig& eval_config) {
            return dagEvaluatePythonWrapper(model, {data}, labels, eval_config);
          },
          py::arg("test_data"), py::arg("test_labels"), py::arg("eval_config"),
          bolt::python::OutputRedirect())
      .def(
          "evaluate", &dagEvaluatePythonWrapper, py::arg("test_data"),
          py::arg("test_labels"), py::arg("eval_config"),
          "Predicts the output given the input vectors and evaluates the "
          "predictions based on the given metrics.\n"
          "Arguments:\n"
          " * test_data: PyObject - Test data, in the same one of 3 formats as "
          "accepted by the train method (a BoltDataset, a dense numpy array, "
          "or a tuple of dense numpy arrays representing a sparse dataset)\n"
          " * test_labels: PyObject - Test labels, in the same format as "
          "test_data. This can also additionally be passed as None, in which "
          "case no metrics can be computed.\n"
          " * eval_config: EvalConfig - the additional prediction "
          "parameters. See the EvalConfig documentation above.\n\n"
          "Returns a tuple, where the first element is a mapping from metric "
          "names to their values. The second element, the output activation "
          "matrix, is only present if dont_return_activations was not called. "
          "The third element, the active neuron matrix, is only present if "
          "we are returning activations AND the ouptut is sparse.",
          bolt::python::OutputRedirect())
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
           "setting layer sparsity, freezing weights, or saving to a file")
      .def("nodes", &BoltGraph::getNodeTraversalOrder,
           "Returns a list of all Nodes that make up the graph in traversal "
           "order. This list is guaranetted to be static after a model is "
           "compiled.")
      .def(getPickleFunction<BoltGraph>());

  py::class_<DistributedTrainingWrapper>(bolt_submodule,
                                         "DistributedTrainingWrapper")
      .def(py::init<BoltGraphPtr, TrainConfig, uint32_t>(), py::arg("model"),
           py::arg("train_config"), py::arg("worker_id"))
      .def("compute_and_store_batch_gradients",
           &DistributedTrainingWrapper::computeAndStoreBatchGradients,
           py::arg("batch_idx"),
           "Uses the batch_idx'th batch of the currently "
           "set dataset to accumulate a single batche's gradients in the "
           "wrapped Bolt model.")
      .def("update_parameters", &DistributedTrainingWrapper::updateParameters,
           "Updates the parameters of the wrapped Bolt model. You should call "
           "this manually after setting the gradients of the wrapped model.")
      .def("num_batches", &DistributedTrainingWrapper::numBatches)
      .def("set_datasets", &DistributedTrainingWrapper::setDatasets,
           py::arg("train_data"), py::arg("train_labels"),
           "Sets the current train data and labels the wrapper class uses for "
           "computeAndStoreBatchGradients. We need this method instead of just "
           "passing in a single pair of training data and training labels at "
           "construction time because we might have a streaming dataset we "
           "want to train on, which will entail switching out the current "
           "datasets dynamically. If this is not the first time this method has"
           "been called, the batch sizes of the passed in datasets must be the "
           "same as when this method was called the first time.")
      .def("finish_training", &DistributedTrainingWrapper::finishTraining, "")
      .def_property_readonly(
          "model",
          [](DistributedTrainingWrapper& node) { return node.getModel(); },
          py::return_value_policy::reference_internal,
          "The underlying Bolt model wrapped by this "
          "DistributedTrainingWrapper.")
      .def(
          "gradient_reference",
          [](DistributedTrainingWrapper& node) {
            return GradientReference(*node.getModel().get());
          },
          py::return_value_policy::reference_internal,
          "Returns gradient reference for Distributed Training Wrapper");

  createLossesSubmodule(nn_submodule);
}

void createLossesSubmodule(py::module_& nn_submodule) {
  auto losses_submodule = nn_submodule.def_submodule("losses");

  py::class_<LossFunction, std::shared_ptr<LossFunction>>(  // NOLINT
      losses_submodule, "LossFunction", "Base class for all loss functions");

  py::class_<CategoricalCrossEntropyLoss,
             std::shared_ptr<CategoricalCrossEntropyLoss>, LossFunction>(
      losses_submodule, "CategoricalCrossEntropy",
      "A loss function for multi-class (one label per sample) classification "
      "tasks.")
      .def(py::init<>(), "Constructs a CategoricalCrossEntropyLoss object.");

  py::class_<BinaryCrossEntropyLoss, std::shared_ptr<BinaryCrossEntropyLoss>,
             LossFunction>(
      losses_submodule, "BinaryCrossEntropy",
      "A loss function for multi-label (multiple class labels per each sample) "
      "classification tasks.")
      .def(py::init<>(), "Constructs a BinaryCrossEntropyLoss object.");

  py::class_<MeanSquaredError, std::shared_ptr<MeanSquaredError>, LossFunction>(
      losses_submodule, "MeanSquaredError",
      "A loss function that minimizes mean squared error (MSE) for regression "
      "tasks. "
      ":math:`MSE = sum( (actual - prediction)^2 )`")
      .def(py::init<>(), "Constructs a MeanSquaredError object.");

  py::class_<WeightedMeanAbsolutePercentageErrorLoss,
             std::shared_ptr<WeightedMeanAbsolutePercentageErrorLoss>,
             LossFunction>(
      losses_submodule, "WeightedMeanAbsolutePercentageError",
      "A loss function to minimize weighted mean absolute percentage error "
      "(WMAPE) "
      "for regression tasks. :math:`WMAPE = 100% * sum(|actual - prediction|) "
      "/ sum(|actual|)`")
      .def(py::init<>(),
           "Constructs a WeightedMeanAbsolutePercentageError object.");

  py::class_<MarginBCE, std::shared_ptr<MarginBCE>, LossFunction>(
      losses_submodule, "MarginBCE")
      .def(py::init<float, float, bool>(), py::arg("positive_margin"),
           py::arg("negative_margin"), py::arg("bound"));
}

py::tuple dagEvaluatePythonWrapper(BoltGraph& model,
                                   const dataset::BoltDatasetList& data,
                                   const dataset::BoltDatasetPtr& labels,
                                   const EvalConfig& eval_config) {
  auto [metrics, output] = model.evaluate(data, labels, eval_config);

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

py::tuple dagGetInputGradientSingleWrapper(
    const std::pair<std::optional<std::vector<uint32_t>>, std::vector<float>>&
        gradients) {
  if (gradients.first == std::nullopt) {
    return py::cast(gradients.second);
  }
  return py::cast(gradients);
}

}  // namespace thirdai::bolt::python
