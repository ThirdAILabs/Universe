#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

namespace thirdai::automl::models {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

/**
 * Simple text classifier for CS Disco. Takes in bert tokens and metadata.
 * Trains model to score n output classes using sigmoid and binary cross entropy
 * loss.
 */
class TextClassifier {
 public:
  TextClassifier(uint32_t input_vocab_size, uint32_t metadata_dim,
                 uint32_t n_classes, const std::string& model_size);

  /**
   * Trains on a single batch. The input data should have the format specified
   * in the featurize method. The labels should have the format specified by the
   * convertLabelsToBoltBatch method. Returns the average loss over the batch.
   */
  float trainOnBatch(const py::dict& data, NumpyArray<float>& labels,
                     float learning_rate);

  /**
   * Validates on a single batch. The input data should have the format
   * specified in the featurize method. The labels should have the format
   * specified by the convertLabelsToBoltBatch method.. Returns a dictionary
   * containing the average loss over all the classes, and the loss per class
   * for the batch.
   */
  py::dict validateOnBatch(const py::dict& data, NumpyArray<float>& labels);

  /**
   * Takes in a single input batch and returns an array of the scores for each
   * class for each sample in the batch.  The input data should have the format
   * specified in the featurize method.
   */
  py::array_t<float, py::array::c_style> predict(const py::dict& data);

  /**
   * Used to save the model.
   */
  void save_stream(std::ostream& output_stream) const;

  /**
   * Used to load the model.
   */
  static std::shared_ptr<TextClassifier> load_stream(
      std::istream& input_stream);

 private:
  /**
   * Takes in the input batch and returns the input to the bolt model. The data
   * should be a python dictionary with three fields. Two fields contain the
   * bert tokens for each sample in the batch in CSR format. The field
   * "tokens" should be a flattened numpy array of uint32 of all the tokens. The
   * field "offsets" is a numpy array of uint32 of length (batch_size + 1) that
   * gives the offsets of the tokens for each document. The tokens for document
   * i should be in the range [offsets[i], offsets[i+1]) in the tokens array.
   * The field "metadata" should be a 2D numpy array of 0/1 values (dtype is
   * uint32) that represent the metadata for each document.
   */
  std::vector<BoltBatch> featurize(const py::dict& data) const;

  /**
   * Converts the labels to a bolt batch. Does not copy the data but instead
   * uses BoltVectors that refer to data in the numpy array. The labels should
   * be a 2D numpy arrays of float32.
   */
  BoltBatch convertLabelsToBoltBatch(NumpyArray<float>& labels,
                                     uint32_t batch_size) const;

  /**
   * Gets a list of the nonzero indices in the metadata. Adds the bert token
   * dimension to each nonzero so that the returned indices are ready to be
   * concatenated with the tokens.
   */
  std::vector<uint32_t> getMetadataNonzeros(const uint32_t* metadata) const;

  /**
   * Concatenates the berk tokens and metadata nonzeros into a sparse
   * BoltVector.
   */
  static BoltVector concatTokensAndMetadata(
      const uint32_t* tokens, uint32_t n_tokens,
      const std::vector<uint32_t>& metadata_nonzeros);

  /**
   * Computes the mean binary cross entropy loss over each output class, and
   * optionally the loss foreach class as well.
   */
  std::pair<float, std::optional<NumpyArray<float>>> binaryCrossEntropyLoss(
      const NumpyArray<float>& labels, bool return_loss_per_class);

  /**
   * Checks that the given array has the correct number of dimensions.
   */
  static void verifyArrayHasNDimensions(const NumpyArray<uint32_t>& array,
                                        uint32_t n, const std::string& name);

  /**
   * Checks that the given array has the correct shape. Assumes that
   * verifyArrayHasNDimensions has already been called and it can safely access
   * each dimension of the provided array.
   */
  static void verifyArrayShape(const NumpyArray<uint32_t>& array,
                               std::vector<uint32_t> expected_dims,
                               const std::string& name);

  /**
   * Verifies that the offsets in the "doc_offsets" array are valid and do not
   * exceed the number of tokens.
   */
  static void verifyOffsets(const NumpyArray<uint32_t>& offsets,
                            uint32_t tokens_length);

  // Private constructor for cereal.
  TextClassifier() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive);

  bolt::BoltGraphPtr _model;

  uint32_t _input_vocab_size;
  uint32_t _metadata_dim;
  uint32_t _n_classes;
};

}  // namespace thirdai::automl::models