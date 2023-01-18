#include "TextClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/Node.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

namespace thirdai::automl::models {

TextClassifier::TextClassifier(uint32_t input_vocab_size, uint32_t metadata_dim,
                               uint32_t n_classes,
                               const std::string& model_size)
    : _input_vocab_size(input_vocab_size),
      _metadata_dim(metadata_dim),
      _n_classes(n_classes) {
  auto input = bolt::Input::make(input_vocab_size + metadata_dim);

  bolt::FullyConnectedNodePtr hidden =
      getHiddenLayer(model_size)->addPredecessor(input);

  auto output = bolt::FullyConnectedNode::makeDense(
                    /* dim= */ n_classes, /* activation= */ "sigmoid")
                    ->addPredecessor(hidden);

  _model = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ std::vector<bolt::InputPtr>{input},
      /* output= */ output);
  _model->compile(std::make_shared<bolt::BinaryCrossEntropyLoss>());
}

float TextClassifier::trainOnBatch(const py::dict& data,
                                   NumpyArray<float>& labels,
                                   float learning_rate) {
  auto bolt_input = featurize(data);
  BoltBatch bolt_labels =
      convertLabelsToBoltBatch(labels, bolt_input[0].getBatchSize());

  bolt::MetricAggregator metrics({});
  _model->trainOnBatch(std::move(bolt_input), bolt_labels, learning_rate,
                       metrics,
                       /* rebuild_hash_tables_interval= */ 25,
                       /* reconstruct_hash_functions_interval= */
                       100);

  auto mean_loss = binaryCrossEntropyLoss(labels).first;
  return mean_loss;
}

py::dict TextClassifier::validateOnBatch(const py::dict& data,
                                         NumpyArray<float>& labels) {
  auto bolt_input = featurize(data);

  verifyArrayHasNDimensions(labels, /* ndim= */ 2, /* name= */ "labels");
  verifyArrayShape(
      labels, /* expected_shape= */ {bolt_input[0].getBatchSize(), _n_classes},
      /* name= */ "labels");

  _model->predictSingleBatch(std::move(bolt_input),
                             /* use_sparse_inference= */ false);

  auto [mean_loss, per_class_loss] = binaryCrossEntropyLoss(labels);

  py::dict output;
  output["mean_loss"] = mean_loss;
  output["per_class_loss"] = per_class_loss;

  return output;
}

py::array_t<float, py::array::c_style> TextClassifier::predict(
    const py::dict& data) {
  auto bolt_input = featurize(data);

  uint32_t n_samples = bolt_input[0].getBatchSize();

  BoltBatch outputs =
      _model->predictSingleBatch(std::move(bolt_input),
                                 /* use_sparse_inference= */ false);

  uint32_t output_dim = _n_classes;
  py::array_t<float, py::array::c_style> output(
      /* shape= */ {n_samples, output_dim});

  for (uint32_t i = 0; i < n_samples; i++) {
    float* activations = outputs[i].activations;
    std::copy(activations, activations + output_dim, output.mutable_data(i));
  }

  return output;
}

void TextClassifier::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<TextClassifier> TextClassifier::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<TextClassifier> deserialize_into(new TextClassifier());
  iarchive(*deserialize_into);

  return deserialize_into;
}

std::vector<BoltBatch> TextClassifier::featurize(const py::dict& data) const {
  NumpyArray<uint32_t> tokens = data["tokens"].cast<NumpyArray<uint32_t>>();
  verifyArrayHasNDimensions(tokens, /* ndim= */ 1, /* name= */ "tokens");

  NumpyArray<uint32_t> metadata = data["metadata"].cast<NumpyArray<uint32_t>>();
  verifyArrayHasNDimensions(metadata, /* ndim= */ 2, /* name= */ "metadata");
  uint32_t batch_size = metadata.shape(0);
  verifyArrayShape(metadata, /* expected_shape= */ {batch_size, _metadata_dim},
                   /* name= */ "metadata");

  NumpyArray<uint32_t> offsets = data["offsets"].cast<NumpyArray<uint32_t>>();
  verifyArrayHasNDimensions(offsets, /* ndim= */ 1, /* name= */ "offsets");
  verifyArrayShape(offsets, /* expected_shape= */ {batch_size + 1},
                   /* name= */ "offsets");
  verifyOffsets(offsets, /* num_tokens= */ tokens.shape(0));

  std::vector<BoltVector> vectors(batch_size);

  // We use pointers here because py::array_t access methods throw exceptions
  // and can't be called within omp loops.
  const uint32_t* tokens_ptr = tokens.data();
  const uint32_t* offsets_ptr = offsets.data();
  const uint32_t* metadata_ptr = metadata.data();

#pragma omp parallel for default(none) \
    shared(batch_size, metadata_ptr, tokens_ptr, offsets_ptr, vectors)
  for (uint32_t i = 0; i < batch_size; i++) {
    std::vector<uint32_t> metadata_nonzeros =
        getMetadataNonzeros(metadata_ptr + i * _metadata_dim);

    uint32_t tokens_start = offsets_ptr[i];
    uint32_t n_tokens = offsets_ptr[i + 1] - tokens_start;
    vectors[i] = concatTokensAndMetadata(tokens_ptr + tokens_start, n_tokens,
                                         metadata_nonzeros);
  }

  std::vector<BoltBatch> output_batches;
  output_batches.emplace_back(std::move(vectors));
  return output_batches;
}

BoltBatch TextClassifier::convertLabelsToBoltBatch(NumpyArray<float>& labels,
                                                   uint32_t batch_size) const {
  verifyArrayHasNDimensions(labels, /* ndim= */ 2, /* name= */ "labels");
  verifyArrayShape(labels, /* expected_shape= */ {batch_size, _n_classes},
                   /* name= */ "labels");

  std::vector<BoltVector> label_vectors(batch_size);

  for (uint32_t i = 0; i < batch_size; i++) {
    label_vectors[i] =
        BoltVector(/* an= */ nullptr, /* a= */ labels.mutable_data(i),
                   /* g= */ nullptr, /* l= */ _n_classes);
  }

  return BoltBatch(std::move(label_vectors));
}

std::vector<uint32_t> TextClassifier::getMetadataNonzeros(
    const uint32_t* metadata) const {
  std::vector<uint32_t> metadata_nonzeros;
  try {  // This is to make sure that no exceptions get thrown in omp loops.
    for (uint32_t i = 0; i < _metadata_dim; i++) {
      if (metadata[i] != 0) {
        metadata_nonzeros.push_back(_input_vocab_size + i);
      }
    }
  } catch (std::exception& e) {
    return {};
  }
  return metadata_nonzeros;
}

BoltVector TextClassifier::concatTokensAndMetadata(
    const uint32_t* tokens, uint32_t n_tokens,
    const std::vector<uint32_t>& metadata_nonzeros) {
  BoltVector vector(/* l= */ n_tokens + metadata_nonzeros.size(),
                    /* is_dense= */ false, /* has_gradient= */ false);

  std::copy(tokens, tokens + n_tokens, vector.active_neurons);

  std::copy(metadata_nonzeros.begin(), metadata_nonzeros.end(),
            vector.active_neurons + n_tokens);

  std::fill_n(vector.activations, vector.len, 1.0);

  return vector;
}

std::pair<float, NumpyArray<float>> TextClassifier::binaryCrossEntropyLoss(
    const NumpyArray<float>& labels) {
  uint32_t batch_size = labels.shape(0);

  NumpyArray<float> per_class_loss(_n_classes);

  float loss = 0.0;

  // We use pointers here because py::array_t access methods throw exceptions
  // and can't be called within omp loops.
  float* per_class_loss_ptr = per_class_loss.mutable_data();
  const float* labels_ptr = labels.data();

// Reduction will sum up the loss values automatically between all the threads
// omp uses.
#pragma omp parallel for default(none) shared(batch_size, per_class_loss_ptr, labels_ptr) reduction(+ : loss)
  for (uint32_t class_id = 0; class_id < _n_classes; class_id++) {
    per_class_loss_ptr[class_id] = 0.0;

    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      float activation =
          _model->output()->getOutputVector(batch_idx).activations[class_id];

      activation = std::clamp(activation, 1e-6F, 1 - 1e-6F);

      float label = labels_ptr[batch_idx * _n_classes + class_id];

      float single_loss =
          label * std::log(activation) + (1 - label) * std::log(1 - activation);

      loss -= single_loss;

      per_class_loss_ptr[class_id] -= single_loss;
    }

    per_class_loss_ptr[class_id] /= batch_size;
  }

  return std::make_pair(loss / (batch_size * _n_classes),
                        std::move(per_class_loss));
}

bolt::FullyConnectedNodePtr TextClassifier::getHiddenLayer(
    const std::string& model_size) {
  if (model_size == "small") {
    return bolt::FullyConnectedNode::makeDense(/* dim= */ 512,
                                               /* activation= */ "relu");
  }

  if (model_size == "medium") {
    return bolt::FullyConnectedNode::make(
        /* dim= */ 1024, /* sparsity= */ 0.5, /* activation= */ "relu",
        /* sampling_config= */
        std::make_shared<bolt::RandomSamplingConfig>());
  }

  if (model_size == "large") {
    return bolt::FullyConnectedNode::make(
        /* dim= */ 2048, /* sparsity= */ 0.35, /* activation= */ "relu",
        /* sampling_config= */
        std::make_shared<bolt::RandomSamplingConfig>());
  }

  throw std::invalid_argument("Invalid model size '" + model_size +
                              "', expected 'small', 'medium', or 'large'.");
}

void TextClassifier::verifyArrayHasNDimensions(
    const NumpyArray<uint32_t>& array, uint32_t ndim, const std::string& name) {
  if (array.ndim() != ndim) {
    std::stringstream error;
    error << "Expected " << name << " to have " << ndim
          << " dimensions, but received array with " << array.ndim()
          << " dimensions.";
    throw std::invalid_argument(error.str());
  }
}

void TextClassifier::verifyArrayShape(const NumpyArray<uint32_t>& array,
                                      std::vector<uint32_t> expected_shape,
                                      const std::string& name) {
  if (expected_shape.empty()) {
    return;
  }

  for (uint32_t i = 0; i < expected_shape.size(); i++) {
    if (array.shape(i) != expected_shape.at(i)) {
      std::stringstream error;
      error << "Expected " << name << " to have shape (";
      std::copy(expected_shape.begin(), expected_shape.end(),
                std::ostream_iterator<uint32_t>(error, ", "));
      error << "), but received array with shape (";
      std::copy(array.shape(), array.shape() + array.ndim(),
                std::ostream_iterator<uint32_t>(error, ", "));
      error << ").";
      throw std::invalid_argument(error.str());
    }
  }
}

void TextClassifier::verifyOffsets(const NumpyArray<uint32_t>& offsets,
                                   uint32_t num_tokens) {
  for (uint32_t i = 0; i < offsets.shape(0) - 1; i++) {
    if (offsets.at(i) >= num_tokens) {
      throw std::invalid_argument("Invalid offset " +
                                  std::to_string(offsets.at(i)) +
                                  " for CSR tokens array of length " +
                                  std::to_string(num_tokens) + ".");
    }
  }
  if (offsets.at(offsets.shape(0) - 1) != num_tokens) {
    throw std::invalid_argument(
        "The last offset should be the number of tokens + 1.");
  }
}

template <class Archive>
void TextClassifier::serialize(Archive& archive) {
  archive(_model, _input_vocab_size, _metadata_dim, _n_classes);
}

}  // namespace thirdai::automl::models