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
#include <sstream>
#include <stdexcept>

namespace thirdai::automl::models {

TextClassifier::TextClassifier(uint32_t input_vocab_size, uint32_t metadata_dim,
                               uint32_t n_classes,
                               const std::string& model_size)
    : _input_vocab_size(input_vocab_size),
      _metadata_dim(metadata_dim),
      _n_classes(n_classes) {
  auto input = bolt::Input::make(input_vocab_size + metadata_dim);

  bolt::FullyConnectedNodePtr hidden;
  if (utils::lower(model_size) == "small") {
    hidden = bolt::FullyConnectedNode::makeDense(/* dim= */ 512,
                                                 /* activation= */ "relu")
                 ->addPredecessor(input);
  } else if (utils::lower(model_size) == "medium") {
    hidden = bolt::FullyConnectedNode::make(
                 /* dim= */ 1024, /* sparsity= */ 0.5, /* activation= */ "relu",
                 /* sampling_config= */
                 std::make_shared<bolt::RandomSamplingConfig>())
                 ->addPredecessor(input);
  } else if (utils::lower(model_size) == "large") {
    hidden =
        bolt::FullyConnectedNode::make(
            /* dim= */ 2048, /* sparsity= */ 0.35, /* activation= */ "relu",
            /* sampling_config= */
            std::make_shared<bolt::RandomSamplingConfig>())
            ->addPredecessor(input);
  }

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

  auto loss =
      binaryCrossEntropyLoss(labels, /* return_loss_per_class= */ false);
  return loss.first;
}

py::dict TextClassifier::validateOnBatch(const py::dict& data,
                                         NumpyArray<float>& labels) {
  auto bolt_input = featurize(data);

  verifyArrayHasNDimensions(labels, 2, "labels");
  verifyArrayShape(labels, {bolt_input[0].getBatchSize(), _n_classes},
                   "labels");

  _model->predictSingleBatchNoReturn(std::move(bolt_input),
                                     /* use_sparse_inference= */ false);

  auto [mean_loss, per_class_loss] =
      binaryCrossEntropyLoss(labels, /* return_loss_per_class= */ true);

  py::dict output;
  output["mean_loss"] = mean_loss;
  output["per_class_loss"] = per_class_loss;

  return output;
}

py::array_t<float, py::array::c_style> TextClassifier::predict(
    const py::dict& data) {
  auto bolt_input = featurize(data);

  uint32_t n_samples = bolt_input[0].getBatchSize();

  _model->predictSingleBatchNoReturn(std::move(bolt_input),
                                     /* use_sparse_inference= */ false);

  uint32_t output_dim = _n_classes;
  py::array_t<float, py::array::c_style> output(
      /* shape= */ {n_samples, output_dim});

  for (uint32_t i = 0; i < n_samples; i++) {
    float* activations = _model->output()->getOutputVector(i).activations;
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
  NumpyArray<uint32_t> tokens =
      data["bert_tokens"].cast<NumpyArray<uint32_t>>();
  verifyArrayHasNDimensions(tokens, 1, "bert_tokens");

  NumpyArray<float> metadata = data["metadata"].cast<NumpyArray<float>>();
  verifyArrayHasNDimensions(metadata, 2, "metadata");
  uint32_t batch_size = metadata.shape(0);
  verifyArrayShape(metadata, {batch_size, _metadata_dim}, "metadata");

  NumpyArray<uint32_t> offsets =
      data["doc_offsets"].cast<NumpyArray<uint32_t>>();
  verifyArrayHasNDimensions(offsets, 1, "doc_offsets");
  verifyArrayShape(offsets, {batch_size + 1}, "doc_offsets");

  std::vector<BoltVector> vectors(batch_size);

#pragma omp parallel for default(none) \
    shared(batch_size, metadata, tokens, offsets, vectors)
  for (uint32_t i = 0; i < batch_size; i++) {
    std::vector<uint32_t> metadata_nonzeros;
    for (uint32_t j = 0; j < _metadata_dim; j++) {
      if (metadata.at(i, j) != 0) {
        metadata_nonzeros.push_back(_input_vocab_size + j);
      }
    }

    uint32_t n_tokens = offsets.at(i + 1) - offsets.at(i);
    vectors[i] = BoltVector(/* l= */ n_tokens + metadata_nonzeros.size(),
                            /* is_dense= */ false, /* has_gradient= */ false);

    const uint32_t* tokens_ptr = tokens.data(offsets.at(i));
    std::copy(tokens_ptr, tokens_ptr + n_tokens, vectors[i].active_neurons);

    std::copy(metadata_nonzeros.begin(), metadata_nonzeros.end(),
              vectors[i].active_neurons + n_tokens);

    std::fill_n(vectors[i].activations, vectors[i].len, 1.0);
  }

  std::vector<BoltBatch> output_batches;
  output_batches.emplace_back(std::move(vectors));
  return output_batches;
}

BoltBatch TextClassifier::convertLabelsToBoltBatch(NumpyArray<float>& labels,
                                                   uint32_t batch_size) const {
  verifyArrayHasNDimensions(labels, 2, "labels");
  verifyArrayShape(labels, {batch_size, _n_classes}, "labels");

  std::vector<BoltVector> label_vectors(batch_size);

  for (uint32_t i = 0; i < batch_size; i++) {
    label_vectors[i] =
        BoltVector(/* an= */ nullptr, /* a= */ labels.mutable_data(i),
                   /* g= */ nullptr, /* l= */ _n_classes);
  }

  return BoltBatch(std::move(label_vectors));
}

std::pair<float, std::optional<NumpyArray<float>>>
TextClassifier::binaryCrossEntropyLoss(const NumpyArray<float>& labels,
                                       bool return_loss_per_class) {
  uint32_t batch_size = labels.shape(0);

  std::optional<NumpyArray<float>> per_class_loss;
  if (return_loss_per_class) {
    per_class_loss = NumpyArray<float>(_n_classes);
  }

  float loss = 0.0;

#pragma omp parallel for default(none) shared(batch_size, per_class_loss, labels) reduction(+ : loss)
  for (uint32_t class_id = 0; class_id < _n_classes; class_id++) {
    if (per_class_loss) {
      per_class_loss->mutable_at(class_id) = 0.0;
    }
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      float activation =
          _model->output()->getOutputVector(batch_idx).activations[class_id];
      float label = labels.at(batch_idx, class_id);

      float single_loss =
          label == 1.0 ? std::log(activation) : std::log(1 - activation);

      loss += single_loss;

      if (per_class_loss) {
        per_class_loss->mutable_at(class_id) += single_loss;
      }
    }

    if (per_class_loss) {
      per_class_loss->mutable_at(class_id) /= batch_size;
    }
  }

  return std::make_pair(loss / batch_size, std::move(per_class_loss));
}

template <typename T>
void TextClassifier::verifyArrayHasNDimensions(const NumpyArray<T>& array,
                                               uint32_t n,
                                               const std::string& name) {
  if (array.ndim() != n) {
    std::stringstream error;
    error << "Expected " << name << " to have " << n
          << " dimensions, but received array with " << array.ndim()
          << " dimensions.";
    throw std::invalid_argument(error.str());
  }
}

template <typename T>
void TextClassifier::verifyArrayShape(const NumpyArray<T>& array,
                                      std::vector<uint32_t> expected_dims,
                                      const std::string& name) {
  if (expected_dims.empty()) {
    return;
  }

  std::stringstream error;
  error << "Expected " << name << " to have shape (" << expected_dims.at(0);
  for (uint32_t i = 1; i < expected_dims.size(); i++) {
    error << ", " << expected_dims.at(i);
  }
  error << "), but recieved array with shape (" << array.shape(0);
  for (uint32_t i = 1; i < expected_dims.size(); i++) {
    error << ", " << array.shape(i);
  }
  error << ").";
  throw std::invalid_argument(error.str());
}

template <class Archive>
void TextClassifier::serialize(Archive& archive) {
  archive(_model, _input_vocab_size, _metadata_dim, _n_classes);
}

}  // namespace thirdai::automl::models