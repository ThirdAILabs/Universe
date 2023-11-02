#include "SeismicEmbedding.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/tuple.hpp>
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/ExternalLoss.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/PatchEmbedding.h>
#include <bolt/src/nn/ops/PatchSum.h>
#include <bolt/src/seismic/SeismicLabels.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt/src/utils/Timer.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <optional>
#include <stdexcept>
#include <utility>

namespace thirdai::bolt::seismic {

std::pair<size_t, float> nOutputClasses(const std::string& model_size) {
  if (model_size == "small") {
    return {20000, 0.1};
  }
  if (model_size == "medium") {
    return {50000, 0.05};
  }
  if (model_size == "large") {
    return {100000, 0.025};
  }
  throw std::invalid_argument(
      "Invalid model size. Please use 'small', 'medium', or 'large.");
}

SeismicEmbedding::SeismicEmbedding(InputShapeData input_shape_data,
                                   ModelPtr model)
    : SeismicBase(input_shape_data, std::move(model)),
      _training_type(TrainingType::UnsupervisedPretraining) {
  if (getModel()->labelDims().size() != 1) {
    throw std::invalid_argument("Expected model to only have 1 output layer.");
  }
}

std::shared_ptr<SeismicEmbedding> SeismicEmbedding::make(
    size_t subcube_shape, size_t patch_shape, size_t embedding_dim,
    const std::string& model_size, std::optional<size_t> max_pool) {
  InputShapeData input_shape_data(subcube_shape, patch_shape, max_pool);

  auto [n_output_classes, output_sparsity] = nOutputClasses(model_size);

  auto model = buildModel(input_shape_data.nPatches(),
                          input_shape_data.flattenedPatchDim(), embedding_dim,
                          n_output_classes, output_sparsity);

  return std::make_shared<SeismicEmbedding>(input_shape_data, model);
}

metrics::History SeismicEmbedding::trainOnPatches(
    const NumpyArray& subcubes,
    const std::vector<SubcubeMetadata>& subcube_metadata, float learning_rate,
    size_t batch_size, const std::vector<callbacks::CallbackPtr>& callbacks,
    std::optional<uint32_t> log_interval, const DistributedCommPtr& comm) {
  if (_training_type != TrainingType::UnsupervisedPretraining) {
    throw std::invalid_argument(
        "Can not use unsupervised pretraining on a model after using "
        "finetuning since the decoder has been invalidated.");
  }

  if (static_cast<size_t>(subcubes.shape(0)) != subcube_metadata.size()) {
    throw std::invalid_argument(
        "Expected number of subcubes to match the number of subcube "
        "metadatas.");
  }

  auto labels = makeLabelBatches(subcube_metadata, batch_size);

  return SeismicBase::trainOnPatches(subcubes, std::move(labels), learning_rate,
                                     batch_size, callbacks, log_interval, comm);
}

py::object SeismicEmbedding::forward(const NumpyArray& subcubes) {
  switchToFinetuning();

  auto batch = convertToBatches(subcubes, subcubes.shape(0)).at(0);

  auto emb = _model->forward(batch, /* use_sparsity= */ true).at(0);

  return python::tensorToNumpy(emb, /* single_row_to_vector= */ false);
}

void SeismicEmbedding::backpropagate(const NumpyArray& gradients) {
  switchToFinetuning();

  auto grad_tensor = python::fromNumpyDense(gradients, /* with_grad= */ false);

  _model->backpropagate({grad_tensor});
}

void SeismicEmbedding::updateParameters(float learning_rate) {
  _model->updateParameters(learning_rate);
}

void SeismicEmbedding::switchToFinetuning() {
  if (_training_type == TrainingType::Finetuning) {
    return;
  }

  auto patches = Input::make(_model->inputDims().at(0));

  auto patch_emb_op =
      std::dynamic_pointer_cast<PatchEmbedding>(_model->getOp("patch_emb"));
  auto patch_emb = patch_emb_op->apply(patches);

  auto patch_sum_op =
      std::dynamic_pointer_cast<PatchSum>(_model->getOp("patch_sum"));
  auto patch_sum = patch_sum_op->apply(patch_emb);

  auto emb_op = std::dynamic_pointer_cast<FullyConnected>(_model->getOp("emb"));
  auto emb = emb_op->apply(patch_sum);

  auto loss = std::make_shared<ExternalLoss>(emb, Input::make(emb->dim()));

  auto model = Model::make({patches}, {emb}, {loss});

  _training_type = TrainingType::Finetuning;
  setModel(model);
}

void SeismicEmbedding::setModel(ModelPtr model) {
  _model = std::move(model);
  auto computations = _model->computationOrderWithoutInputs();
  size_t emb_pos = _training_type == TrainingType::Finetuning ? 1 : 2;
  auto new_emb = computations.at(computations.size() - emb_pos);
  if (_emb && _emb->dim() != new_emb->dim()) {
    throw std::runtime_error("Cannot set a model with embedding dimension " +
                             std::to_string(new_emb->dim()) +
                             " in place of a model with embedding dimension " +
                             std::to_string(_emb->dim()));
  }
  _emb = new_emb;
}

Dataset SeismicEmbedding::makeLabelBatches(
    const std::vector<SubcubeMetadata>& subcube_metadata,
    size_t batch_size) const {
  size_t n_batches = (subcube_metadata.size() + batch_size - 1) / batch_size;

  Dataset label_batches;

  size_t n_output_classes = labelDim();

  for (size_t batch = 0; batch < n_batches; batch++) {
    size_t start = batch * batch_size;
    size_t end = std::min(start + batch_size, subcube_metadata.size());

    std::vector<uint32_t> indices;
    std::vector<float> values;
    std::vector<size_t> lens;

    for (size_t i = start; i < end; i++) {
      auto labels =
          seismicLabelsFromMetadata(subcube_metadata[i], subcubeShape(),
                                    _label_cube_dim, n_output_classes);

      indices.insert(indices.end(), labels.begin(), labels.end());

      float label_val = 1.0 / labels.size();
      for (size_t j = 0; j < labels.size(); j++) {
        values.push_back(label_val);
      }

      lens.push_back(labels.size());
    }

    auto tensor = Tensor::sparse(std::move(indices), std::move(values),
                                 std::move(lens), n_output_classes);

    label_batches.push_back({tensor});
  }

  return label_batches;
}

ModelPtr SeismicEmbedding::buildModel(size_t n_patches, size_t patch_dim,
                                      size_t embedding_dim,
                                      size_t n_output_classes,
                                      float output_sparsity) {
  auto patches = Input::make(n_patches * patch_dim);

  size_t patch_emb_dim = 100000;

  // Create sparse embedding for each patch.
  auto patch_emb_op = PatchEmbedding::make(
      /*emb_dim=*/patch_emb_dim, /*patch_dim=*/patch_dim,
      /*n_patches=*/n_patches, /*sparsity=*/0.01, /*activation=*/"relu");
  patch_emb_op->setName("patch_emb");
  auto patch_emb = patch_emb_op->apply(patches);

  // Aggregate patch embeddings.
  auto patch_sum = PatchSum::make(/*n_patches=*/n_patches,
                                  /*patch_dim=*/patch_emb_dim);
  patch_sum->setName("patch_sum");
  auto aggregated_embs = patch_sum->apply(patch_emb);

  // Map to final embedding for subcube.
  auto emb_op = FullyConnected::make(
      /*dim=*/embedding_dim, /*input_dim=*/aggregated_embs->dim(),
      /* sparsity=*/1.0, /*activation=*/"tanh");
  emb_op->setName("emb");
  auto emb = emb_op->apply(aggregated_embs);

  // Output decoder head for training.
  auto output = FullyConnected::make(
                    /*dim=*/n_output_classes, /*input_dim=*/emb->dim(),
                    /*sparsity=*/output_sparsity,
                    /*activation=*/"softmax")
                    ->apply(emb);

  auto loss =
      CategoricalCrossEntropy::make(output, Input::make(n_output_classes));

  return Model::make({patches}, {output}, {loss});
}

void SeismicEmbedding::save(const std::string& filename) const {
  auto output = dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output);
}

void SeismicEmbedding::save_stream(std::ostream& output) const {
  cereal::BinaryOutputArchive oarchive(output);
  getModel()->setSerializeOptimizer(/* should_save_optimizer= */ false);
  oarchive(*this);
}

std::shared_ptr<SeismicEmbedding> SeismicEmbedding::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(input_stream);
}

std::shared_ptr<SeismicEmbedding> SeismicEmbedding::load_stream(
    std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<SeismicEmbedding> deserialize_into(new SeismicEmbedding());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template void SeismicEmbedding::serialize(cereal::BinaryInputArchive&);
template void SeismicEmbedding::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void SeismicEmbedding::serialize(Archive& archive) {
  archive(cereal::base_class<SeismicBase>(this), _label_cube_dim,
          _training_type);
}

}  // namespace thirdai::bolt::seismic

CEREAL_REGISTER_TYPE(thirdai::bolt::seismic::SeismicEmbedding)