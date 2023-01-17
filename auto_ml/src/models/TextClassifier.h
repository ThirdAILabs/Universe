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

class TextClassifier {
 public:
  TextClassifier(uint32_t input_vocab_size, uint32_t metadata_dim,
                 uint32_t n_classes, const std::string& model_size);

  float trainOnBatch(const py::dict& data, NumpyArray<float>& labels,
                     float learning_rate);

  py::dict validateOnBatch(const py::dict& data, NumpyArray<float>& labels);

  py::array_t<float, py::array::c_style> predict(const py::dict& data);

  void save_stream(std::ostream& output_stream) const;

  static std::shared_ptr<TextClassifier> load_stream(
      std::istream& input_stream);

 private:
  std::vector<BoltBatch> featurize(const py::dict& data) const;

  BoltBatch convertLabelsToBoltBatch(NumpyArray<float>& labels,
                                     uint32_t batch_size) const;

  std::vector<uint32_t> getMetadataNonzeros(const float* metadata) const;

  static BoltVector concatTokensAndMetadata(
      const uint32_t* tokens, uint32_t n_tokens,
      const std::vector<uint32_t>& metadata_nonzeros);

  std::pair<float, std::optional<NumpyArray<float>>> binaryCrossEntropyLoss(
      const NumpyArray<float>& labels, bool return_loss_per_class);

  template <typename T>
  static void verifyArrayHasNDimensions(const NumpyArray<T>& array, uint32_t n,
                                        const std::string& name);

  template <typename T>
  static void verifyArrayShape(const NumpyArray<T>& array,
                               std::vector<uint32_t> expected_dims,
                               const std::string& name);

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