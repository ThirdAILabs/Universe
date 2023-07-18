#include "DataPython.h"
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TabularHashedFeatures.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::data::python {

namespace py = pybind11;

void createColumnsSubmodule(py::module_& dataset_submodule);

void createTransformationsSubmodule(py::module_& dataset_submodule);

void createDataSubmodule(py::module_& dataset_submodule) {
  py::class_<ColumnMap>(dataset_submodule, "ColumnMap")
      .def(py::init<std::unordered_map<std::string, ColumnPtr>>(),
           py::arg("columns"))
      .def("num_rows", &ColumnMap::numRows)
      .def("__getitem__", &ColumnMap::getColumn)
      .def("columns", &ColumnMap::columns);

  createColumnsSubmodule(dataset_submodule);

  createTransformationsSubmodule(dataset_submodule);
}

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T, class COL>
NumpyArray<T> getRowNumpy(const std::shared_ptr<COL>& column, size_t n) {
  RowView<T> row = column->row(n);
  return NumpyArray<T>(row.size(), row.data());
}

template <typename T>
std::vector<T> fromNumpy1D(const NumpyArray<T>& array) {
  if (array.ndim() != 1) {
    throw std::invalid_argument("Expected 1D array when creating ValueColumn.");
  }
  return std::vector<T>(array.data(), array.data() + array.size());
}

template <typename T>
std::vector<std::vector<T>> fromNumpy2D(const NumpyArray<T>& array) {
  if (array.ndim() != 2) {
    throw std::invalid_argument("Expected 2D array when creating ArrayColumn.");
  }

  size_t rows = array.shape(0), cols = array.shape(1);

  std::vector<std::vector<T>> vector(array.shape(0));
  for (size_t i = 0; i < rows; i++) {
    vector[i] = std::vector<T>(array.data(i), array.data(i) + cols);
  }

  return vector;
}

auto tokenColumnFromNumpy(const NumpyArray<uint32_t>& array,
                          std::optional<size_t> dim) {
  return TokenColumn::make(fromNumpy1D(array), dim);
}

auto decimalColumnFromNumpy(const NumpyArray<float>& array) {
  return DecimalColumn::make(fromNumpy1D(array));
}

auto tokenArrayColumnFromNumpy(const NumpyArray<uint32_t>& array,
                               std::optional<size_t> dim) {
  return TokenArrayColumn::make(fromNumpy2D(array), dim);
}

auto decimalArrayColumnFromNumpy(const NumpyArray<float>& array) {
  return DecimalArrayColumn::make(fromNumpy2D(array));
}

void createColumnsSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column")
      .def("dimension", &Column::dimension)
      .def("__len__", &Column::numRows);

  py::class_<ColumnDimension>(columns_submodule, "Dimension")
      .def_readonly("dim", &ColumnDimension::dim)
      .def_readonly("is_dense", &ColumnDimension::is_dense);

  py::class_<TokenColumn, Column, std::shared_ptr<TokenColumn>>(
      columns_submodule, "TokenColumn")
      .def(py::init(&TokenColumn::make), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(py::init(&tokenColumnFromNumpy), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(("__getitem__"), &TokenColumn::value)
      .def("data", &TokenColumn::data);

  py::class_<DecimalColumn, Column, std::shared_ptr<DecimalColumn>>(
      columns_submodule, "DecimalColumn")
      .def(py::init(&DecimalColumn::make), py::arg("data"))
      .def(py::init(&decimalColumnFromNumpy), py::arg("data"))
      .def(("__getitem__"), &DecimalColumn::value)
      .def("data", &DecimalColumn::data);

  py::class_<StringColumn, Column, std::shared_ptr<StringColumn>>(
      columns_submodule, "StringColumn")
      .def(py::init(&StringColumn::make), py::arg("data"))
      .def("__getitem__", &StringColumn::value)
      .def("data", &StringColumn::data);

  py::class_<TimestampColumn, Column, std::shared_ptr<TimestampColumn>>(
      columns_submodule, "TimestampColumn")
      .def(py::init(&TimestampColumn::make), py::arg("data"))
      .def("__getitem__", &TimestampColumn::value)
      .def("data", &TimestampColumn::data);

  py::class_<TokenArrayColumn, Column, std::shared_ptr<TokenArrayColumn>>(
      columns_submodule, "TokenArrayColumn")
      .def(py::init(&TokenArrayColumn::make), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(py::init(&tokenArrayColumnFromNumpy), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def("__getitem__", &getRowNumpy<uint32_t, TokenArrayColumn>,
           py::return_value_policy::reference_internal)
      .def("data", &TokenArrayColumn::data);

  py::class_<DecimalArrayColumn, Column, std::shared_ptr<DecimalArrayColumn>>(
      columns_submodule, "DecimalArrayColumn")
      .def(py::init(&DecimalArrayColumn::make), py::arg("data"))
      .def(py::init(&decimalArrayColumnFromNumpy), py::arg("data"))
      .def("__getitem__", &getRowNumpy<float, DecimalArrayColumn>,
           py::return_value_policy::reference_internal)
      .def("data", &DecimalArrayColumn::data);
}

void createTransformationsSubmodule(py::module_& dataset_submodule) {
  auto transformations_submodule =
      dataset_submodule.def_submodule("transformations");

  py::class_<Transformation, std::shared_ptr<Transformation>>(
      transformations_submodule, "Transformation")
      .def("__call__", &Transformation::apply, py::arg("columns"));

  py::class_<TextTokenizer, Transformation, std::shared_ptr<TextTokenizer>>(
      transformations_submodule, "Text")
      .def(py::init<std::string, std::string, dataset::TextTokenizerPtr,
                    dataset::TextEncoderPtr, bool, size_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("tokenizer") = dataset::NaiveSplitTokenizer::make(),
           py::arg("encoder") = dataset::NGramEncoder(1),
           py::arg("lowercase") = false,
           py::arg("dim") = dataset::token_encoding::DEFAULT_TEXT_ENCODING_DIM);

  py::class_<BinningTransformation, Transformation,
             std::shared_ptr<BinningTransformation>>(transformations_submodule,
                                                     "Binning")
      .def(py::init<std::string, std::string, float, float, uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("inclusive_min"), py::arg("exclusive_max"),
           py::arg("num_bins"));

  py::class_<StringHash, Transformation, std::shared_ptr<StringHash>>(
      transformations_submodule, "StringHash")
      .def(py::init<std::string, std::string, std::optional<uint32_t>,
                    uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("output_range") = std::nullopt, py::arg("seed") = 42);

  py::class_<TabularHashedFeatures, Transformation,
             std::shared_ptr<TabularHashedFeatures>>(transformations_submodule,
                                                     "TabularHashedFeatures")
      .def(py::init<std::vector<std::string>, std::string, uint32_t, bool>(),
           py::arg("input_columns"), py::arg("output_column"),
           py::arg("output_range"), py::arg("use_pairgrams") = false);

  py::class_<TransformationList, Transformation,
             std::shared_ptr<TransformationList>>(transformations_submodule,
                                                  "TransformationList")
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"));

#if THIRDAI_EXPOSE_ALL
  py::class_<ColdStartTextAugmentation, Transformation,
             std::shared_ptr<ColdStartTextAugmentation>>(
      transformations_submodule, "ColdStartText")
      .def(py::init([](std::vector<std::string> strong_column_names,
                       std::vector<std::string> weak_column_names,
                       std::string label_column_name,
                       std::string output_column_name,
                       std::optional<uint32_t> weak_min_len,
                       std::optional<uint32_t> weak_max_len,
                       std::optional<uint32_t> weak_chunk_len,
                       std::optional<uint32_t> weak_sample_num_words,
                       uint32_t weak_sample_reps,
                       std::optional<uint32_t> strong_max_len,
                       std::optional<uint32_t> strong_sample_num_words,
                       uint32_t seed) {
             return std::make_shared<ColdStartTextAugmentation>(
                 std::move(strong_column_names), std::move(weak_column_names),
                 std::move(label_column_name), std::move(output_column_name),
                 ColdStartConfig(weak_min_len, weak_max_len, weak_chunk_len,
                                 weak_sample_num_words, weak_sample_reps,
                                 strong_max_len, strong_sample_num_words),
                 seed);
           }),
           py::arg("strong_columns"), py::arg("weak_columns"),
           py::arg("label_column"), py::arg("output_column"),
           py::arg("weak_min_len") = std::nullopt,
           py::arg("weak_max_len") = std::nullopt,
           py::arg("weak_chunk_len") = std::nullopt,
           py::arg("weak_sample_num_words") = std::nullopt,
           py::arg("weak_sample_reps") = 1,
           py::arg("strong_max_len") = std::nullopt,
           py::arg("strong_sample_num_words") = std::nullopt,
           py::arg("seed") = 42803)
      .def("augment_single_row", &ColdStartTextAugmentation::augmentSingleRow,
           py::arg("strong_text"), py::arg("weak_text"))
      .def("augment_map_input", &ColdStartTextAugmentation::augmentMapInput,
           py::arg("document"));
#endif
}

}  // namespace thirdai::data::python