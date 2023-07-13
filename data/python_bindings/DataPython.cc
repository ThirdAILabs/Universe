#include "DataPython.h"
#include "DataDocs.h"
#include <data/src/columns/NumpyColumns.h>
#include <data/src/columns/VectorColumns.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/ColdStartText.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/TabularHashedFeatures.h>
#include <data/src/transformations/Text.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/TransformationList.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <pybind11/stl.h>
#include <optional>
#include <string>

namespace thirdai::data::python {

namespace py = pybind11;

void createDataSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column", docs::COLUMN_BASE)
      .def("dimension_info", &Column::dimension);

  py::class_<DimensionInfo>(columns_submodule, "DimensionInfo",
                            docs::DIMENSION_INFO)
      .def_readonly("dim", &DimensionInfo::dim)
      .def_readonly("is_dense", &DimensionInfo::is_dense);

  py::class_<PyTokenColumn, Column, std::shared_ptr<PyTokenColumn>>(
      columns_submodule, "TokenColumn")
      .def(py::init<const NumpyArray<uint32_t>&, std::optional<uint32_t>>(),
           py::arg("array"), py::arg("dim"), docs::TOKEN_COLUMN)
      .def(("__getitem__"), &PyTokenColumn::at);

  py::class_<PyDenseFeatureColumn, Column,
             std::shared_ptr<PyDenseFeatureColumn>>(columns_submodule,
                                                    "DenseFeatureColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"),
           docs::DENSE_FEATURE_COLUMN)
      .def(("__getitem__"), &PyDenseFeatureColumn::at);

  py::class_<CppStringColumn, Column, std::shared_ptr<CppStringColumn>>(
      columns_submodule, "StringColumn")
      .def(py::init<std::vector<std::string>>(), py::arg("values"),
           docs::STRING_COLUMN)
      .def("__getitem__", &CppStringColumn::at);

  py::class_<PyTokenArrayColumn, Column, std::shared_ptr<PyTokenArrayColumn>>(
      columns_submodule, "TokenArrayColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"), docs::TOKEN_ARRAY_COLUMN);

  py::class_<PyDenseArrayColumn, Column, std::shared_ptr<PyDenseArrayColumn>>(
      columns_submodule, "DenseArrayColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"),
           docs::DENSE_ARRAY_COLUMN);

  auto transformations_submodule =
      dataset_submodule.def_submodule("transformations");

  py::class_<Transformation, std::shared_ptr<Transformation>>(
      transformations_submodule, "Transformation", docs::TRANSFORMATION_BASE)
      .def("__call__", &Transformation::apply, py::arg("columns"));

  py::class_<Text, Transformation, std::shared_ptr<Text>>(
      transformations_submodule, "Text")
      .def(py::init<std::string, std::string, TextTokenizerPtr, TextEncoderPtr,
                    bool, size_t>(),
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
           py::arg("num_bins"), docs::BINNING);

  py::class_<StringHash, Transformation, std::shared_ptr<StringHash>>(
      transformations_submodule, "StringHash")
      .def(py::init<std::string, std::string, std::optional<uint32_t>,
                    uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("output_range") = std::nullopt, py::arg("seed") = 42,
           docs::STRING_HASH);

  py::class_<TabularHashedFeatures, Transformation,
             std::shared_ptr<TabularHashedFeatures>>(transformations_submodule,
                                                     "TabularHashedFeatures")
      .def(py::init<std::vector<std::string>, std::string, uint32_t, bool>(),
           py::arg("input_columns"), py::arg("output_column"),
           py::arg("output_range"), py::arg("use_pairgrams") = false,
           docs::COLUMN_PAIRGRAM);

  py::class_<TransformationList, Transformation,
             std::shared_ptr<TransformationList>>(transformations_submodule,
                                                  "TransformationList",
                                                  docs::TRANSFORMATION_LIST)
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"), docs::TRANSFORMATION_LIST_INIT);

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

  py::class_<ColumnMap>(dataset_submodule, "ColumnMap", docs::COLUMN_MAP_CLASS)
      .def(py::init<std::unordered_map<std::string, ColumnPtr>>(),
           py::arg("columns"), docs::COLUMN_MAP_INIT)
      .def("convert_to_dataset", &ColumnMap::convertToDataset,
           py::arg("columns"), py::arg("batch_size"),
           docs::COLUMN_MAP_TO_DATASET)
      .def("num_rows", &ColumnMap::numRows)
      .def("__getitem__", &ColumnMap::getColumn)
      .def("columns", &ColumnMap::columns);
}

}  // namespace thirdai::data::python