#include "FeaturizationPython.h"
#include "FeaturizationDocs.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <new_dataset/src/featurization_pipeline/Augmentation.h>
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/columns/NumpyColumns.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <new_dataset/src/featurization_pipeline/transformations/Binning.h>
#include <new_dataset/src/featurization_pipeline/transformations/SentenceUnigram.h>
#include <new_dataset/src/featurization_pipeline/transformations/StringHash.h>
#include <new_dataset/src/featurization_pipeline/transformations/TabularHashedFeatures.h>
#include <new_dataset/src/featurization_pipeline/transformations/TokenPairgram.h>
#include <pybind11/stl.h>
#include <optional>
#include <string>

namespace thirdai::data::python {

namespace py = pybind11;

void createFeaturizationSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<columns::Column, columns::ColumnPtr>(columns_submodule, "Column",
                                                  docs::COLUMN_BASE)
      .def("dimension_info", &columns::Column::dimension);

  py::class_<columns::DimensionInfo>(columns_submodule, "DimensionInfo",
                                     docs::DIMENSION_INFO)
      .def_readonly("dim", &columns::DimensionInfo::dim)
      .def_readonly("is_dense", &columns::DimensionInfo::is_dense);

  py::class_<columns::PyTokenColumn, columns::Column,
             std::shared_ptr<columns::PyTokenColumn>>(columns_submodule,
                                                      "TokenColumn")
      .def(py::init<const columns::NumpyArray<uint32_t>&,
                    std::optional<uint32_t>>(),
           py::arg("array"), py::arg("dim"), docs::TOKEN_COLUMN)
      .def(("__getitem__"), &columns::PyTokenColumn::operator[]);

  py::class_<columns::PyDenseFeatureColumn, columns::Column,
             std::shared_ptr<columns::PyDenseFeatureColumn>>(
      columns_submodule, "DenseFeatureColumn")
      .def(py::init<const columns::NumpyArray<float>&>(), py::arg("array"),
           docs::DENSE_FEATURE_COLUMN)
      .def(("__getitem__"), &columns::PyDenseFeatureColumn::operator[]);

  py::class_<columns::CppStringColumn, columns::Column,
             std::shared_ptr<columns::CppStringColumn>>(columns_submodule,
                                                        "StringColumn")
      .def(py::init<std::vector<std::string>>(), py::arg("values"),
           docs::STRING_COLUMN);

  py::class_<columns::PyTokenArrayColumn, columns::Column,
             std::shared_ptr<columns::PyTokenArrayColumn>>(columns_submodule,
                                                           "TokenArrayColumn")
      .def(py::init<const columns::NumpyArray<uint32_t>&, uint32_t>(),
           py::arg("array"), py::arg("dim"), docs::TOKEN_ARRAY_COLUMN);

  py::class_<columns::PyDenseArrayColumn, columns::Column,
             std::shared_ptr<columns::PyDenseArrayColumn>>(columns_submodule,
                                                           "DenseArrayColumn")
      .def(py::init<const columns::NumpyArray<float>&>(), py::arg("array"),
           docs::DENSE_ARRAY_COLUMN);

  auto augmentations_submodule =
      dataset_submodule.def_submodule("augmentations");

  py::class_<Augmentation, std::shared_ptr<Augmentation>>(  // NOLINT
      augmentations_submodule, "Augmentation")
      .def("apply", &Augmentation::apply);

  py::class_<ColdStartTextAugmentation, Augmentation,
             std::shared_ptr<ColdStartTextAugmentation>>(
      augmentations_submodule, "ColdStartText")
      .def(py::init<std::vector<std::string>, std::vector<std::string>,
                    std::string, std::string, std::optional<uint32_t>,
                    std::optional<uint32_t>, std::optional<uint32_t>,
                    std::optional<uint32_t>, uint32_t, std::optional<uint32_t>,
                    std::optional<uint32_t>, uint32_t>(),
           py::arg("strong_columns"), py::arg("weak_columns"),
           py::arg("label_column"), py::arg("output_column"),
           py::arg("weak_min_len") = std::nullopt,
           py::arg("weak_max_len") = std::nullopt,
           py::arg("weak_chunk_len") = std::nullopt,
           py::arg("weak_sample_num_words") = std::nullopt,
           py::arg("weak_sample_reps") = 1,
           py::arg("strong_max_len") = std::nullopt,
           py::arg("strong_sample_num_words") = std::nullopt,
           py::arg("seed") = 42803);

  auto transformations_submodule =
      dataset_submodule.def_submodule("transformations");

  py::class_<Transformation, std::shared_ptr<Transformation>>(  // NOLINT
      transformations_submodule, "Transformation", docs::TRANSFORMATION_BASE);

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

  py::class_<SentenceUnigram, Transformation, std::shared_ptr<SentenceUnigram>>(
      transformations_submodule, "SentenceUnigram")
      .def(py::init<std::string, std::string, bool, std::optional<uint32_t>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("deduplicate") = false,
           py::arg("output_range") = std::nullopt, docs::SENTENCE_UNIGRAM);

  py::class_<TokenPairgram, Transformation, std::shared_ptr<TokenPairgram>>(
      transformations_submodule, "TokenPairgram")
      .def(py::init<std::string, std::string, uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("output_range"), docs::TOKEN_PAIRGRAM);

  py::class_<ColumnMap>(dataset_submodule, "ColumnMap", docs::COLUMN_MAP_CLASS)
      .def(py::init<std::unordered_map<std::string, columns::ColumnPtr>>(),
           py::arg("columns"), docs::COLUMN_MAP_INIT)
      .def("convert_to_dataset", &ColumnMap::convertToDataset,
           py::arg("columns"), py::arg("batch_size"),
           docs::COLUMN_MAP_TO_DATASET)
      .def("num_rows", &ColumnMap::numRows)
      .def("__getitem__", &ColumnMap::getColumn)
      .def("columns", &ColumnMap::columns);

  py::class_<FeaturizationPipeline, FeaturizationPipelinePtr>(
      dataset_submodule, "FeaturizationPipeline",
      docs::FEATURIZATION_PIPELINE_CLASS)
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"), docs::FEATURIZATION_PIPELINE_INIT)
      .def("featurize", &FeaturizationPipeline::featurize, py::arg("columns"),
           docs::FEATURIZATION_PIPELINE_FEATURIZE)
      .def(bolt::python::getPickleFunction<FeaturizationPipeline>());
}

}  // namespace thirdai::data::python