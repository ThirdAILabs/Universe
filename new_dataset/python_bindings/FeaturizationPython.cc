#include "FeaturizationPython.h"
#include <bolt/python_bindings/ConversionUtils.h>
#include <new_dataset/src/featurization_pipeline/Augmentation.h>
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/columns/NumpyColumns.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <new_dataset/src/featurization_pipeline/transformations/Binning.h>
#include <new_dataset/src/featurization_pipeline/transformations/CrossColumnPairgram.h>
#include <new_dataset/src/featurization_pipeline/transformations/SentenceUnigram.h>
#include <new_dataset/src/featurization_pipeline/transformations/StringHash.h>
#include <new_dataset/src/featurization_pipeline/transformations/TokenPairgram.h>
#include <pybind11/stl.h>
#include <optional>
#include <string>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createFeaturizationSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column")
      .def("dimension_info", &Column::dimension);

  py::class_<DimensionInfo>(columns_submodule, "DimensionInfo")
      .def_readonly("dim", &DimensionInfo::dim)
      .def_readonly("is_dense", &DimensionInfo::is_dense);

  py::class_<NumpySparseValueColumn, Column,
             std::shared_ptr<NumpySparseValueColumn>>(columns_submodule,
                                                      "NumpySparseValueColumn")
      .def(py::init<const NumpyArray<uint32_t>&, std::optional<uint32_t>>(),
           py::arg("array"), py::arg("dim"))
      .def(("__getitem__"), &NumpySparseValueColumn::operator[]);

  py::class_<NumpyDenseValueColumn, Column,
             std::shared_ptr<NumpyDenseValueColumn>>(columns_submodule,
                                                     "NumpyDenseValueColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"))
      .def(("__getitem__"), &NumpyDenseValueColumn::operator[]);

  py::class_<VectorStringValueColumn, Column,
             std::shared_ptr<VectorStringValueColumn>>(columns_submodule,
                                                       "StringColumn")
      .def(py::init<std::vector<std::string>>(), py::arg("array"));

  py::class_<NumpySparseArrayColumn, Column,
             std::shared_ptr<NumpySparseArrayColumn>>(columns_submodule,
                                                      "NumpySparseArrayColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"));

  py::class_<NumpyDenseArrayColumn, Column,
             std::shared_ptr<NumpyDenseArrayColumn>>(columns_submodule,
                                                     "NumpyDenseArrayColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"));

  auto augmentations_submodule =
      dataset_submodule.def_submodule("augmentations");

  py::class_<Augmentation, std::shared_ptr<Augmentation>>(  // NOLINT
      augmentations_submodule, "Augmentation")
      .def(("apply"), &Augmentation::apply);

  py::class_<ColdStartTextAugmentation, Augmentation,
             std::shared_ptr<ColdStartTextAugmentation>>(augmentations_submodule,
                                                     "ColdStartText")
      .def(py::init<std::vector<std::string>, std::vector<std::string>,
                    std::string, std::string, std::optional<uint32_t>,
                    std::optional<uint32_t>, std::optional<uint32_t>,
                    std::optional<uint32_t>, uint32_t, std::optional<uint32_t>,
                    std::optional<uint32_t>, uint32_t>(),
           py::arg("strong_columns"), py::arg("weak_columns"),
           py::arg("label_column"),
           py::arg("output_column"),
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
      transformations_submodule, "Transformation");

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

  py::class_<CrossColumnPairgram, Transformation,
             std::shared_ptr<CrossColumnPairgram>>(transformations_submodule,
                                                   "CrossColumnPairgram")
      .def(py::init<std::vector<std::string>, std::string, uint32_t>(),
           py::arg("input_columns"), py::arg("output_column"),
           py::arg("output_range"));

  py::class_<SentenceUnigram, Transformation, std::shared_ptr<SentenceUnigram>>(
      transformations_submodule, "SentenceUnigram")
      .def(py::init<std::string, std::string, bool, std::optional<uint32_t>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("deduplicate") = false,
           py::arg("output_range") = std::nullopt);

  py::class_<TokenPairgram, Transformation, std::shared_ptr<TokenPairgram>>(
      transformations_submodule, "TokenPairgram")
      .def(py::init<std::string, std::string, uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("output_range"));

  py::class_<ColumnMap>(dataset_submodule, "ColumnMap")
      .def(py::init<std::unordered_map<std::string, ColumnPtr>>(),
           py::arg("columns"))
      .def("convert_to_dataset", &ColumnMap::convertToDataset,
           py::arg("columns"), py::arg("batch_size"))
      .def("num_rows", &ColumnMap::numRows)
      .def("__getitem__", &ColumnMap::getColumn)
      .def("columns", &ColumnMap::columns);

  py::class_<FeaturizationPipeline, FeaturizationPipelinePtr>(
      dataset_submodule, "FeaturizationPipeline")
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"))
      .def("featurize", &FeaturizationPipeline::featurize, py::arg("columns"))
      .def(bolt::python::getPickleFunction<FeaturizationPipeline>());
}

}  // namespace thirdai::dataset::python