#include "FeaturizationPython.h"
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/Transformation.h>
#include <new_dataset/src/featurization_pipeline/columns/NumpyColumns.h>
#include <new_dataset/src/featurization_pipeline/transformations/Binning.h>
#include <pybind11/stl.h>
#include <string>

namespace thirdai::dataset::python {

namespace py = pybind11;

void createFeaturizationSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column");  // NOLINT

  py::class_<NumpyValueColumn<uint32_t>, Column,
             std::shared_ptr<NumpyValueColumn<uint32_t>>>(
      columns_submodule, "NumpySparseValueColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"));

  py::class_<NumpyValueColumn<float>, Column,
             std::shared_ptr<NumpyValueColumn<float>>>(columns_submodule,
                                                       "NumpyDenseValueColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"));

  py::class_<VectorValueColumn<std::string>, Column,
             std::shared_ptr<VectorValueColumn<std::string>>>(columns_submodule,
                                                              "StringColumn")
      .def(py::init<std::vector<std::string>>(), py::arg("array"));

  py::class_<NumpyArrayColumn<uint32_t>, Column,
             std::shared_ptr<NumpyArrayColumn<uint32_t>>>(
      columns_submodule, "NumpySparseArrayColumn")
      .def(py::init<const NumpyArray<uint32_t>&, uint32_t>(), py::arg("array"),
           py::arg("dim"));

  py::class_<NumpyArrayColumn<float>, Column,
             std::shared_ptr<NumpyArrayColumn<float>>>(columns_submodule,
                                                       "NumpyDenseArrayColumn")
      .def(py::init<const NumpyArray<float>&>(), py::arg("array"));

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

  py::class_<ColumnMap>(dataset_submodule, "ColumnMap")
      .def(py::init<std::unordered_map<std::string, ColumnPtr>>(),
           py::arg("columns"))
      .def("convert_to_dataset", &ColumnMap::convertToDataset,
           py::arg("columns"), py::arg("batch_size"))
      .def("__getitem__", &ColumnMap::getColumn);

  py::class_<FeaturizationPipeline>(dataset_submodule, "FeaturizationPipeline")
      .def(py::init<std::vector<TransformationPtr>>(),
           py::arg("transformations"))
      .def("featurize", &FeaturizationPipeline::featurize, py::arg("columns"));
}

}  // namespace thirdai::dataset::python