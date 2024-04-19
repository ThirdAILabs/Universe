#include "DataPython.h"
#include <bolt/python_bindings/PybindUtils.h>
#include <data/src/ColumnMapIterator.h>
#include <data/src/Loader.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ArrayColumns.h>
#include <data/src/columns/ValueColumns.h>
#include <data/src/transformations/Binning.h>
#include <data/src/transformations/CategoricalTemporal.h>
#include <data/src/transformations/CrossColumnPairgrams.h>
#include <data/src/transformations/Date.h>
#include <data/src/transformations/DyadicInterval.h>
#include <data/src/transformations/FeatureHash.h>
#include <data/src/transformations/MachLabel.h>
#include <data/src/transformations/NextWordPrediction.h>
#include <data/src/transformations/Pipeline.h>
#include <data/src/transformations/SpladeAugmentation.h>
#include <data/src/transformations/SpladeMachAugmentation.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringConcat.h>
#include <data/src/transformations/StringHash.h>
#include <data/src/transformations/StringIDLookup.h>
#include <data/src/transformations/TextTokenizer.h>
#include <data/src/transformations/Transformation.h>
#include <data/src/transformations/cold_start/ColdStartText.h>
#include <data/src/transformations/cold_start/VariableLengthColdStart.h>
#include <dataset/src/blocks/text/TextEncoder.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <pybind11/attr.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utils/Random.h>
#include <utils/text/PorterStemmer.h>
#include <cstddef>
#include <memory>
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
      .def(py::init(&ColumnMap::fromMapInput), py::arg("sample"))
      .def(py::init(&ColumnMap::fromMapInputBatch), py::arg("samples"))
      .def("num_rows", &ColumnMap::numRows)
      .def("__getitem__", &ColumnMap::getColumn)
      .def(
          "__iter__",
          [](const ColumnMap& columns) {
            return py::make_iterator(columns.begin(), columns.end());
          },
          py::keep_alive<0, 1>())
      .def("__len__", &ColumnMap::numRows)
      .def("columns", &ColumnMap::columns)
      .def("shuffle", &ColumnMap::shuffle,
           py::arg("seed") = global_random::nextSeed())
      .def("concat", &ColumnMap::concat, py::arg("other"))
      .def("split", &ColumnMap::split, py::arg("offset"));

  createColumnsSubmodule(dataset_submodule);

  createTransformationsSubmodule(dataset_submodule);

  py::class_<ColumnMapIterator, ColumnMapIteratorPtr>(dataset_submodule,
                                                      "ColumnMapIterator")
      .def("next", &ColumnMapIterator::next);

  py::class_<CsvIterator, std::shared_ptr<CsvIterator>, ColumnMapIterator>(
      dataset_submodule, "CsvIterator")
      .def(py::init<const std::string&, char, size_t>(), py::arg("filename"),
           py::arg("delimiter") = ',',
           py::arg("rows_per_load") = ColumnMapIterator::DEFAULT_ROWS_PER_LOAD)
      .def(py::init<DataSourcePtr, char, size_t>(), py::arg("data_source"),
           py::arg("delimiter") = ',',
           py::arg("rows_per_load") = ColumnMapIterator::DEFAULT_ROWS_PER_LOAD)
      .def_static("all", &CsvIterator::all, py::arg("data_source"),
                  py::arg("delimiter"));

  py::class_<JsonIterator, std::shared_ptr<JsonIterator>, ColumnMapIterator>(
      dataset_submodule, "JsonIterator")
      .def(py::init<DataSourcePtr, std::vector<std::string>, size_t>(),
           py::arg("data_source"), py::arg("columns"),
           py::arg("rows_per_load") = ColumnMapIterator::DEFAULT_ROWS_PER_LOAD);

  py::class_<TransformedIterator, std::shared_ptr<TransformedIterator>,
             ColumnMapIterator>(dataset_submodule, "TransformedIterator")
      .def(py::init<ColumnMapIteratorPtr, TransformationPtr, StatePtr>(),
           py::arg("iter"), py::arg("transformation"),
           py::arg("state") = nullptr);

  py::enum_<ValueFillType>(dataset_submodule, "ValueFillType")
      .value("Ones", ValueFillType::Ones)
      .value("SumToOne", ValueFillType::SumToOne)
      .export_values();

  py::class_<OutputColumns>(dataset_submodule, "OutputColumns")
      .def(py::init<std::string, std::string>(), py::arg("indices"),
           py::arg("values"))
      .def(py::init<std::string, ValueFillType>(), py::arg("indices"),
           py::arg("value_fill_type") = ValueFillType::Ones);

  py::class_<Loader, LoaderPtr>(dataset_submodule, "Loader")
      .def(py::init(&Loader::make), py::arg("data_iterator"),
           py::arg("transformation"), py::arg("state"),
           py::arg("input_columns"), py::arg("output_columns"),
           py::arg("batch_size"), py::arg("shuffle") = true,
           py::arg("verbose") = true,
           py::arg("shuffle_buffer_size") = Loader::DEFAULT_SHUFFLE_BUFFER_SIZE,
           py::arg("shuffle_seed") = global_random::nextSeed())
      .def("next", &Loader::next, py::arg("max_batches") = Loader::NO_LIMIT)
      .def("next_column_map", &Loader::nextColumnMap,
           py::arg("max_batches") = Loader::NO_LIMIT)
      .def("all", &Loader::all);

  dataset_submodule.def("to_tensors", &toTensorBatches, py::arg("column_map"),
                        py::arg("columns_to_convert"), py::arg("batch_size"));

  dataset_submodule.def(
      "stem",
      py::overload_cast<const std::vector<std::string>&, bool>(
          &text::porter_stemmer::stem),
      py::arg("words"), py::arg("lowercase") = true);

  dataset_submodule.def(
      "stem",
      py::overload_cast<const std::string&, bool>(&text::porter_stemmer::stem),
      py::arg("word"), py::arg("lowercase") = true);
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
  return ValueColumn<uint32_t>::make(fromNumpy1D(array), dim);
}

auto decimalColumnFromNumpy(const NumpyArray<float>& array) {
  return ValueColumn<float>::make(fromNumpy1D(array));
}

auto tokenArrayColumnFromNumpy(const NumpyArray<uint32_t>& array,
                               std::optional<size_t> dim) {
  return ArrayColumn<uint32_t>::make(fromNumpy2D(array), dim);
}

auto decimalArrayColumnFromNumpy(const NumpyArray<float>& array,
                                 std::optional<size_t> dim) {
  return ArrayColumn<float>::make(fromNumpy2D(array), dim);
}

void createColumnsSubmodule(py::module_& dataset_submodule) {
  auto columns_submodule = dataset_submodule.def_submodule("columns");

  py::class_<Column, ColumnPtr>(columns_submodule, "Column")
      .def("dim", &Column::dim)
      .def("__len__", &Column::numRows);

  py::class_<ValueColumn<uint32_t>, Column, ValueColumnPtr<uint32_t>>(
      columns_submodule, "TokenColumn")
      .def(
          py::init(
              py::overload_cast<std::vector<uint32_t>&&, std::optional<size_t>>(
                  &ValueColumn<uint32_t>::make)),
          py::arg("data"), py::arg("dim") = std::nullopt)
      .def(py::init(&tokenColumnFromNumpy), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(("__getitem__"), &ValueColumn<uint32_t>::value)
      .def("data", &ValueColumn<uint32_t>::data);

  py::class_<ValueColumn<float>, Column, ValueColumnPtr<float>>(
      columns_submodule, "DecimalColumn")
      .def(py::init(py::overload_cast<std::vector<float>&&>(
               &ValueColumn<float>::make)),
           py::arg("data"))
      .def(py::init(&decimalColumnFromNumpy), py::arg("data"))
      .def(("__getitem__"), &ValueColumn<float>::value)
      .def("data", &ValueColumn<float>::data);

  py::class_<ValueColumn<std::string>, Column, ValueColumnPtr<std::string>>(
      columns_submodule, "StringColumn")
      .def(py::init(py::overload_cast<std::vector<std::string>&&>(
               &ValueColumn<std::string>::make)),
           py::arg("data"))
      .def("__getitem__", &ValueColumn<std::string>::value)
      .def("data", &ValueColumn<std::string>::data);

  py::class_<ValueColumn<int64_t>, Column, ValueColumnPtr<int64_t>>(
      columns_submodule, "TimestampColumn")
      .def(py::init(py::overload_cast<std::vector<int64_t>&&>(
               &ValueColumn<int64_t>::make)),
           py::arg("data"))
      .def("__getitem__", &ValueColumn<int64_t>::value)
      .def("data", &ValueColumn<int64_t>::data);

  py::class_<ArrayColumn<uint32_t>, Column, ArrayColumnPtr<uint32_t>>(
      columns_submodule, "TokenArrayColumn")
      .def(py::init(&ArrayColumn<uint32_t>::make), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(py::init(&tokenArrayColumnFromNumpy), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def("__getitem__", &getRowNumpy<uint32_t, ArrayColumn<uint32_t>>,
           py::return_value_policy::reference_internal)
      .def("data", &ArrayColumn<uint32_t>::data);

  py::class_<ArrayColumn<float>, Column, ArrayColumnPtr<float>>(
      columns_submodule, "DecimalArrayColumn")
      .def(py::init(&ArrayColumn<float>::make), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def(py::init(&decimalArrayColumnFromNumpy), py::arg("data"),
           py::arg("dim") = std::nullopt)
      .def("__getitem__", &getRowNumpy<float, ArrayColumn<float>>,
           py::return_value_policy::reference_internal)
      .def("data", &ArrayColumn<float>::data);
}

void createTransformationsSubmodule(py::module_& dataset_submodule) {
  auto transformations_submodule =
      dataset_submodule.def_submodule("transformations");

  py::class_<State, StatePtr>(transformations_submodule, "State")
      .def(py::init<>())
      .def(py::init<MachIndexPtr>(), py::arg("mach_index"));

  py::class_<Transformation, std::shared_ptr<Transformation>>(
      transformations_submodule, "Transformation")
      .def("__call__", &Transformation::applyStateless, py::arg("columns"))
      .def("__call__", &Transformation::apply, py::arg("columns"),
           py::arg("state"))
      .def("serialize", [](const TransformationPtr& transformation) {
        return py::bytes(transformation->serialize());
      });

  transformations_submodule.def("deserialize", &Transformation::deserialize,
                                py::arg("binary"));

  py::class_<Pipeline, Transformation, PipelinePtr>(transformations_submodule,
                                                    "Pipeline")
      .def(py::init(&Pipeline::make),
           py::arg("transformations") = std::vector<TransformationPtr>{})
      .def("then", &Pipeline::then, py::arg("transformation"));

  py::class_<StringToToken, Transformation, std::shared_ptr<StringToToken>>(
      transformations_submodule, "ToTokens")
      .def(py::init<std::string, std::string, std::optional<uint32_t>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("dim") = std::nullopt);

  py::class_<StringToTokenArray, Transformation,
             std::shared_ptr<StringToTokenArray>>(transformations_submodule,
                                                  "ToTokenArrays")
      .def(py::init<std::string, std::string, char, std::optional<uint32_t>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("delimiter"), py::arg("dim") = std::nullopt);

  py::class_<StringToDecimal, Transformation, std::shared_ptr<StringToDecimal>>(
      transformations_submodule, "ToDecimals")
      .def(py::init<std::string, std::string>(), py::arg("input_column"),
           py::arg("output_column"));

  py::class_<StringToDecimalArray, Transformation,
             std::shared_ptr<StringToDecimalArray>>(transformations_submodule,
                                                    "ToDecimalArrays")
      .def(py::init<std::string, std::string, char, std::optional<size_t>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("delimiter"), py::arg("dim") = std::nullopt);

  py::class_<StringToTimestamp, Transformation,
             std::shared_ptr<StringToTimestamp>>(transformations_submodule,
                                                 "ToTimestamps")
      .def(py::init<std::string, std::string, std::string>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("format") = "%Y-%m-%d");

#if THIRDAI_EXPOSE_ALL
  py::class_<TextTokenizer, Transformation, std::shared_ptr<TextTokenizer>>(
      transformations_submodule, "Text")
      .def(py::init<std::string, std::string, std::optional<std::string>,
                    dataset::TextTokenizerPtr, dataset::TextEncoderPtr, bool,
                    size_t>(),
           py::arg("input_column"), py::arg("output_indices"),
           py::arg("output_values") = std::nullopt,
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
                    std::optional<char>, uint32_t>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("output_range") = std::nullopt,
           py::arg("delimiter") = std::nullopt, py::arg("seed") = 42);

  py::class_<CrossColumnPairgrams, Transformation,
             std::shared_ptr<CrossColumnPairgrams>>(transformations_submodule,
                                                    "CrossColumnPairgrams")
      .def(py::init<std::vector<std::string>, std::string, uint32_t>(),
           py::arg("input_columns"), py::arg("output_column"),
           py::arg("hash_range") = std::numeric_limits<size_t>::max());

  py::class_<FeatureHash, Transformation, std::shared_ptr<FeatureHash>>(
      transformations_submodule, "FeatureHash")
      .def(py::init<std::vector<std::string>, std::string, std::string,
                    size_t>(),
           py::arg("input_columns"), py::arg("output_indices_column"),
           py::arg("output_values_column"), py::arg("hash_range"));

  py::class_<StringIDLookup, Transformation, std::shared_ptr<StringIDLookup>>(
      transformations_submodule, "StringLookup")
      .def(py::init<std::string, std::string, std::string,
                    std::optional<size_t>, std::optional<char>>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("vocab_key"), py::arg("max_vocab_size") = std::nullopt,
           py::arg("delimiter") = std::nullopt);

  py::class_<CategoricalTemporal, Transformation,
             std::shared_ptr<CategoricalTemporal>>(transformations_submodule,
                                                   "CategoricalTemporal")
      .def(py::init<std::string, std::string, std::string, std::string,
                    std::string, size_t, bool, bool, int64_t>(),
           py::arg("user_column"), py::arg("item_column"),
           py::arg("timestamp_column"), py::arg("output_column"),
           py::arg("tracker_key"), py::arg("track_last_n"),
           py::arg("should_update_history") = true,
           py::arg("include_current_row") = false, py::arg("time_lag") = 0);

  py::class_<Date, Transformation, std::shared_ptr<Date>>(
      transformations_submodule, "Date")
      .def(py::init<std::string, std::string, std::string>(),
           py::arg("input_column"), py::arg("output_column"),
           py::arg("format") = "%Y-%m-%d");

  py::class_<StringConcat, Transformation, std::shared_ptr<StringConcat>>(
      transformations_submodule, "StringConcat")
      .def(py::init<std::vector<std::string>, std::string, std::string>(),
           py::arg("input_columns"), py::arg("output_column"),
           py::arg("separator") = "");

  py::class_<ColdStartConfig>(transformations_submodule, "ColdStartConfig")
      .def(py::init<std::optional<uint32_t>, std::optional<uint32_t>,
                    std::optional<uint32_t>, std::optional<uint32_t>, uint32_t,
                    std::optional<uint32_t>, std::optional<uint32_t>,
                    std::optional<uint32_t>>(),
           py::arg("weak_min_len") = std::nullopt,
           py::arg("weak_max_len") = std::nullopt,
           py::arg("weak_chunk_len") = std::nullopt,
           py::arg("weak_sample_num_words") = std::nullopt,
           py::arg("weak_sample_reps") = 1,
           py::arg("strong_max_len") = std::nullopt,
           py::arg("strong_sample_num_words") = std::nullopt,
           py::arg("strong_to_weak_ratio") = std::nullopt);

  py::class_<ColdStartTextAugmentation, Transformation,
             std::shared_ptr<ColdStartTextAugmentation>>(
      transformations_submodule, "ColdStartText")
      .def(py::init<std::vector<std::string>, std::vector<std::string>,
                    std::string, const ColdStartConfig&, uint32_t>(),
           py::arg("strong_columns"), py::arg("weak_columns"),
           py::arg("output_column"),
           py::arg("config") = ColdStartConfig::longBothPhrases(),
           py::arg("seed") = global_random::nextSeed())
      .def("augment_single_row", &ColdStartTextAugmentation::augmentSingleRow,
           py::arg("strong_text"), py::arg("weak_text"),
           py::arg("row_id_salt") = global_random::nextSeed())
      .def("augment_map_input", &ColdStartTextAugmentation::augmentMapInput,
           py::arg("document"));
#endif

  py::class_<VariableLengthConfig,
             std::shared_ptr<VariableLengthConfig>>(  // NOLINT
      transformations_submodule, "VariableLengthConfig")
#if THIRDAI_EXPOSE_ALL
      .def(py::init<size_t, size_t, std::optional<uint32_t>, size_t,
                    std::optional<size_t>, uint32_t, bool, bool, uint32_t,
                    std::optional<uint32_t>, float, float, float, float, size_t,
                    size_t, size_t, size_t>(),
           py::arg("covering_min_length") = 5,
           py::arg("covering_max_length") = 40,
           py::arg("max_covering_samples") = std::nullopt,
           py::arg("slice_min_length") = 5,
           py::arg("slice_max_length") = std::nullopt,
           py::arg("num_slices") = 7, py::arg("add_whole_doc") = true,
           py::arg("prefilter_punctuation") = true,
           py::arg("strong_sample_num_words") = 3,
           py::arg("strong_to_weak_ratio") = std::nullopt,
           py::arg("stopword_removal_probability") = 0,
           py::arg("stopword_insertion_probability") = 0,
           py::arg("word_removal_probability") = 0,
           py::arg("word_perturbation_probability") = 0,
           py::arg("chars_replace_with_space") = 0,
           py::arg("chars_deleted") = 0, py::arg("chars_duplicated") = 0,
           py::arg("chars_replace_with_adjacents") = 0)
      .def("__str__", &VariableLengthConfig::to_string)
#else
      .def(py::init<>())
#endif
      .def(thirdai::bolt::python::getPickleFunction<VariableLengthConfig>());

#if THIRDAI_EXPOSE_ALL
  py::class_<VariableLengthColdStart, Transformation,
             std::shared_ptr<VariableLengthColdStart>>(
      transformations_submodule, "VariableLengthColdStart")
      .def(py::init<std::vector<std::string>, std::vector<std::string>,
                    std::string, VariableLengthConfig, uint32_t>(),
           py::arg("strong_columns"), py::arg("weak_columns"),
           py::arg("output_column"), py::arg("config") = VariableLengthConfig(),
           py::arg("seed") = global_random::nextSeed())
      .def("augment_single_row", &VariableLengthColdStart::augmentSingleRow,
           py::arg("strong_text"), py::arg("weak_text"),
           py::arg("row_id_salt") = global_random::nextSeed());

  py::class_<MachLabel, Transformation, std::shared_ptr<MachLabel>>(
      transformations_submodule, "MachLabel")
      .def(py::init<std::string, std::string>(), py::arg("input_column"),
           py::arg("output_column"));

  py::class_<DyadicInterval, Transformation, std::shared_ptr<DyadicInterval>>(
      transformations_submodule, "DyadicInterval")
      .def(py::init<std::string, std::optional<std::string>,
                    std::optional<std::string>, std::string, std::string,
                    size_t, bool>(),
           py::arg("input_column"), py::arg("context_column") = std::nullopt,
           py::arg("prompt_column") = std::nullopt,
           py::arg("output_interval_prefix"), py::arg("target_column"),
           py::arg("n_intervals"), py::arg("is_bidirectional") = false)
      .def("inference_featurization", &DyadicInterval::inferenceFeaturization,
           py::arg("columns"));

  py::class_<NextWordPrediction, Transformation,
             std::shared_ptr<NextWordPrediction>>(transformations_submodule,
                                                  "NextWordPrediction")
      .def(py::init<std::string, std::string, std::string>(),
           py::arg("input_column"), py::arg("context_column"),
           py::arg("target_column"));
#endif

  py::class_<SpladeConfig, std::shared_ptr<SpladeConfig>>(
      transformations_submodule, "SpladeConfig")
      .def(py::init<std::string, std::string, std::optional<size_t>,
                    std::optional<float>, bool, size_t, bool,
                    std::optional<uint32_t>>(),
           py::arg("model_checkpoint"), py::arg("tokenizer_vocab"),
           py::arg("n_augmented_tokens") = 100,
           py::arg("augmentation_frac") = std::nullopt,
           py::arg("filter_tokens") = true, py::arg("batch_size") = 4096,
           py::arg("lowercase") = true, py::arg("strong_sample_override") = 7)
      .def(bolt::python::getPickleFunction<SpladeConfig>());

  py::class_<SpladeAugmentation, Transformation,
             std::shared_ptr<SpladeAugmentation>>(transformations_submodule,
                                                  "SpladeAugmentation")
      .def(py::init<std::string, std::string, const SpladeConfig&>(),
           py::arg("input_column"), py::arg("output_column"), py::arg("config"))
      .def(py::init<std::string, std::string, bolt::ModelPtr,
                    dataset::WordpieceTokenizerPtr, std::optional<size_t>,
                    std::optional<float>, bool, size_t>(),
           py::arg("input_column"), py::arg("output_column"), py::arg("model"),
           py::arg("tokenizer"), py::arg("n_augmented_tokens") = 100,
           py::arg("augmentation_frac") = std::nullopt,
           py::arg("filter_tokens") = true, py::arg("batch_size") = 4096);

  py::class_<SpladeMachAugmentation, Transformation,
             std::shared_ptr<SpladeMachAugmentation>>(transformations_submodule,
                                                      "SpladeMachAugmentation")
      .def(py::init<std::string, std::string,
                    std::shared_ptr<automl::SpladeMach>, size_t>(),
           py::arg("input_column"), py::arg("output_column"), py::arg("model"),
           py::arg("n_hashes_per_model"));
}

}  // namespace thirdai::data::python