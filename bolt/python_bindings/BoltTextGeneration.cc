#include "BoltTextGeneration.h"
#include "PybindUtils.h"
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/text_generation/ContextualModel.h>
#include <bolt/src/text_generation/DyadicModel.h>
#include <bolt/src/text_generation/GenerativeModel.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/DyadicInterval.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <pybind11/detail/common.h>
#include <pybind11/stl.h>
#include <cstddef>
#include <optional>
#include <unordered_set>

namespace thirdai::bolt::python {

void addTextGenerationModels(py::module_& module) {
#if THIRDAI_EXPOSE_ALL
  py::class_<GenerativeBackend, std::shared_ptr<GenerativeBackend>>(  // NOLINT
      module, "GenerativeBackend");

  py::class_<DyadicModel, GenerativeBackend, std::shared_ptr<DyadicModel>>(
      module, "DyadicModel")
      .def(py::init<bolt::ModelPtr, data::DyadicInterval,
                    data::OutputColumnsList>(),
           py::arg("model"), py::arg("dyadic_transform"),
           py::arg("bolt_inputs"));

  py::class_<ContextualModel, GenerativeBackend,
             std::shared_ptr<ContextualModel>>(module, "ContextualModel")
      .def(py::init<bolt::ModelPtr, dataset::TextGenerationFeaturizerPtr>(),
           py::arg("model"), py::arg("featurizer"));
#endif

  py::class_<BeamSearchDecoder>(module, "GenerativeDecoder")
      .def("__iter__", [](BeamSearchDecoder& decoder) { return decoder; })
      .def("__next__", [](BeamSearchDecoder& decoder) {
        if (auto tokens = decoder.next()) {
          return *tokens;
        }
        throw py::stop_iteration();
      });

  py::class_<GenerativeModel, std::shared_ptr<GenerativeModel>>(
      module, "GenerativeModel")
#if THIRDAI_EXPOSE_ALL
      .def(py::init(&GenerativeModel::make), py::arg("model"),
           py::arg("allowed_repeats"), py::arg("punctuation_tokens"),
           py::arg("punctuation_repeat_threshold") = 0.8)
#endif
      .def("generate", &GenerativeModel::generate, py::arg("input_tokens"),
           py::arg("prompt") = std::vector<std::uint32_t>{},
           py::arg("max_predictions"), py::arg("beam_width"),
           py::arg("temperature") = std::nullopt)
      .def("generate_batch", &GenerativeModel::generateBatch,
           py::arg("input_tokens_batch"),
           py::arg("prompt_batch") = std::vector<std::vector<std::uint32_t>>{},
           py::arg("max_predictions"), py::arg("beam_width"),
           py::arg("temperature") = std::nullopt)
      .def("streaming_generate", &GenerativeModel::streamingGenerate,
           py::arg("input_tokens"),
           py::arg("prompt") = std::vector<std::uint32_t>{},
           py::arg("prediction_chunk_size"), py::arg("max_predictions"),
           py::arg("beam_width"), py::arg("temperature") = std::nullopt)
      .def("train", &GenerativeModel::train, py::arg("train_data"),
           py::arg("learning_rate") = 1e-5, py::arg("epochs") = 5,
           py::arg("batch_size") = 10000,
           py::arg("train_metrics") = std::vector<std::string>{"loss"},
           py::arg("val_data") = nullptr,
           py::arg("val_metrics") = std::vector<std::string>{},
           py::arg("max_in_memory_batches") = std::nullopt,
           py::arg("callbacks") = std::vector<callbacks::CallbackPtr>{},
           py::arg("comm") = nullptr)
      .def("save", &GenerativeModel::save)
      .def_static("load", &GenerativeModel::load, py::arg("filename"))
      .def_property_readonly("model", &GenerativeModel::getBoltModel,
                             py::return_value_policy::reference_internal)
      .def(thirdai::bolt::python::getPickleFunction<GenerativeModel>());
}

}  // namespace thirdai::bolt::python