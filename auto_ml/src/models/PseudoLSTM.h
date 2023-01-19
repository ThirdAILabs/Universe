
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

namespace thirdai::automl::models {

// This copies the UDT API just with the requirements of LSTM to take command of
// data-loading and training internally.

class LSTMClassifier final : public ModelPipeline {
 public:
  void train(const std::shared_ptr<dataset::DataSource>& data_source,
             bolt::TrainConfig& train_config,
             const std::optional<ValidationOptions>& validation,
             std::optional<uint32_t> max_in_memory_batches) {
    // Command parsing and dataloading here.
    // LSTM generates multiple batches from a line.
    char delimiter = ',';  // HARDCODE
    std::optional<std::string> line = data_source->nextLine();
    std::vector<std::string> header_columns =
        dataset::ProcessorUtils::parseCsvRow(line.value(), delimiter);

    // Buffer in lines, create a batch.
    while (line = data_source->nextLine() &&) {
      std::vector<BoltBatch> batch;
      std::vector<BoltVector> vector;
    }

    if (max_in_memory_batches) {
      trainOnStream(dataset, train_config, max_in_memory_batches.value(),
                    validation);
    } else {
      trainInMemory(dataset, train_config, validation);
    }
  }

  py::object predict(const MapInput& sample_in, bool use_sparse_inference,
                     bool return_predicted_class) {
    // Copy the sample to add the recursive predictions without modifying the
    // original.
    MapInput sample = sample_in;

    // The previous predictions of the model are initialized as empty. The are
    // filled in after each call to predict.
    for (uint32_t t = 1; t < _prediction_depth; t++) {
      setPredictionAtTimestep(sample, t, "");
    }

    NumpyArray<uint32_t> output_predictions(_prediction_depth);

    for (uint32_t t = 1; t <= _prediction_depth; t++) {
      py::object prediction =
          ModelPipeline::predict(sample, use_sparse_inference,
                                 /* return_predicted_class= */ true);

      // For V0 we are only supporting this feature for categorical tasks, not
      // regression.
      if (py::isinstance<py::int_>(prediction)) {
        // Update the sample with the current prediction. When the sample is
        // featurized in the next call to predict the information of this
        // prediction will then be passed into the model.
        uint32_t predicted_class = prediction.cast<uint32_t>();
        setPredictionAtTimestep(sample, t, className(predicted_class));

        // Update the array of returned predictions.
        output_predictions.mutable_at(t - 1) = predicted_class;
      } else {
        throw std::invalid_argument(
            "Unsupported prediction type for recursive predictions '" +
            py::str(prediction.get_type()).cast<std::string>() + "'.");
      }
    }

    return py::object(std::move(output_predictions));
  }

  py::object predictBatch(const MapInputBatch& samples_in,
                          bool use_sparse_inference,
                          bool return_predicted_class) {
    // Copy the sample to add the recursive predictions without modifying the
    // original.
    MapInputBatch samples = samples_in;

    // The previous predictions of the model are initialized as empty. The are
    // filled in after each call to predictBatch.
    for (auto& sample : samples) {
      for (uint32_t t = 1; t < _prediction_depth; t++) {
        setPredictionAtTimestep(sample, t, "");
      }
    }

    NumpyArray<uint32_t> output_predictions(
        /* shape= */ {samples.size(), static_cast<size_t>(_prediction_depth)});

    for (uint32_t t = 1; t <= _prediction_depth; t++) {
      py::object predictions =
          ModelPipeline::predictBatch(samples, use_sparse_inference,
                                      /* return_predicted_class= */ true);

      // For V0 we are only supporting this feature for categorical tasks, not
      // regression.
      if (py::isinstance<NumpyArray<uint32_t>>(predictions)) {
        NumpyArray<uint32_t> predictions_np =
            predictions.cast<NumpyArray<uint32_t>>();

        assert(predictions_np.ndim() == 1);
        assert(static_cast<uint32_t>(predictions_np.shape(0)) ==
               samples.size());

        for (uint32_t i = 0; i < predictions_np.shape(0); i++) {
          // Update each sample with the current predictions. When the samples
          // are featurized in the next call to predictBatch the information of
          // these predictions will then be passed into the model.
          setPredictionAtTimestep(samples[i], t,
                                  className(predictions_np.at(i)));

          // Update the list of returned predictions.
          output_predictions.mutable_at(i, t - 1) = predictions_np.at(i);
        }
      } else {
        throw std::invalid_argument(
            "Unsupported prediction type for recursive predictions '" +
            py::str(predictions.get_type()).cast<std::string>() + "'.");
      }
    }

    return py::object(std::move(output_predictions));
  }

  void processUDTOptions(const deployment::UserInputMap& options_map) {}

 private:
  uint32_t _prediction_depth;
};
}  // namespace thirdai::automl::models