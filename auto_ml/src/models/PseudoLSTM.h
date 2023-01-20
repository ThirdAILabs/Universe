
#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Vocabulary.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace py = pybind11;

namespace thirdai::automl::models {

// This copies the UDT API just with the requirements of LSTM to take command of
// data-loading and training internally.

class LSTMClassifier final : public ModelPipeline {
  static constexpr uint32_t VOCABULARY_SIZE = 10000;

 public:
  LSTMClassifier(uint32_t n_classes = VOCABULARY_SIZE)
      : _vocab(dataset::ThreadSafeVocabulary::make(n_classes)) {}
  void train(const std::shared_ptr<dataset::DataSource>& data_source,
             bolt::TrainConfig& train_config,
             const std::optional<ValidationOptions>& validation,
             std::optional<uint32_t> max_in_memory_batches) {
    // Command parsing and dataloading here.
    // LSTM generates multiple batches from a line.
    char delimiter = ',';  // HARDCODE

    // We always expect a header for v1
    std::optional<std::string> line = data_source->nextLine();
    std::vector<std::string> header_columns =
        dataset::ProcessorUtils::parseCsvRow(line.value(), delimiter);

    uint32_t batch_count = 0;
    uint32_t INPUT_DIM = 1024;
    uint32_t batch_size = data_source->getMaxBatchSize();
    bool eof = false;
    while (eof) {
      // Buffer in lines, create a batch.
      std::vector<BoltBatch> input_batches;
      std::vector<BoltBatch> label_batches;
      std::vector<BoltVector> inputs;
      std::vector<BoltVector> labels;

      std::shared_ptr<Block> text =
          std::make_shared<dataset::TextBlock>(0, INPUT_DIM);

      while (input_batches.size() < max_in_memory_batches) {
        uint32_t sample_count = 0;
        line = data_source->nextLine();
        while (line) {
          std::vector<std::string_view> rows =
              dataset::ProcessorUtils::parseCsvRow(line.value(), delimiter);

          std::string source = std::string(rows[0].data(), rows[0].size());
          std::string target = std::string(rows[1].data(), rows[1].size());

          for (size_t i = 0; i < target.size(); i++) {
            std::string adjusted_source = source + target.substr(0, i);
            std::string adjusted_target = target.substr(i, 1);
            std::string sample = adjusted_source + "," + adjusted_target;

            std::vector<std::string_view> sample_rows =
                dataset::ProcessorUtils::parseCsvRow(line.value(), delimiter);

          std:
            std::shared_ptr<dataset::SegmentedFeatureVector>
                segmented_feature_vector =
                    std::make_shared<dataset::SegmentedDenseFeatureVector>();
            dataset::RowSampleRef ref(sample_rows);
            text->buildSegment(sample_rows, *segmented_feature_vector);
            inputs.push_back(segmented_feature_vector->toBoltVector());

            uint32_t target_id = _vocab->getUid(adjusted_target);
            labels.push_back(BoltVector::makeSparseVector({target_id}, {1.0}));

            if (sample_count == batch_count) }{
              // Cleave off a batch.
              input_batches.emplace_back(std::move(inputs));
              label_batches.emplace_back(std::move(labels));
            }
          
          line = data_source->nextLine();
        }
      }

      dataset::BoltDatasetPtr dataset;
      if (max_in_memory_batches) {
        trainOnStream(dataset, train_config, max_in_memory_batches.value(),
                      validation);
      } else {
        trainInMemory(dataset, train_config, validation);
      }

      if (!line) {
        eof = true;
      }
    }
  }

  template <class... Args>
  static std::shared_ptr<LSTMClassifier> buildLSTM(Args&&... args) {
    return std::make_shared<LSTMClassifier>();
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
  dataset::ThreadSafeVocabularyPtr vocab_;
};
}  // namespace thirdai::automl::models