#include "Metric.h"

namespace thirdai::bolt_v1 {

void CategoricalAccuracy::record(const BoltVector& output,
                                 const BoltVector& labels) {
  float max_act = -std::numeric_limits<float>::max();
  std::optional<uint32_t> max_act_index = std::nullopt;
  for (uint32_t i = 0; i < output.len; i++) {
    if (output.activations[i] > max_act) {
      max_act = output.activations[i];
      max_act_index = i;
    }
  }

  if (!max_act_index) {
    throw std::runtime_error(
        "Unable to find a output activation larger than the minimum "
        "representable float. This is likely due to a Nan or incorrect "
        "activation function in the final layer.");
  }

  // The nueron with the largest activation is the prediction
  uint32_t pred =
      output.isDense() ? *max_act_index : output.active_neurons[*max_act_index];

  if (labels.isDense()) {
    // If labels are dense we check if the prediction has a non-zero label.
    if (labels.activations[pred] > 0) {
      _correct++;
    }
  } else {
    // If the labels are sparse then we have to search the list of labels for
    // the prediction.
    const uint32_t* label_start = labels.active_neurons;
    const uint32_t* label_end = labels.active_neurons + labels.len;
    if (std::find(label_start, label_end, pred) != label_end) {
      _correct++;
    }
  }
  _num_samples++;
}

double CategoricalAccuracy::value() {
  double acc = static_cast<double>(_correct) / _num_samples;
  return acc;
}

void CategoricalAccuracy::reset() {
  _correct = 0;
  _num_samples = 0;
}

std::string CategoricalAccuracy::summary() {
  std::stringstream stream;
  stream << NAME << ": " << value();
  return stream.str();
}

void MeanSquaredErrorMetric::record(const BoltVector& output,
                                    const BoltVector& labels) {
  float error;
  if (output.isDense()) {
    if (labels.isDense()) {
      error = computeMSE<true, true>(output, labels);
    } else {
      error = computeMSE<true, false>(output, labels);
    }
  } else {
    if (labels.isDense()) {
      error = computeMSE<false, true>(output, labels);

    } else {
      error = computeMSE<false, false>(output, labels);
    }
  }

  MetricUtilities::incrementAtomicFloat(_mse, error);
  _num_samples++;
}

double MeanSquaredErrorMetric::value() {
  double error = _mse / _num_samples;
  return error;
}

void MeanSquaredErrorMetric::reset() {
  _mse = 0;
  _num_samples = 0;
}

template <bool OUTPUT_DENSE, bool LABEL_DENSE>
float MeanSquaredErrorMetric::computeMSE(const BoltVector& output,
                                         const BoltVector& labels) {
  if constexpr (OUTPUT_DENSE || LABEL_DENSE) {
    // If either vector is dense then we need to iterate over the full
    // dimension from the layer.
    uint32_t dim = std::max(output.len, labels.len);

    float error = 0.0;
    for (uint32_t i = 0; i < dim; i++) {
      float label = labels.findActiveNeuron<LABEL_DENSE>(i).activation;
      float act = output.findActiveNeuron<OUTPUT_DENSE>(i).activation;
      float delta = label - act;
      error += delta * delta;
    }
    return error;
  }

  // If both are sparse then we need to iterate over the nonzeros from both
  // vectors. To avoid double counting the overlapping neurons we avoid
  // computing the error while iterating over the labels for neurons that are
  // also in the output active neurons.
  float error = 0.0;
  for (uint32_t i = 0; i < output.len; i++) {
    float label = labels.findActiveNeuron<LABEL_DENSE>(output.active_neurons[i])
                      .activation;
    float act = output.activations[i];
    float delta = label - act;
    error += delta * delta;
  }

  for (uint32_t i = 0; i < labels.len; i++) {
    auto output_neuron =
        output.findActiveNeuron<OUTPUT_DENSE>(labels.active_neurons[i]);
    // Skip any neurons that were in the active neuron set since the loss was
    // already computed for them.
    if (!output_neuron.pos) {
      float label = labels.activations[i];
      // The activation is 0 since this isn't in the output active neurons.
      error += label * label;
    }
  }
  return error;
}

template float MeanSquaredErrorMetric::computeMSE<false, false>(
    const BoltVector& output, const BoltVector& labels);
template float MeanSquaredErrorMetric::computeMSE<false, true>(
    const BoltVector& output, const BoltVector& labels);
template float MeanSquaredErrorMetric::computeMSE<true, false>(
    const BoltVector& output, const BoltVector& labels);
template float MeanSquaredErrorMetric::computeMSE<true, true>(
    const BoltVector& output, const BoltVector& labels);

std::string MeanSquaredErrorMetric::summary() {
  std::stringstream stream;
  stream << NAME << ": " << value();
  return stream.str();
}

void WeightedMeanAbsolutePercentageError::record(const BoltVector& output,
                                                 const BoltVector& labels) {
  // Calculate |actual - predicted| and |actual|.
  float sum_of_squared_differences = 0.0;
  float sum_of_squared_label_elems = 0.0;
  MetricUtilities::visitActiveNeurons(
      output, labels, [&](float label_val, float output_val) {
        float difference = label_val - output_val;
        sum_of_squared_differences += difference * difference;
        sum_of_squared_label_elems += label_val * label_val;
      });

  // Add to respective atomic accumulators
  MetricUtilities::incrementAtomicFloat(_sum_of_deviations,
                                        std::sqrt(sum_of_squared_differences));
  MetricUtilities::incrementAtomicFloat(_sum_of_truths,
                                        std::sqrt(sum_of_squared_label_elems));
}

double WeightedMeanAbsolutePercentageError::value() {
  double wmape = _sum_of_deviations /
                 std::max(_sum_of_truths.load(std::memory_order_relaxed),
                          std::numeric_limits<float>::epsilon());
  return wmape;
}

void WeightedMeanAbsolutePercentageError::reset() {
  _sum_of_deviations = 0.0;
  _sum_of_truths = 0.0;
}

std::string WeightedMeanAbsolutePercentageError::summary() {
  std::stringstream stream;
  stream << NAME << ": " << value();
  return stream.str();
}

void RecallAtK::record(const BoltVector& output, const BoltVector& labels) {
  auto top_k = output.topKNeurons(_k);

  uint32_t matches = 0;
  while (!top_k.empty()) {
    if (labels
            .findActiveNeuronNoTemplate(
                /* active_neuron= */ top_k.top().second)
            .activation > 0) {
      matches++;
    }
    top_k.pop();
  }

  _matches.fetch_add(matches);
  _label_count.fetch_add(countLabels(labels));
}

double RecallAtK::value() {
  double metric = static_cast<double>(_matches) / _label_count;
  return metric;
}

void RecallAtK::reset() {
  _matches = 0;
  _label_count = 0;
}

std::string RecallAtK::summary() {
  std::stringstream stream;
  stream << "Recall@" << _k << ": " << std::setprecision(3) << value();
  return stream.str();
}

bool RecallAtK::isRecallAtK(const std::string& name) {
  return std::regex_match(name, std::regex("recall@[1-9]\\d*"));
}

std::shared_ptr<Metric> RecallAtK::make(const std::string& name) {
  if (!isRecallAtK(name)) {
    std::stringstream error_ss;
    error_ss << "Invoked RecallAtK::make with invalid string '" << name
             << "'. RecallAtK::make should be invoked with a string in "
                "the format 'recall@k', where k is a positive integer.";
    throw std::invalid_argument(error_ss.str());
  }

  char* end_ptr;
  auto k = std::strtol(name.data() + 7, &end_ptr, 10);
  if (k <= 0) {
    std::stringstream error_ss;
    error_ss << "RecallAtK invoked with k = " << k
             << ". k should be greater than 0.";
    throw std::invalid_argument(error_ss.str());
  }

  return std::make_shared<RecallAtK>(k);
}

uint32_t RecallAtK::countLabels(const BoltVector& labels) {
  uint32_t correct_labels = 0;
  for (uint32_t i = 0; i < labels.len; i++) {
    if (labels.activations[i] > 0) {
      correct_labels++;
    }
  }
  return correct_labels;
}

void PrecisionAtK::record(const BoltVector& output, const BoltVector& labels) {
  auto top_k = output.topKNeurons(_k);

  uint32_t correct_guesses = 0;
  while (!top_k.empty()) {
    if (labels
            .findActiveNeuronNoTemplate(
                /* active_neuron= */ top_k.top().second)
            .activation > 0) {
      correct_guesses++;
    }
    top_k.pop();
  }

  _correct_guesses += correct_guesses;
  _samples += 1;
}

double PrecisionAtK::value() {
  double metric = static_cast<double>(_correct_guesses) / (_samples * _k);
  return metric;
}

void PrecisionAtK::reset() {
  _correct_guesses = 0;
  _samples = 0;
}

std::string PrecisionAtK::summary() {
  std::stringstream stream;
  stream << "Precision@" << _k << ": " << std::setprecision(3) << value();
  return stream.str();
}

bool PrecisionAtK::isPrecisionAtK(const std::string& name) {
  return std::regex_match(name, std::regex("precision@[1-9]\\d*"));
}

std::shared_ptr<Metric> PrecisionAtK::make(const std::string& name) {
  if (!isPrecisionAtK(name)) {
    std::stringstream error_ss;
    error_ss << "Invoked PrecisionAtK::make with invalid string '" << name
             << "'. PrecisionAtK::make should be invoked with a string in "
                "the format 'precision@k', where k is a positive integer.";
    throw std::invalid_argument(error_ss.str());
  }

  char* end_ptr;
  auto k = std::strtol(name.data() + 10, &end_ptr, 10);
  if (k <= 0) {
    std::stringstream error_ss;
    error_ss << "PrecisionAtK invoked with k = " << k
             << ". k should be greater than 0.";
    throw std::invalid_argument(error_ss.str());
  }

  return std::make_shared<PrecisionAtK>(k);
}

void FMeasure::record(const BoltVector& output, const BoltVector& labels) {
  auto predictions = output.getThresholdedNeurons(
      /* activation_threshold = */ _threshold,
      /* return_at_least_one = */ true,
      /* max_count_to_return = */ std::numeric_limits<uint32_t>::max());

  for (uint32_t pred : predictions) {
    if (labels.findActiveNeuronNoTemplate(pred).activation > 0) {
      _true_positive++;
    } else {
      _false_positive++;
    }
  }

  for (uint32_t pos = 0; pos < labels.len; pos++) {
    uint32_t label_active_neuron =
        labels.isDense() ? pos : labels.active_neurons[pos];
    if (labels.findActiveNeuronNoTemplate(label_active_neuron).activation > 0) {
      if (std::find(predictions.begin(), predictions.end(),
                    label_active_neuron) == predictions.end()) {
        _false_negative++;
      }
    }
  }
}

double FMeasure::value() {
  auto [precision, recall, f_measure] = metrics();
  return f_measure;
}

std::tuple<double, double, double> FMeasure::metrics() {
  double prec =
      static_cast<double>(_true_positive) / (_true_positive + _false_positive);
  double recall =
      static_cast<double>(_true_positive) / (_true_positive + _false_negative);
  double f_measure;

  /*
    P = Precision
    R = Recall
    F = (1 + beta^2) * PR) / (beta^2 * P + R)
  */
  double denom = _beta_squared * prec + recall;

  if (denom == 0) {
    f_measure = 0;
  } else {
    f_measure = (1 + _beta_squared) * prec * recall / denom;
  }

  return {prec, recall, f_measure};
}

void FMeasure::reset() {
  _true_positive = 0;
  _false_positive = 0;
  _false_negative = 0;
}

std::string FMeasure::name() {
  std::stringstream name_ss;
  name_ss << NAME << '(' << _threshold << ')';
  return name_ss.str();
}

std::string FMeasure::summary() {
  auto [precision, recall, f_measure] = metrics();
  std::stringstream stream;
  stream << "precision(t=" << _threshold << "):" << precision;
  stream << ", "
         << "recall(t=" << _threshold << "):" << recall;
  stream << ", "
         << "f-measure(t=" << _threshold << "):" << f_measure;
  return stream.str();
}

void CategoricalCrossEntropy::record(const BoltVector& outputs,
                                     const BoltVector& labels) {
  float sample_loss = 0;
  const float EPS = 1e-7;
  if (outputs.isDense()) {
    if (labels.isDense()) {
      // (Dense Output, Dense Labels)
      // If both are dense, they're expected to have the same length.
      // In this case, we may simply run over the dense vectors and compute
      // sum((p_i)log(q_i)).
      assert(outputs.len == labels.len);
      for (uint32_t i = 0; i < outputs.len; i++) {
        sample_loss +=
            labels.activations[i] * std::log(outputs.activations[i] + EPS);
      }
    } else {
      // (Dense Output, Sparse Labels)
      // In this case, outputs are dense. There could potentially be 0 values,
      // but log(0+EPS) takes care of those. We only need to add terms if
      // there's labels active. For the non-active labels, 0*log(x) = 0.
      for (uint32_t i = 0; i < outputs.len; i++) {
        const uint32_t* label_start = labels.active_neurons;
        const uint32_t* label_end = labels.active_neurons + labels.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* label_query = std::find(label_start, label_end, i);

        if (label_query != label_end) {
          // In this case, we have found the labels. Other label activations
          // are 0, so we can ignore (0*log(whatever)).
          //
          // Compute label_index to lookup the value from labels
          // sparse-vector.
          size_t label_index = std::distance(label_start, label_query);

          sample_loss += labels.activations[label_index] *
                         std::log(outputs.activations[i] + EPS);
        }
      }
    }
  } else {
    // In case of sparse outputs, we iterate over labels below. If output
    // neuron corresponding to the label is found, then the value is computable.
    //
    // If not found, we assume output activations are 0, add an EPS to log (0 +
    // EPS) so the computed loss value doesn't go very high to create overflow.
    if (labels.isDense()) {
      // (Sparse Output, Dense Label)
      for (uint32_t i = 0; i < labels.len; i++) {
        const uint32_t* output_start = outputs.active_neurons;
        const uint32_t* output_end = outputs.active_neurons + outputs.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* output_query = std::find(output_start, output_end, i);

        if (output_query != output_end) {
          // Compute output_index to lookup the value from output
          // sparse-vector.
          size_t output_index = std::distance(output_start, output_query);

          sample_loss += labels.activations[i] *
                         std::log(outputs.activations[output_index] + EPS);
        } else {
          // Output activation is set to 0.0F. log(0) is -infinity, but for
          // reporting it suffices to show this value is really huge, so we use
          // EPS to get something like -7 from 1e-7.
          float output_activation = 0.0F;
          sample_loss +=
              labels.activations[i] * std::log(output_activation + EPS);
        }
      }
    } else {
      // We iterate over labels with non-zero activations. 0*log(x) = 0, so we
      // can ignore these terms.
      for (uint32_t i = 0; i < labels.len; i++) {
        const uint32_t* output_start = outputs.active_neurons;
        const uint32_t* output_end = outputs.active_neurons + outputs.len;

        // Find the position of the active neuron if it exists in the labels.
        const uint32_t* output_query =
            std::find(output_start, output_end, labels.active_neurons[i]);

        if (output_query != output_end) {
          // Compute output_index to lookup the value from outputs
          // sparse-vector.
          size_t output_index = std::distance(output_start, output_query);

          sample_loss += labels.activations[i] *
                         std::log(outputs.activations[output_index] + EPS);
        } else {
          // Output activation is set to 0.0F. log(0) is -infinity, but for
          // reporting it suffices to show this value is really huge, so we use
          // EPS to get something like -7 from 1e-7.
          float output_activation = 0.0F;
          sample_loss +=
              labels.activations[i] * std::log(output_activation + EPS);
        }
      }
    }
  }

  MetricUtilities::incrementAtomicFloat(_sum, -1 * sample_loss);
  _num_samples++;
}

bool FMeasure::isFMeasure(const std::string& name) {
  return std::regex_match(name,
                          std::regex(R"(f(\d+\.?\d*)?_measure\(0\.\d+\))"));
}

std::shared_ptr<Metric> FMeasure::make(const std::string& name) {
  if (!isFMeasure(name)) {
    std::stringstream error_ss;
    error_ss << "Invoked FMeasure::make with invalid string '" << name
             << "'. FMeasure::make should be invoked with a string "
                "in the format 'f_measure(threshold)', where "
                "threshold is a positive floating point number.";
    throw std::invalid_argument(error_ss.str());
  }

  std::string token = name.substr(name.find('('));  // token = (X.XXX)
  token = token.substr(1, token.length() - 2);      // token = X.XXX
  float threshold = std::stof(token);

  if (threshold <= 0) {
    std::stringstream error_ss;
    error_ss << "FMeasure invoked with threshold = " << threshold
             << ". The threshold should be greater than 0.";
    throw std::invalid_argument(error_ss.str());
  }

  float beta = 1.0;
  // Name is f<optional alpha>_(<threshold>)
  auto beta_end = name.find('_');
  auto beta_len = beta_end - 1;
  if (beta_len > 0) {
    beta = std::stof(name.substr(1, beta_len));

    if (beta < 0) {
      std::stringstream error_ss;
      error_ss << "FMeasure invoked with alpha = " << beta
               << ". The beta should be at least 0.";
      throw std::invalid_argument(error_ss.str());
    }
  }

  return std::make_shared<FMeasure>(threshold, beta);
}

std::shared_ptr<Metric> makeMetric(const std::string& name) {
  if (name == CategoricalAccuracy::NAME) {
    return std::make_shared<CategoricalAccuracy>();
  }
  if (name == WeightedMeanAbsolutePercentageError::NAME) {
    return std::make_shared<WeightedMeanAbsolutePercentageError>();
  }
  if (name == MeanSquaredErrorMetric::NAME) {
    return std::make_shared<MeanSquaredErrorMetric>();
  }
  if (FMeasure::isFMeasure(name)) {
    return FMeasure::make(name);
  }
  if (RecallAtK::isRecallAtK(name)) {
    return RecallAtK::make(name);
  }
  if (PrecisionAtK::isPrecisionAtK(name)) {
    return PrecisionAtK::make(name);
  }
  if (name == CategoricalCrossEntropy::NAME) {
    return std::make_shared<CategoricalCrossEntropy>();
  }
  throw std::invalid_argument("'" + name + "' is not a valid metric.");
}
}  // namespace thirdai::bolt_v1
