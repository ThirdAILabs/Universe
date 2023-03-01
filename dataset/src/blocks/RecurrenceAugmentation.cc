#include "RecurrenceAugmentation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::dataset {

RecurrenceAugmentation::RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                               char delimiter,
                                               uint32_t max_recurrence,
                                               uint32_t vocab_size,
                                               uint32_t input_vector_index,
                                               uint32_t label_vector_index)
    : _sequence_column(std::move(sequence_column)),
      _delimiter(delimiter),
      _max_recurrence(max_recurrence),
      _in_progress_vector_index(input_vector_index),
      _label_vector_index(label_vector_index),
      _vocab(vocab_size + 1, true) {}

Vectors RecurrenceAugmentation::augment(Vectors&& vectors,
                                        ColumnarInputSample& input_sample) {
  auto target_sequence = sequence(input_sample);

  Vectors augmented_vectors;
  augmented_vectors.reserve(vectors.size() * target_sequence.size());

  auto element_ids = elementIds(target_sequence);
  for (auto& sample_vectors : vectors) {
    auto augmentations = augmentEach(std::move(sample_vectors), element_ids);
    augmented_vectors.insert(augmented_vectors.end(),
                             std::make_move_iterator(augmentations.begin()),
                             std::make_move_iterator(augmentations.end()));
  }
  return augmented_vectors;
}

uint32_t RecurrenceAugmentation::expectedNumColumns() const {
  if (!_sequence_column.hasNumber()) {
    return 0;
  }
  return _sequence_column.number() + 1;
}

uint32_t RecurrenceAugmentation::elementIdAtStep(const BoltVector& output,
                                                 uint32_t step) {
  if (!output.isDense()) {
    throw std::invalid_argument(
        "Cannot get sequence element name from dense output");
  }
  auto begin = step * _vocab.vocabSize();
  auto end = begin + _vocab.vocabSize();

  uint32_t arg_max = 0;
  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t neuron = begin; neuron < end; neuron++) {
    if (output.activations[neuron] > max_act) {
      arg_max = neuron;
      max_act = output.activations[neuron];
    }
  }

  return arg_max;
}

std::string RecurrenceAugmentation::elementString(uint32_t element_id) {
  uint32_t element_id_without_position = element_id % _vocab.vocabSize();
  return _vocab.getString(element_id_without_position);
}

bool RecurrenceAugmentation::isEOS(uint32_t element_id) {
  return elementString(element_id) == EOS;
}

std::vector<std::string_view> RecurrenceAugmentation::sequence(
    ColumnarInputSample& input_sample) const {
  auto sequence =
      text::split(input_sample.column(_sequence_column), _delimiter);
  if (sequence.size() < _max_recurrence) {
    sequence.push_back(std::string_view(EOS, EOS_SIZE));
  }

  if (sequence.size() > _max_recurrence) {
    sequence.resize(_max_recurrence);
  }
  return sequence;
}

std::vector<uint32_t> RecurrenceAugmentation::elementIds(
    const std::vector<std::string_view>& sequence) {
  std::vector<uint32_t> element_ids(sequence.size());
  for (uint32_t i = 0; i < element_ids.size(); i++) {
    uint32_t offset = i * _vocab.vocabSize();
    element_ids[i] = offset + _vocab.getUid(std::string(sequence[i]));
  }
  return element_ids;
}

Vectors RecurrenceAugmentation::augmentEach(
    SampleVector&& vectors, const std::vector<uint32_t>& element_ids) const {
  auto augmented_vectors =
      multiply(std::move(vectors), /* times= */ element_ids.size());

  for (uint32_t step = 0; step < augmented_vectors.size(); step++) {
    addInProgressFeatures(augmented_vectors[step], element_ids, step);
    addLabelFeatures(augmented_vectors[step], element_ids, step);
  }
  return augmented_vectors;
}

Vectors RecurrenceAugmentation::multiply(SampleVector&& vectors,
                                         uint32_t times) {
  Vectors multiplied(times);
  for (uint32_t i = 0; i < times - 1; i++) {
    multiplied[i] = clone(vectors);
  }
  multiplied.back() = std::move(vectors);
  return multiplied;
}

SampleVector RecurrenceAugmentation::clone(const SampleVector& vectors) {
  SampleVector cloned(vectors.size());
  for (uint32_t i = 0; i < cloned.size(); i++) {
    cloned[i] = vectors[i]->clone();
  }
  return cloned;
}

void RecurrenceAugmentation::addInProgressFeatures(
    SampleVector& vectors, const std::vector<uint32_t>& element_ids,
    uint32_t step) const {
  assert(step < element_ids.size());
  vectors.at(_in_progress_vector_index)
      ->addFeatureSegment(_vocab.vocabSize() * _max_recurrence);
  for (uint32_t i = 0; i < step; i++) {
    vectors.at(_in_progress_vector_index)
        ->addSparseFeatureToSegment(/* index= */ element_ids[i],
                                    /* value= */ 1.0);
  }
}

void RecurrenceAugmentation::addLabelFeatures(
    SampleVector& vectors, const std::vector<uint32_t>& element_ids,
    uint32_t step) const {
  assert(step < element_ids.size());
  auto& ref = vectors.at(_label_vector_index);
  ref->addFeatureSegment(_vocab.vocabSize() * _max_recurrence);
  ref->addSparseFeatureToSegment(/* index= */ element_ids[step],
                                 /* value= */ 1.0);
}

template void RecurrenceAugmentation::serialize(cereal::BinaryInputArchive&);
template void RecurrenceAugmentation::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void RecurrenceAugmentation::serialize(Archive& archive) {
  archive(cereal::base_class<Augmentation>(this), _sequence_column, _delimiter,
          _max_recurrence, _in_progress_vector_index, _label_vector_index,
          _vocab);
}
}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::RecurrenceAugmentation)