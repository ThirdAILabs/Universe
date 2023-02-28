#include "RecurrenceAugmentation.h"

namespace thirdai::dataset {

RecurrenceAugmentation::RecurrenceAugmentation(ColumnIdentifier sequence_column,
                                               uint32_t max_recurrence,
                                               uint32_t vocab_size,
                                               uint32_t input_vector_index,
                                               uint32_t label_vector_index)
    : _sequence_column(std::move(sequence_column)),
      _max_recurrence(max_recurrence),
      _vocab_size_with_eos(vocab_size + 1),
      _in_progress_vector_index(input_vector_index),
      _label_vector_index(label_vector_index),
      _vocab(_vocab_size_with_eos, true) {}

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

std::vector<std::string_view> RecurrenceAugmentation::sequence(
    ColumnarInputSample& input_sample) const {
  auto sequence = text::split(input_sample.column(_sequence_column));
  if (sequence.size() < _max_recurrence) {
    sequence.push_back(std::string_view(EOS, EOS_SIZE));
  }
  return sequence;
}

std::vector<uint32_t> RecurrenceAugmentation::elementIds(
    const std::vector<std::string_view>& sequence) {
  std::vector<uint32_t> element_ids(sequence.size());
  for (uint32_t i = 0; i < element_ids.size(); i++) {
    uint32_t offset = i * _vocab_size_with_eos;
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
      ->addFeatureSegment(_vocab_size_with_eos * _max_recurrence);
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
  ref->addFeatureSegment(_vocab_size_with_eos * _max_recurrence);
  ref->addSparseFeatureToSegment(/* index= */ element_ids[step],
                                 /* value= */ 1.0);

}

}  // namespace thirdai::dataset