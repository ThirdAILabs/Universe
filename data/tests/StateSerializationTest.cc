#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <data/src/transformations/State.h>
#include <dataset/src/mach/MachIndex.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace thirdai::data {

bool operator==(const ItemRecord& a, const ItemRecord& b) {
  return (a.item == b.item) && (a.timestamp == b.timestamp);
}

}  // namespace thirdai::data

namespace thirdai::data::tests {

using VocabMap = std::unordered_map<std::string, uint32_t>;

void compareVocab(const VocabMap& expected,
                  std::optional<uint32_t> expected_max_size,
                  const ThreadSafeVocabularyPtr& vocab) {
  EXPECT_EQ(expected.size(), vocab->size());
  EXPECT_EQ(expected_max_size, vocab->maxSize());

  for (const auto& [word, id] : expected) {
    // Use getString instead of getUid so that it doesn't create a new uuid for
    // the word and have it accidentally match.
    ASSERT_EQ(vocab->getString(id), word);
  }
}

automl::GraphInfoPtr makeGraph() {
  auto graph = std::make_shared<automl::GraphInfo>(/* feature_dim= */ 3);

  graph->insertNode(10, {10.5, 20.5, 30.5}, {2, 4, 6, 8});
  graph->insertNode(100, {100.5, 200.5, 300.5}, {5, 2, 7});
  graph->insertNode(1000, {1000.5, 2000.5, 3000.5}, {73, 28, 54, 89, 46});

  return graph;
}

void compareGraphs(const automl::GraphInfoPtr& old_graph,
                   const automl::GraphInfoPtr& new_graph) {
  ASSERT_EQ(old_graph->featureDim(), new_graph->featureDim());
  ASSERT_EQ(old_graph->nodeFeatures(), new_graph->nodeFeatures());
  ASSERT_EQ(old_graph->neighbors(), new_graph->neighbors());
}

TEST(StateSerializationTest, StateIsMaintained) {
  std::unordered_map<uint32_t, std::vector<uint32_t>> entity_to_hashes = {
      {1, {2, 8, 3}}, {2, {0, 6, 9}}, {3, {4, 1, 7}}};

  auto mach_index = std::make_shared<dataset::mach::MachIndex>(
      entity_to_hashes, /* num_buckets= */ 10, /* num_hashes= */ 3);

  State state(mach_index, makeGraph());

  VocabMap vocab_a = {{"a", 1}, {"b", 0}, {"c", 2}};

  VocabMap vocab_b = {{"d", 0}, {"e", 2}, {"f", 1}, {"e", 3}};

  state.addVocab("vocab_a",
                 dataset::ThreadSafeVocabulary::make(VocabMap(vocab_a), 100));
  state.addVocab("vocab_b", dataset::ThreadSafeVocabulary::make(
                                VocabMap(vocab_b), std::nullopt));

  ItemHistoryTracker tracker_1{{{"user_1", {{10, 1001}, {12, 1002}}},
                                {"user_2", {{40, 1}, {50, 2}, {60, 3}}}},
                               100};

  ItemHistoryTracker tracker_2{
      {{"user_a", {{0, 10}}}, {"user_b", {{1, 2}, {3, 4}, {5, 6}, {7, 8}}}},
      42};

  state.getItemHistoryTracker("tracker_1") = tracker_1;
  state.getItemHistoryTracker("tracker_2") = tracker_2;

  std::stringstream buffer;
  ar::serialize(state.toArchive(), buffer);

  auto new_state = State::fromArchive(*ar::deserialize(buffer));

  ASSERT_EQ(new_state->machIndex()->numHashes(), 3);
  ASSERT_EQ(new_state->machIndex()->numBuckets(), 10);
  ASSERT_EQ(new_state->machIndex()->numEntities(), 3);
  ASSERT_EQ(new_state->machIndex()->entityToHashes(), entity_to_hashes);

  compareVocab(vocab_a, 100, new_state->getVocab("vocab_a"));
  compareVocab(vocab_b, std::nullopt, new_state->getVocab("vocab_b"));

  ASSERT_EQ(new_state->getItemHistoryTracker("tracker_1"), tracker_1);
  ASSERT_EQ(new_state->getItemHistoryTracker("tracker_1"), tracker_1);

  compareGraphs(state.graph(), new_state->graph());
}

}  // namespace thirdai::data::tests