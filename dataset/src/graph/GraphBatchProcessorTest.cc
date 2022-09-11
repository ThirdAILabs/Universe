#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/graph/BatchProcessor.h>
#include <dataset/src/graph/Node.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <memory>

namespace thirdai::dataset::graph {

TEST(GraphBatchProcessorTest, CompilesLol) {
  /*
    STATES / META
  */

  uint32_t n_users = 10000;
  uint32_t n_items = 10000;
  std::vector<uint32_t> history_lengths = {1, 2, 5, 10, 20, 50};
  auto histories = ItemHistoryCollection::make(n_users, 50);

  /*
    DEFINE BATCH PROCESSOR BELOW
  */

  GraphBatchProcessor proc([&](const StringInputPtr& input) {
    auto row = CsvRow::make(input);

    auto user_id = SingleStringLookup::make(row->at(0), n_users);
    auto item_id = SingleStringLookup::make(row->at(1), n_items);
    auto timestamp_seconds = TimeStringToSeconds::make(row->at(2));

    std::vector<VectorProducerNodePtr> vector_segments = {
        VectorFromTokens::make(user_id, n_users)};

    std::vector<NodePtr> update_history_wait_list;

    for (auto length : history_lengths) {
      auto history = HistoryLookup::make(user_id, histories, length);
      update_history_wait_list.push_back(history);
      vector_segments.push_back(VectorFromTokens::make(history, n_items));
    }

    auto update_history_side_effect =
        UpdateHistory::make(user_id, item_id, timestamp_seconds, histories);
    update_history_side_effect->waitFor(update_history_wait_list);

    return WorkflowOutput(
        /* input_vector= */ ConcatenateVectorSegments::make(vector_segments),
        /* label_vector= */ VectorFromTokens::make(item_id, n_items),
        /* side_effects= */ {update_history_side_effect});
  });

  /* SEE IF IT WORKS */

  auto [input, label] = proc.createBatch({
      "0,0,2022-09-10",
      "0,0,2022-09-11",
      "0,0,2022-09-12",
  });
  std::cout << input[0] << std::endl;
  std::cout << label[0] << std::endl;
  std::cout << input[2] << std::endl;
  std::cout << label[2] << std::endl;
}

}  // namespace thirdai::dataset::graph