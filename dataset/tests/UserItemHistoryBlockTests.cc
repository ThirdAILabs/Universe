#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TimeUtils.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset {

static constexpr uint32_t ITEM_HASH_RANGE = 100000;

class TimeGenerator {
 public:
  std::string nextTimeString() {
    uint32_t date = _days % 28 + 1;  // Every month has at least 28 days.
    uint32_t month = (_days / 28) % 12 + 1;
    uint32_t year = _days / (28 * 12) + 1970;
    _days++;

    std::stringstream ss;
    ss << year << "-" << zeroPad(month) << "-" << zeroPad(date);
    return ss.str();
  }

 private:
  static std::string zeroPad(uint32_t number) {
    std::stringstream ss;
    if (number < 10) {
      ss << "0";
    }
    ss << number;
    return ss.str();
  }
  uint32_t _days = 0;
};

std::vector<uint32_t> makeShuffledUserIdSequence(size_t n_users,
                                                 size_t n_items_per_user) {
  std::vector<uint32_t> user_seq(n_users * n_items_per_user);
  for (uint32_t i = 0; i < user_seq.size(); i++) {
    user_seq[i] = i / n_items_per_user;
  }

  auto rng = std::default_random_engine{};
  std::shuffle(user_seq.begin(), user_seq.end(), rng);
  return user_seq;
}

std::vector<uint32_t> makeItemIdSequence(
    std::vector<uint32_t>& user_id_sequence, uint32_t n_users,
    uint32_t n_items_per_user) {
  std::vector<uint32_t> items_generated_for_user(n_users, 0);
  std::vector<uint32_t> item_id_sequence;
  for (auto user_id : user_id_sequence) {
    /*
      To easily check the validity of the item history:
      1) Item IDs are ordered (Easily check that an item in the history came
      before the current item) 2) Item IDs are disjoint for each user (Easily
      check whether the history is corrupted)
    */
    item_id_sequence.push_back(user_id * n_items_per_user +
                               items_generated_for_user[user_id]);
    items_generated_for_user[user_id]++;
  }
  return item_id_sequence;
}

std::vector<std::string> makeSamples(std::vector<uint32_t>& user_id_sequence,
                                     std::vector<uint32_t>& item_id_sequence) {
  TimeGenerator time_generator;

  std::vector<std::string> samples(user_id_sequence.size());
  for (uint32_t i = 0; i < user_id_sequence.size(); i++) {
    std::stringstream ss;
    ss << user_id_sequence[i] << "," << item_id_sequence[i] << ","
       << time_generator.nextTimeString();
    samples[i] = ss.str();
  }

  return samples;
}

std::vector<BoltVector> processSamples(std::vector<std::string>& samples,
                                       uint32_t track_last_n, bool parallel) {
  auto records = ItemHistoryCollection::make();

  auto user_item_history_block = UserItemHistoryBlock::make(
      /* user_col = */ 0, /* item_col = */ 1, /* timestamp_col = */ 2, records,
      track_last_n, ITEM_HASH_RANGE);

  TabularFeaturizer processor(
      /* input_blocks = */ {user_item_history_block},
      /* label_blocks = */ {}, /* has_header= */ false, /* delimiter= */ ',',
      /* parallel= */ parallel);

  auto batch = processor.featurize(samples).at(0);

  return std::move(batch);
}

auto groupVectorsByUser(std::vector<BoltVector>&& batch,
                        const std::vector<uint32_t>& user_ids) {
  std::unordered_map<uint32_t, std::vector<BoltVector>> user_to_vectors;
  for (uint32_t i = 0; i < batch.size(); i++) {
    user_to_vectors[user_ids[i]].push_back(std::move(batch[i]));
  }
  return user_to_vectors;
}

auto vectorAsWeightedSet(const BoltVector& vector) {
  std::unordered_map<uint32_t, float> set;
  for (uint32_t pos = 0; pos < vector.len; pos++) {
    set[vector.active_neurons[pos]] += vector.activations[pos];
  }
  return set;
}

auto countElements(const std::unordered_map<uint32_t, float>& weighted_set) {
  float sum = 0;
  for (auto [elem, count] : weighted_set) {
    sum += count;
  }
  return sum;
}

/**
 * This test checks the correctness of vectors produced by
 * UserItemHistoryBlock in sequential execution.
 *
 * Suppose a user A interacts with item 1, then with item 2,
 * and finally with item 3. Since UserItemHistoryBlock should
 * encode a user's past interactions (up to a certain number of
 * them), we expect to get the following output vectors:
 * 1) a representation of an empty set.
 * 2) a representation of the set {item 1}
 * 3) a representation of the set {item 1, item 2}
 *
 * From this example, we can generalize that an output vector is
 * correct if:
 * 1) The number of elements encoded in the vector is equal to
 * the number of previous samples for the user.
 * 2) There is exactly one element from the next vector for the
 * user that is missing in the current vector.
 */
TEST(UserItemHistoryBlockTests, CorrectSingleThread) {
  int32_t n_users = 120;
  uint32_t n_items_per_user = 300;
  uint32_t track_last_n = 10;

  auto user_id_seq = makeShuffledUserIdSequence(n_users, n_items_per_user);
  auto item_id_seq = makeItemIdSequence(user_id_seq, n_users, n_items_per_user);
  auto samples = makeSamples(user_id_seq, item_id_seq);

  auto batch = processSamples(samples, track_last_n, /* parallel= */ false);
  auto user_to_vectors = groupVectorsByUser(std::move(batch), user_id_seq);

  for (const auto& [_, vectors] : user_to_vectors) {
    for (int i = 0; i < static_cast<int>(vectors.size()) - 1; i++) {
      auto current_elements = vectorAsWeightedSet(vectors[i]);
      auto next_elements = vectorAsWeightedSet(vectors[i + 1]);

      ASSERT_EQ(countElements(current_elements),
                std::min<int>(i, track_last_n));
      ASSERT_EQ(countElements(next_elements),
                std::min<int>(i + 1, track_last_n));

      // We consider the weighted set to handle hash collisions.
      uint32_t n_elems_only_in_b = 0;
      for (const auto& [elem, count] : next_elements) {
        n_elems_only_in_b += std::max<float>(count - current_elements[elem], 0);
      }
      ASSERT_NEAR(n_elems_only_in_b, 1.0, /* abs_error= */ 0.01);
    }
  }
}

/**
 * This test checks the correctness of vectors produced by
 * UserItemHistoryBlock in parallel execution.
 *
 * The output vectors will be different than sequential execution
 * because we can no longer guarantee that samples are processed
 * in the order that they appear in the dataset. However,
 * UserItemHistoryBlock guarantees that output vectors never
 * encode interactions that occur in the future; the timestamps
 * of the encoded interactions do not exceed the current sample's
 * timestamp.
 *
 * In our mock dataset, the number of items per user equals the maximum
 * number of tracked items, and the timestamp of each sample strictly
 * increases. Thus, we expect that the set of elements encoded in
 * each output vector is a subset of the sequential execution
 * counterpart.
 */
TEST(UserItemHistoryBlockTests, CorrectMultiThread) {
  uint32_t n_users = 120;
  uint32_t n_items_per_user = 50;
  uint32_t track_last_n = 50;

  auto user_id_seq = makeShuffledUserIdSequence(n_users, n_items_per_user);
  auto item_id_seq = makeItemIdSequence(user_id_seq, n_users, n_items_per_user);
  auto samples = makeSamples(user_id_seq, item_id_seq);

  auto sequential_batch =
      processSamples(samples, track_last_n, /* parallel= */ false);
  auto parallel_batch =
      processSamples(samples, track_last_n, /* parallel= */ true);

  ASSERT_EQ(sequential_batch.size(), parallel_batch.size());

  for (uint32_t i = 0; i < sequential_batch.size(); i++) {
    auto sequential_elements = vectorAsWeightedSet(sequential_batch[i]);
    auto parallel_elements = vectorAsWeightedSet(parallel_batch[i]);

    for (const auto& [elem, count] : parallel_elements) {
      ASSERT_LE(count, sequential_elements[elem]);
    }
  }
}

TEST(UserItemHistoryBlockTests, CorrectMultiItem) {
  std::vector<std::string> samples = {{"user1,item1 item2 item3,2022-02-02"},
                                      {"user1,item4,2022-02-02"}};

  auto records = ItemHistoryCollection::make();

  TabularFeaturizer processor(
      /* input_blocks= */ {UserItemHistoryBlock::make(
          /* user_col= */ 0, /* item_col= */ 1, /* timestamp_col= */ 2,
          /* records= */ records, /* track_last_n= */ 3,
          /* item_hash_range= */ ITEM_HASH_RANGE,
          /* should_update_history= */ true,
          /* include_current_row= */ false,
          /* item_col_delimiter= */ ' ')},
      /* label_blocks= */ {},
      /* has_header= */ false,
      /* delimiter= */ ',',
      /* parallel= */ false);

  auto batch = processor.featurize(samples).at(0);

  ASSERT_EQ(batch[0].len, 0);

  std::unordered_set<uint32_t> active_neurons;
  for (uint32_t i = 0; i < batch[1].len; i++) {
    active_neurons.insert(batch[1].active_neurons[i]);
  }

  ASSERT_EQ(active_neurons.size(), 3);
}

TEST(UserItemHistoryBlockTests, HandlesTimeLagProperly) {
  std::vector<std::string> samples = {
      {"user1,item1,2022-02-01"},
      {"user1,item2,2022-02-02"},
      {"user1,item3,2022-02-03"},
      {"user1,item4,2022-02-08"},  // this sample is not past the lag.
      {"user1,item5,2022-02-10"},
  };

  auto records = ItemHistoryCollection::make();

  TabularFeaturizer processor(
      /* input_blocks= */ {UserItemHistoryBlock::make(
          /* user_col= */ 0, /* item_col= */ 1, /* timestamp_col= */ 2,
          /* records= */ records, /* track_last_n= */ 3,
          /* item_hash_range= */ 10000,
          /* should_update_history= */ true,
          /* include_current_row= */ false,
          /* item_col_delimiter= */ ' ',
          /* time_lag= */ TimeObject::SECONDS_IN_DAY * 3)},
      /* label_blocks= */ {},
      /* has_header= */ false,
      /* delimiter= */ ',',
      /* parallel= */ false);

  auto batch = processor.featurize(samples).at(0);

  // This means the block tracks the last 3 beyond lag.
  ASSERT_EQ(batch[4].len, 3);
}

TabularFeaturizer makeItemHistoryFeaturizer(ItemHistoryCollectionPtr history,
                                            uint32_t track_last_n,
                                            bool should_update_history) {
  return {/* input_blocks= */ {UserItemHistoryBlock::make(
              /* user_col= */ 0, /* item_col= */ 1, /* timestamp_col= */ 2,
              std::move(history), track_last_n, ITEM_HASH_RANGE,
              should_update_history,
              /* include_current_row= */ true)},
          /* label_blocks= */ {},
          /* has_header= */ false,
          /* delimiter= */ ',',
          /* parallel= */ false};
}

TEST(UserItemHistoryBlockTests, HandlesNoUpdateCaseProperly) {
  std::vector<std::string> updating_samples = {
      {"user1,item1,2022-02-01"},
      {"user1,item2,2022-02-02"},
      {"user1,item3,2022-02-03"},
      {"user1,item4,2022-02-08"},
  };

  std::vector<std::string> non_updating_sample = {
      {"user1,item5,2022-02-10"},
  };

  auto history = ItemHistoryCollection::make();

  auto updating_processor =
      makeItemHistoryFeaturizer(history,
                                /* track_last_n= */ updating_samples.size() + 1,
                                /* should_update_history= */ true);
  updating_processor.featurize(updating_samples);

  std::vector<std::string> items_in_history;
  for (const auto& item_struct : history->at("user1")) {
    items_in_history.push_back(item_struct.item);
  }
  ASSERT_EQ(items_in_history.size(), updating_samples.size());

  auto non_updating_processor =
      makeItemHistoryFeaturizer(history,
                                /* track_last_n= */ updating_samples.size() + 1,
                                /* should_update_history= */ false);
  auto non_updating_batch =
      non_updating_processor.featurize(non_updating_sample);

  // Size is updating_samples.size() + 1 because it includes the non-updating
  // sample
  ASSERT_EQ(non_updating_batch.at(0)[0].len, updating_samples.size() + 1);

  std::vector<std::string> final_items_in_history;
  for (const auto& item_struct : history->at("user1")) {
    final_items_in_history.push_back(item_struct.item);
  }

  ASSERT_EQ(items_in_history, final_items_in_history);
}

}  // namespace thirdai::dataset
