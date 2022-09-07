#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/UserItemHistory.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
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

void assertItemHistoryValid(std::vector<std::vector<uint32_t>>& batch,
                            std::vector<uint32_t>& user_id_sequence,
                            std::vector<uint32_t>& item_id_sequence,
                            uint32_t n_items) {
  for (uint32_t idx = 0; idx < batch.size(); idx++) {
    for (auto item_id : batch[idx]) {
      // in user's range; not corrupted
      ASSERT_GE(item_id, user_id_sequence[idx] * n_items);
      ASSERT_LT(item_id, (user_id_sequence[idx] + 1) * n_items);
      // does not contain item IDs from the future.
      ASSERT_LT(item_id, item_id_sequence[idx]);
    }
  }
}

void assertItemHistoryNotEmpty(std::vector<std::vector<uint32_t>>& batch) {
  uint32_t total_entries = 0;
  for (auto& items : batch) {
    total_entries += items.size();
  }
  ASSERT_GT(total_entries, 0);
}

std::vector<std::vector<uint32_t>> processSamples(
    std::vector<std::string>& samples, uint32_t n_users,
    uint32_t n_items_per_user, uint32_t track_last_n) {
  auto user_id_lookup = ThreadSafeVocabulary::make(n_users);
  auto item_id_lookup = ThreadSafeVocabulary::make(n_users * n_items_per_user);

  auto records =
      ItemHistoryCollection::make(user_id_lookup->vocabSize(), track_last_n);

  auto user_item_history_block = UserItemHistoryBlock::make(
      /* user_col = */ 0, /* item_col = */ 1, /* timestamp_col = */ 2,
      user_id_lookup, item_id_lookup, records);

  GenericBatchProcessor processor(
      /* input_blocks = */ {user_item_history_block},
      /* label_blocks = */ {});

  auto [batch, _] = processor.createBatch(samples);

  std::vector<std::vector<uint32_t>> histories;
  for (const auto& vec : batch) {
    std::vector<uint32_t> items;
    for (uint32_t pos = 0; pos < vec.len; pos++) {
      auto encoded_item = vec.active_neurons[pos];
      auto original_item_id_str = item_id_lookup->getString(encoded_item);
      items.push_back(std::stoull(original_item_id_str));
    }
    histories.push_back(items);
  }

  return histories;
}

void assertItemHistoryGetsUpdated(std::vector<std::vector<uint32_t>>& batch,
                                  std::vector<uint32_t>& user_id_sequence,
                                  uint32_t n_users) {
  std::vector<std::unordered_set<uint32_t>> last_user_item_history(n_users);
  std::vector<bool> user_history_changes(n_users);

  for (uint32_t idx = 0; idx < batch.size(); idx++) {
    auto user_id = user_id_sequence[idx];
    std::unordered_set<uint32_t> current_history;

    current_history.insert(batch[idx].begin(), batch[idx].end());

    if (!last_user_item_history[user_id].empty() &&
        last_user_item_history[user_id] != current_history) {
      user_history_changes[user_id] = true;
    }

    last_user_item_history[user_id] = current_history;
  }

  for (const auto& changes : user_history_changes) {
    ASSERT_TRUE(changes);
  }
}

TEST(UserItemHistoryBlockTests, CorrectMultiThread) {
  uint32_t n_users = 120;
  uint32_t n_items_per_user = 300;
  uint32_t track_last_n = 10;

  auto user_id_seq = makeShuffledUserIdSequence(n_users, n_items_per_user);
  auto item_id_seq = makeItemIdSequence(user_id_seq, n_users, n_items_per_user);
  auto samples = makeSamples(user_id_seq, item_id_seq);

  auto batch = processSamples(samples, n_users, n_items_per_user, track_last_n);
  assertItemHistoryValid(batch, user_id_seq, item_id_seq, n_items_per_user);
  assertItemHistoryGetsUpdated(batch, user_id_seq, n_users);
  assertItemHistoryNotEmpty(batch);
}

TEST(UserItemHistoryBlockTests, CorrectMultiItem) {
  std::vector<std::string> samples = {{"user1,item1 item2 item3,2022-02-02"},
                                      {"user1,item4,2022-02-02"}};

  GenericBatchProcessor processor(
      /* input_blocks= */ {UserItemHistoryBlock::make(
          /* user_col= */ 0, /* item_col= */ 1, /* timestamp_col= */ 2,
          /* track_last_n= */ 3, /* n_unique_users= */ 1,
          /* n_unique_items= */ 4, /* item_col_delimiter= */ ' ')},
      /* label_blocks= */ {},
      /* has_header= */ false,
      /* delimiter= */ ',',
      /* parallel= */ false);

  auto [batch, _] = processor.createBatch(samples);

  ASSERT_EQ(batch[0].len, 1);

  std::unordered_set<uint32_t> active_neurons;
  for (uint32_t i = 0; i < batch[1].len; i++) {
    active_neurons.insert(batch[1].active_neurons[i]);
  }

  ASSERT_EQ(active_neurons.size(), 3);
}

}  // namespace thirdai::dataset
