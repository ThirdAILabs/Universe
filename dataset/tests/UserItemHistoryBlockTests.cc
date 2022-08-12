#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/UserItemHistory.h>
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

std::vector<uint32_t> makeShuffledUserIdSequence(size_t n_users, size_t n_items) {
  std::vector<uint32_t> user_seq(n_users * n_items);
  for (uint32_t i = 0; i < user_seq.size(); i++) {
    user_seq[i] = i / n_items;
  }

  auto rng = std::default_random_engine{};
  std::shuffle(user_seq.begin(), user_seq.end(), rng);
  return user_seq;
}

std::vector<uint32_t> makeItemIdSequence(
    std::vector<uint32_t>& user_id_sequence, uint32_t n_users,
    uint32_t n_items) {
  std::vector<uint32_t> items_generated_for_user(n_users);
  std::vector<uint32_t> item_id_sequence;
  for (auto user_id : user_id_sequence) {
    /*
      To easily check the validity of the item history:
      1) Item IDs are ordered (Easily check that an item in the history came
      before the current item) 2) Item IDs are disjoint for each user (Easily
      check whether the history is corrupted)
    */
    item_id_sequence.push_back(user_id * n_items +
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

std::shared_ptr<std::unordered_map<std::string, uint32_t>> makeIdMap(
    std::vector<uint32_t>& id_sequence) {
  std::unordered_map<std::string, uint32_t> map;
  for (auto id : id_sequence) {
    std::stringstream ss;
    ss << id;
    map[ss.str()] = id;
  }
  return std::make_shared<std::unordered_map<std::string, uint32_t>>(
      std::move(map));
}

void assertItemHistoryValid(bolt::BoltBatch& batch,
                            std::vector<uint32_t>& user_id_sequence,
                            std::vector<uint32_t>& item_id_sequence,
                            uint32_t n_items) {
  for (uint32_t idx = 0; idx < batch.getBatchSize(); idx++) {
    for (uint32_t pos = 0; pos < batch[idx].len; pos++) {
      auto active_neuron = batch[idx].active_neurons[pos];
      // in user's range; not corrupted
      ASSERT_GE(active_neuron, user_id_sequence[idx] * n_items);
      ASSERT_LT(active_neuron, (user_id_sequence[idx] + 1) * n_items);
      // does not contain item IDs from the future.
      ASSERT_LT(active_neuron, item_id_sequence[idx]);
    }
  }
}

void assertItemHistoryNotEmpty(bolt::BoltBatch& batch) {
  uint32_t total_entries = 0;
  for (uint32_t idx = 0; idx < batch.getBatchSize(); idx++) {
    total_entries += batch[idx].len;
  }
  ASSERT_GT(total_entries, 0);
};

bolt::BoltBatch processSamples(std::vector<std::string>& samples, std::vector<uint32_t>& user_id_seq, std::vector<uint32_t>& item_id_seq, uint32_t track_last_n) {
  auto user_id_map = makeIdMap(user_id_seq);
  auto item_id_map = makeIdMap(item_id_seq);
  auto records = UserItemHistoryBlock::makeEmptyRecord(user_id_map->size(), track_last_n);

  auto user_item_history_block = std::make_shared<UserItemHistoryBlock>(
          /* user_col = */ 0, /* item_col = */ 1, /* timestamp_col = */ 2,
          track_last_n, user_id_map, item_id_map, records);

  GenericBatchProcessor processor(
      /* input_blocks = */ {user_item_history_block},
      /* label_blocks = */ {});
  
  auto [batch, _] = processor.createBatch(samples);
  return std::move(batch);
}

void assertItemHistoryNotStagnant(bolt::BoltBatch& batch,
                                  std::vector<uint32_t>& user_id_sequence,
                                  uint32_t n_users) {
  std::vector<std::unordered_set<uint32_t>> last_user_item_history(n_users);
  std::vector<bool> user_history_changes(n_users);

  for (uint32_t idx = 0; idx < batch.getBatchSize(); idx++) {
    auto user_id = user_id_sequence[idx];
    std::unordered_set<uint32_t> current_history;

    for (uint32_t pos = 0; pos < batch[idx].len; pos++) {
      current_history.insert(batch[idx].active_neurons[pos]);
    }

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
  uint32_t n_items = 300;
  uint32_t track_last_n = 10;
  
  auto user_id_seq = makeShuffledUserIdSequence(n_users, n_items);
  auto item_id_seq = makeItemIdSequence(user_id_seq, n_users, n_items);
  auto samples = makeSamples(user_id_seq, item_id_seq);

  auto batch = processSamples(samples, user_id_seq, item_id_seq, track_last_n);
  assertItemHistoryValid(batch, user_id_seq, item_id_seq, n_items);
  assertItemHistoryNotStagnant(batch, user_id_seq, n_users);
  assertItemHistoryNotEmpty(batch);
}
}  // namespace thirdai::dataset
