#include <cereal/archives/binary.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/DWTA.h>
#include <hashing/src/MinHash.h>
#include <hashtable/src/SampledHashTable.h>
#include <hashtable/src/VectorHashTable.h>
#include <dataset/src/InMemoryDataset.h>
#include <proto/hashing.pb.h>
#include <proto/udt_query_reformulation.pb.h>
#include <search/src/Flash.h>
#include <utils/Logging.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace thirdai::search {

Flash::Flash(std::shared_ptr<hashing::HashFunction> hash_function)
    : _hash_function(std::move(hash_function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _total_samples_indexed(0),
      _hashtable(
          std::make_shared<hashtable::VectorHashTable>(_num_tables, _range)) {
  thirdai::licensing::checkLicense();
}

Flash::Flash(std::shared_ptr<hashing::HashFunction> hash_function,
             uint32_t reservoir_size)
    : _hash_function(std::move(hash_function)),
      _num_tables(_hash_function->numTables()),
      _range(_hash_function->range()),
      _total_samples_indexed(0),
      _hashtable(std::make_shared<hashtable::VectorHashTable>(
          _num_tables, _range, reservoir_size)) {
  thirdai::licensing::checkLicense();
}

Flash::Flash(const proto::udt::Flash& flash)
    : _num_tables(flash.hash_table().num_tables()),
      _range(flash.hash_table().table_range()),
      _total_samples_indexed(flash.total_samples_indexed()) {
  thirdai::licensing::checkLicense();

  _hashtable = std::make_shared<hashtable::VectorHashTable>(flash.hash_table());

  switch (flash.hash_function().type_case()) {
    case proto::hashing::HashFunction::kDwta:
      _hash_function = std::make_shared<hashing::DWTAHashFunction>(
          flash.hash_function().dwta());
      break;
    case proto::hashing::HashFunction::kMinhash:
      _hash_function =
          std::make_shared<hashing::MinHash>(flash.hash_function().minhash());
      break;
    default:
      throw std::invalid_argument("Invalid HashFunction in fromProto.");
  }
}

void Flash::addBatch(const BoltBatch& batch,
                     const std::vector<uint32_t>& labels,
                     licensing::TrainPermissionsToken token) {
  _total_samples_indexed += batch.getBatchSize();
  licensing::entitlements().verifyAllowedNumberOfTrainingSamples(
      _total_samples_indexed);

  // A token can only be constructed if the user has a full access
  // license or is using a dataset that is allowed under their demo license.
  // Hence this method successfully being called with a token is enough to
  // continue, and we don't actually need to use the token object.
  (void)token;

  if (batch.getBatchSize() != labels.size()) {
    throw std::invalid_argument("Batch size and number of labels must match.");
  }

  std::vector<uint32_t> hashes = hashBatch(batch);

  assert(hashes.size() == batch.getBatchSize() * _num_tables);

  _hashtable->insert(batch.getBatchSize(), labels.data(), hashes.data());
}

std::vector<uint32_t> Flash::hashBatch(const BoltBatch& batch) const {
  auto hashes = _hash_function->hashBatchParallel(batch);
  return hashes;
}

std::pair<std::vector<std::vector<uint32_t>>, std::vector<std::vector<float>>>
Flash::queryBatch(const BoltBatch& batch, uint32_t top_k,
                  bool pad_zeros) const {
  std::vector<std::vector<uint32_t>> results(batch.getBatchSize());
  std::vector<std::vector<float>> scores(batch.getBatchSize());
  auto hashes = hashBatch(batch);

#pragma omp parallel for default(none) \
    shared(batch, top_k, results, scores, hashes, pad_zeros)
  for (uint64_t vec_id = 0; vec_id < batch.getBatchSize(); vec_id++) {
    std::vector<uint32_t> query_result;
    _hashtable->queryByVector(hashes.data() + vec_id * _num_tables,
                              query_result);

    auto [result, score] = getTopKUsingPriorityQueue(query_result, top_k);
    results.at(vec_id) = result;
    scores.at(vec_id) = score;
    if (pad_zeros) {
      while (results.at(vec_id).size() < top_k) {
        results.at(vec_id).push_back(0);
      }
      while (scores.at(vec_id).size() < top_k) {
        scores.at(vec_id).push_back(0);
      }
    }
  }

  return {results, scores};
}

std::pair<std::vector<uint32_t>, std::vector<float>>
Flash::getTopKUsingPriorityQueue(std::vector<uint32_t>& query_result,
                                 uint32_t top_k) const {
  // We sort so counting is easy
  std::sort(query_result.begin(), query_result.end());

  std::priority_queue<std::pair<int32_t, uint32_t>,
                      std::vector<std::pair<int32_t, uint32_t>>,
                      std::greater<std::pair<int32_t, uint32_t>>>
      queue;

  if (!query_result.empty()) {
    uint64_t current_element = query_result.at(0);
    uint32_t current_element_count = 0;
    for (auto element : query_result) {
      if (element == current_element) {
        current_element_count++;
      } else {
        queue.emplace(current_element_count, current_element);
        if (queue.size() > top_k) {
          queue.pop();
        }
        current_element = element;
        current_element_count = 1;
      }
    }
    queue.emplace(current_element_count, current_element);
    if (queue.size() > top_k) {
      queue.pop();
    }
  }

  // Create and save results, scores vector

  std::vector<uint32_t> result;
  std::vector<float> scores;
  while (!queue.empty()) {
    result.push_back(queue.top().second);
    scores.push_back(static_cast<float>(queue.top().first) / _num_tables);
    queue.pop();
  }
  std::reverse(result.begin(), result.end());
  std::reverse(scores.begin(), scores.end());
  return {result, scores};
}

proto::udt::Flash* Flash::toProto() const {
  auto* flash = new proto::udt::Flash();

  flash->set_allocated_hash_function(_hash_function->toProto());

  flash->set_allocated_hash_table(_hashtable->toProto());

  flash->set_total_samples_indexed(_total_samples_indexed);

  return flash;
}

template void Flash::serialize(cereal::BinaryInputArchive&);
template void Flash::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void Flash::serialize(Archive& archive) {
  licensing::entitlements().verifySaveLoad();
  archive(_hash_function, _num_tables, _range, _hashtable,
          _total_samples_indexed);
}

}  // namespace thirdai::search
