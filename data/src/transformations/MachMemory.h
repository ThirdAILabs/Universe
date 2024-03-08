#pragma once

#include <cereal/access.hpp>
#include <data/src/ColumnMap.h>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::data {

struct MachSample {
  std::vector<uint32_t> input_indices;
  std::vector<float> input_values;
  std::vector<uint32_t> mach_buckets;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

class MachMemory {
 public:
  MachMemory(std::string input_indices_column, std::string input_values_column,
             std::string id_column, std::string mach_buckets_column,
             size_t max_ids, size_t max_samples_per_id)
      : _input_indices_column(std::move(input_indices_column)),
        _input_values_column(std::move(input_values_column)),
        _id_column(std::move(id_column)),
        _mach_buckets_column(std::move(mach_buckets_column)),
        _max_ids(max_ids),
        _max_samples_per_id(max_samples_per_id),
        _rng(RNG_SEED) {}

  static auto make(std::string input_indices_column,
                   std::string input_values_column, std::string id_column,
                   std::string mach_buckets_column, size_t max_ids,
                   size_t max_samples_per_doc) {
    return std::make_shared<MachMemory>(
        std::move(input_indices_column), std::move(input_values_column),
        std::move(id_column), std::move(mach_buckets_column), max_ids,
        max_samples_per_doc);
  }

  std::optional<data::ColumnMap> getSamples(size_t num_samples);

  void addSamples(const data::ColumnMap& columns);

  void addSample(uint32_t id, MachSample sample);

  void clear() {
    _id_to_samples = {};
    _ids = {};
  }

  void remove(uint32_t id) {
    _id_to_samples.erase(id);
    _ids.erase(id);
  }

 private:
  static constexpr uint32_t RNG_SEED = 7240924;

  std::string _input_indices_column;
  std::string _input_values_column;
  std::string _id_column;
  std::string _mach_buckets_column;
  std::optional<size_t> _input_indices_dim;
  std::optional<size_t> _num_mach_buckets;

  std::unordered_map<uint32_t, std::vector<MachSample>> _id_to_samples;
  std::unordered_set<uint32_t> _ids;

  size_t _max_ids;
  size_t _max_samples_per_id;

  std::mt19937 _rng{RNG_SEED};
};

using MachMemoryPtr = std::shared_ptr<MachMemory>;

}  // namespace thirdai::data