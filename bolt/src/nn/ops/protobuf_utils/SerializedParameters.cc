#include "SerializedParameters.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <proto/parameter.pb.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::bolt {

inline size_t nShards(const std::vector<float>* parameters, size_t shard_size) {
  return (parameters->size() + shard_size - 1) / shard_size;
}

size_t ParameterToShards::computeTotalShards(
    const SerializableParameters& to_serialize) {
  size_t total_shards = 0;
  for (const auto& [_, parameters] : to_serialize) {
    total_shards += nShards(parameters, SHARD_SIZE);
  }
  return total_shards;
}

void ParameterToShards::serializeParameters(
    const SerializableParameters& to_serialize,
    utils::ProtobufWriter& object_writer) {
  std::unordered_set<std::string> parameter_names;

  for (const auto& [name, parameters] : to_serialize) {
    if (parameter_names.count(name)) {
      throw std::invalid_argument(
          "Cannot serialize multiple parameters with the same name. Found "
          "multiple parameters named '" +
          name + "'.");
    }
    parameter_names.insert(name);

    size_t n_shards = nShards(parameters, SHARD_SIZE);

    for (size_t shard_id = 0; shard_id < n_shards; shard_id++) {
      size_t start = shard_id * SHARD_SIZE;
      size_t end = std::min(start + SHARD_SIZE, parameters->size());

      proto::bolt::ParameterShard shard;

      shard.set_name(name);

      shard.set_num_shards(n_shards);
      shard.set_shard_id(shard_id);

      shard.set_parameter_size(parameters->size());
      shard.set_offset(start);
      shard.set_shard_size(end - start);

      *shard.mutable_data() = {parameters->data() + start,
                               parameters->data() + end};

      object_writer.serialize(shard);
    }
  }
}

struct InProgressParameter {
  std::vector<float> data;
  std::vector<bool> shards_seen;
};

DeserializedParameters parametersFromShards(
    utils::ProtobufReader& object_reader, size_t n_shards) {
  std::unordered_map<std::string, InProgressParameter> parameters;

  for (size_t i = 0; i < n_shards; i++) {
    proto::bolt::ParameterShard shard;
    object_reader.deserialize(shard);

    if (!parameters.count(shard.name())) {
      parameters[shard.name()].data.assign(shard.parameter_size(), 0.0);
      parameters[shard.name()].shards_seen.assign(shard.num_shards(), false);
    }

    if (parameters.count(shard.name())) {
      InProgressParameter& in_progress_param = parameters[shard.name()];
      if (shard.num_shards() != in_progress_param.shards_seen.size()) {
        throw std::runtime_error("Mismatch in num shards for parameter.");
      }
      if (in_progress_param.shards_seen[shard.shard_id()]) {
        throw std::invalid_argument(
            "Duplicate shard ids encounter for the same parameter.");
      }

      if (shard.parameter_size() != in_progress_param.data.size()) {
        throw std::runtime_error("Mismatch in shard parameter sizes.");
      }

      if (shard.offset() + shard.shard_size() > in_progress_param.data.size()) {
        throw std::runtime_error("Invalid parameter shard offset and size.");
      }

      std::copy(shard.data().begin(), shard.data().end(),
                in_progress_param.data.begin() + shard.offset());

      in_progress_param.shards_seen[shard.shard_id()] = true;
    }
  }

  DeserializedParameters deserialized;
  for (auto& [name, params] : parameters) {
    if (!std::all_of(params.shards_seen.begin(), params.shards_seen.end(),
                     [](bool x) { return x; })) {
      throw std::invalid_argument(
          "Error loading model: missing shards for some parameters.");
    }
    deserialized[name] = std::move(params.data);
  }

  return deserialized;
}

}  // namespace thirdai::bolt