#pragma once

#include <utils/ProtobufIO.h>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

using SerializableParameters =
    std::vector<std::pair<std::string, const std::vector<float>*>>;

struct ParameterToShards {
  static size_t computeTotalShards(const SerializableParameters& to_serialize);

  static void serializeParameters(const SerializableParameters& to_serialize,
                                  utils::ProtobufWriter& object_writer);

 private:
  /**
   * This is a private field in this struct so that only the serialization logic
   * can access SHARD_SIZE. This ensures that the deserialization logic is note
   * dependent on the SHARD_SIZE constant, thus ensuring that changes to the
   * constant won't break serialization.
   *
   * This is 1Gb. The maximum size of a protobuf object is 2Gb. This ensures we
   * stay under it.
   */
  static constexpr size_t SHARD_SIZE = static_cast<size_t>(1) << 30;
};

using DeserializedParameters =
    std::unordered_map<std::string, std::vector<float>>;

DeserializedParameters parametersFromShards(
    utils::ProtobufReader& object_reader, size_t n_shards);

}  // namespace thirdai::bolt