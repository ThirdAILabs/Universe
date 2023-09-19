#pragma once

#include <utils/ProtobufIO.h>
#include <string>
#include <unordered_map>

namespace thirdai::bolt {

using SerializableParameters =
    std::vector<std::pair<std::string, const std::vector<float>*>>;

void serialize(const SerializableParameters& to_serialize,
               utils::ProtobufWriter& object_writer);

using DeserializedParameters =
    std::unordered_map<std::string, std::vector<float>>;

DeserializedParameters deserialize(utils::ProtobufReader& object_reader);

}  // namespace thirdai::bolt