#pragma once

#include "ArgumentMap.h"
#include <bolt/src/nn/model/Model.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

namespace thirdai::automl::config {

/**
 * Returns a model from the given config. The argument 'user_input' gives the
 * parameters or options specified by the user. The argument 'input_dims' gives
 * the dimensions of each input to the model.
 *
 * The model config must be in the following form:
 *
 * {
 *    "inputs" ["input_1", "input_2", ...],
 *    "nodes": [
 *        {
 *            "name": <node name>,
 *            "type": <node type>,
 *            "other_params": ...,
 *            "predecessor(s)": ... // any inputs to the node
 *        },
 *        ...
 *    ],
 *    "output": <name of output node>,
 *    "loss": <name of loss function to use>
 * }
 *
 * The inputs are a array of names that correspond to each provided input dim,
 * i.e. the ith input name correponds to an input to the model that has the ith
 * input dim. Nodes can use these names to reference the inputs.
 *
 * The predecessors of a given node must be specified in the nodes list before
 * the the node that uses them as inputs.
 */
bolt::ModelPtr buildModel(const json& config, const ArgumentMap& args,
                          const std::vector<uint32_t>& input_dims,
                          bool mach = false);

/**
 * Takes in a config as a json string and encrypts it using a per byte cipher
 * and writes the resulting config to a file. Takes in a string rather than a
 * json object so that configs can be created in python, converted to a string
 * and then passed to this function.
 */
void dumpConfig(const std::string& config, const std::string& filename);

/**
 * Loads a config from a file and decrypts it, returns the resulting json as a
 * string. Returns a string rather than a parsed json object so that this
 * function can by used in python to view configs.
 */
std::string loadConfig(const std::string& filename);

}  // namespace thirdai::automl::config