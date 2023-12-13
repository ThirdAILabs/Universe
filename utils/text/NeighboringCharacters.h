#pragma once

#include <unordered_map>
#include <vector>

namespace thirdai::text {

const std::unordered_map<char, std::vector<char>> keyboard_char_neighbors = {
    {'a', {'q', 'w', 's', 'z'}},
    {'b', {'v', 'g', 'h', 'n'}},
    {'c', {'x', 'd', 'f', 'v'}},
    {'d', {'e', 'r', 'f', 'c', 'x', 's'}},
    {'e', {'w', 's', 'd', 'r'}},
    {'f', {'r', 't', 'g', 'v', 'c', 'd'}},
    {'g', {'t', 'y', 'h', 'b', 'v', 'f'}},
    {'h', {'y', 'u', 'j', 'n', 'b', 'g'}},
    {'i', {'u', 'j', 'k', 'o'}},
    {'j', {'u', 'i', 'k', 'm', 'n', 'h'}},
    {'k', {'i', 'o', 'l', 'm', 'j'}},
    {'l', {'o', 'p', 'k'}},
    {'m', {'n', 'j', 'k'}},
    {'n', {'b', 'h', 'j', 'm'}},
    {'o', {'i', 'k', 'l', 'p'}},
    {'p', {'o', 'l'}},
    {'q', {'w', 'a'}},
    {'r', {'e', 'd', 'f', 't'}},
    {'s', {'w', 'e', 'd', 'x', 'z', 'a'}},
    {'t', {'r', 'f', 'g', 'y'}},
    {'u', {'y', 'h', 'j', 'i'}},
    {'v', {'c', 'f', 'g', 'b'}},
    {'w', {'q', 'a', 's', 'e'}},
    {'x', {'z', 's', 'd', 'c'}},
    {'y', {'t', 'g', 'h', 'u'}},
    {'z', {'a', 's', 'x'}}};

}  // namespace thirdai::text
