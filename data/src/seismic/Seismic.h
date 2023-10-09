#pragma once

#include <string>
#include <vector>

namespace thirdai::data {

std::vector<uint32_t> seismicLabels(const std::string& trace, size_t x_coord,
                                    size_t y_coord, size_t z_coord,
                                    size_t subcube_dim, size_t label_cube_dim,
                                    size_t max_label);

}  // namespace thirdai::data