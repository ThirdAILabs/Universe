#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace thirdai::search {

using NumpyArray = pybind11::array_t<float, pybind11::array::c_style |
                                                pybind11::array::forcecast>;

using SeqResult = std::pair<std::vector<uint32_t>, float>;

std::vector<std::vector<SeqResult>> beamSearchBatch(
    const NumpyArray& probabilities, const NumpyArray& transition_matrix,
    uint32_t k);

}  // namespace thirdai::search