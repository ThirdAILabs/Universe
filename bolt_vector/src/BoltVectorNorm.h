#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <utils/text/StringManipulation.h>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace thirdai {

enum class LPNorm {
  L1,
  Euclidean,
  LInfinity,
};

static std::string LPNormToStr(LPNorm norm) {
  switch (norm) {
    case LPNorm::L1:
      return "l-1";
    case LPNorm::Euclidean:
      return "euclidean";
    case LPNorm::LInfinity:
      return "l-infinity";
    default:
      throw std::logic_error("Invalid norm passed to the call to LPNormToStr");
  }
}

static LPNorm getNorm(const std::string& norm_order) {
  std::string lower_case_norm_order = text::lower(norm_order);
  if (lower_case_norm_order == "l-1") {
    return LPNorm::L1;
  }
  if (lower_case_norm_order == "euclidean") {
    return LPNorm::Euclidean;
  }
  if (lower_case_norm_order == "l-infinity") {
    return LPNorm::LInfinity;
  }
  throw std::invalid_argument("" + norm_order +
                              " is not a valid Norm. Valid LP norms include "
                              "l-1, euclidean norm and l-infinity.");
}

class BoltVectorNorm {
 public:
  static double norm(const BoltVector& bolt_vector, const LPNorm norm_order) {
    return computeNorm(/* activations= */ bolt_vector.activations,
                       /* len= */ bolt_vector.len,
                       /* norm_order= */ norm_order);
  }

  // We still need to determine the best way to compute the norm of the
  // difference of two vectors when one or both are sparse. One problem
  // that arises is that both vectors can have the same length but a different
  // number of active neurons when one(or both) of them is sparse. For now
  // we expect both vectors to be dense.
  template <bool FIRST_DENSE, bool SECOND_DENSE>
  static double norm(const BoltVector& first_vector,
                     const BoltVector& second_vector, const LPNorm norm_order) {
    if (first_vector.len != second_vector.len) {
      throw std::invalid_argument(
          "Invalid arguments for the call to VectorNorm::norm. BoltVectors "
          "have different lengths. The first vector has length " +
          std::to_string(first_vector.len) +
          " while the second vector has length " +
          std::to_string(second_vector.len) + ".");
    }
    if (!(FIRST_DENSE && SECOND_DENSE)) {
      throw std::invalid_argument(
          "Invalid arguments for the call to VectorNorm::norm. Computing "
          "the norm of the difference between two non-dense"
          " vectors is currently not supported.");
    }

    std::vector<float> vec_difference = {};

    for (uint32_t activation_index = 0; activation_index < first_vector.len;
         activation_index++) {
      vec_difference.push_back(first_vector.activations[activation_index] -
                               second_vector.activations[activation_index]);
    }

    return computeNorm(/* activations= */ vec_difference.data(),
                       /* len= */ first_vector.len,
                       /*norm_order= */ norm_order);
  }

 private:
  static double computeNorm(const float* activations, uint32_t len,
                            LPNorm norm_order) {
    double norm = 0.0;
    switch (norm_order) {
      case LPNorm::L1: {
        std::for_each(activations, activations + len,
                      [&norm](float activation) { norm += fabs(activation); });

        return norm;
      }
      case LPNorm::Euclidean: {
        std::for_each(
            activations, activations + len,
            [&norm](float activation) { norm += (activation * activation); });

        return sqrt(norm);
      }
      case LPNorm::LInfinity: {
        assert(activations != nullptr);
        norm = static_cast<double>(abs(*activations));
        std::for_each(
            activations, activations + len, [&norm](float activation) {
              norm =
                  std::max<double>(norm, fabs(static_cast<double>(activation)));
            });

        return norm;
      }
      default: {
        throw std::invalid_argument(
            "" + LPNormToStr(norm_order) +
            " is not a valid LP Norm. Valid norms include l-1, "
            "euclidean and l-infinity.");
      }
    }
  }
};

}  // namespace thirdai