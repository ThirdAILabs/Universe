#pragma once

#include <bolt/src/graph/Node.h>
#include <bolt/src/layers/BoltVector.h>
#include <numeric>
#include <stdexcept>
#include <string>

namespace thirdai::bolt {

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
  }
  throw std::logic_error("Invalid norm passed to the call to LPNormToStr");
}

static LPNorm getNorm(const std::string& norm_order) {
  std::string lower_case_norm_order;
  for (char c : norm_order) {
    lower_case_norm_order.push_back(std::tolower(c));
  }

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
                              "L-1 norm, Euclidean norm and L-infinity norm.");
}

class NodeProperties {
 public:
  static double norm(const BoltVector& bolt_vector,
                     const std::string& norm_order) {
    LPNorm norm = getNorm(norm_order);
    return computeNorm(/* activations= */ bolt_vector.activations,
                       /* len= */ bolt_vector.len, /* norm= */ norm);
  }

  static double norm(const BoltVector& first_vector,
                     const BoltVector& second_vector,
                     const std::string& norm_order) {
    if (first_vector.len != second_vector.len) {
      throw std::invalid_argument(
          "Invalid arguments for the call to NodeProperties::norm. BoltVectors "
          "have different lengths. ");
    }
    LPNorm norm = getNorm(norm_order);
    std::vector<float> vec_difference = {0};

    for (uint32_t activation_index = 0; activation_index < first_vector.len;
         activation_index++) {
      vec_difference.push_back(first_vector.activations[activation_index] -
                               second_vector.activations[activation_index]);
    }

    return computeNorm(/* activations= */ &vec_difference[0],
                       /* len= */ first_vector.len, /*norm= */ norm);
  }

 private:
  NodeProperties() {}

  static double computeNorm(const float* activations, uint32_t len,
                            LPNorm norm) {
    switch (norm) {
      case LPNorm::L1: {
        double accumulator = 0.0;
        for (uint32_t activation_index = 0; activation_index < len;
             activation_index++) {
          accumulator += fabs(activations[activation_index]);
        }
        return accumulator;
      }
      case LPNorm::Euclidean: {
        double accumulator = 0.0;
        for (uint32_t activation_index = 0; activation_index < len;
             activation_index++) {
          accumulator += pow(activations[activation_index], 2.0);
        }
        return sqrt(accumulator);
      }
      case LPNorm::LInfinity: {
        double accumulator = static_cast<double>(abs(*activations));
        for (uint32_t activation_index = 0; activation_index < len;
             activation_index++) {
          accumulator =
              std::max<double>(accumulator,
                       fabs(static_cast<double>(activations[activation_index])));
        }
        return accumulator;
      }
    }
    throw std::invalid_argument(
        "" + LPNormToStr(norm) +
        " is not a valid LP Norm. Valid norms include L-1 "
        "norm, Euclidean norm and L-infinity norm.");
  }
};

}  // namespace thirdai::bolt