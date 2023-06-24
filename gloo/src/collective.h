#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <gloo/gloo/allreduce.h>
#include <gloo/gloo/context.h>
#include <gloo/gloo/math.h>
#include <gloo/gloo/types.h>

namespace pygloo {

enum class ReduceOp : std::uint8_t {
  SUM = 0,
  PRODUCT,
  MIN,
  MAX,
  BAND, // Bitwise AND
  BOR,  // Bitwise OR
  BXOR, // Bitwise XOR
  UNUSED,
};

typedef void (*ReduceFunc)(void *, const void *, const void *, size_t);

template <typename T> ReduceFunc toFunction(const ReduceOp &r) {
  switch (r) {
  case ReduceOp::SUM:
    return ReduceFunc(&gloo::sum<T>);
  case ReduceOp::PRODUCT:
    return ReduceFunc(&gloo::product<T>);
  case ReduceOp::MIN:
    return ReduceFunc(&gloo::min<T>);
  case ReduceOp::MAX:
    return ReduceFunc(&gloo::max<T>);
  case ReduceOp::BAND:
    throw std::runtime_error(
        "Cannot use ReduceOp.BAND with non-integral dtype");
    break;
  case ReduceOp::BOR:
    throw std::runtime_error("Cannot use ReduceOp.BOR with non-integral dtype");
    break;
  case ReduceOp::BXOR:
    throw std::runtime_error(
        "Cannot use ReduceOp.BXOR with non-integral dtype");
    break;
  case ReduceOp::UNUSED:
    break;
  }

  throw std::runtime_error("Unhandled ReduceOp");
}


} // namespace pygloo