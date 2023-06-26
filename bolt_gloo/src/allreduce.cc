#include "collective.h"
#include <gloo/gloo/allreduce.h>
#include <gloo/gloo/reduce.h>

namespace boltgloo {

template <typename T>
void allreduce(const std::shared_ptr<gloo::Context> &context, intptr_t sendbuf,
               intptr_t recvbuf, size_t size, ReduceOp reduceop,
               gloo::AllreduceOptions::Algorithm algorithm, uint32_t tag) {
  std::vector<T *> input_ptr{reinterpret_cast<T *>(sendbuf)};
  std::vector<T *> output_ptr{reinterpret_cast<T *>(recvbuf)};

  // Configure AllreduceOptions struct and call allreduce function
  gloo::AllreduceOptions opts_(context);
  opts_.setInputs(input_ptr, size);
  opts_.setOutputs(output_ptr, size);
  opts_.setAlgorithm(algorithm);
  gloo::ReduceOptions::Func fn = toFunction<T>(reduceop);
  opts_.setReduceFunction(fn);
  opts_.setTag(tag);

  gloo::allreduce(opts_);
}

} // namespace boltgloo