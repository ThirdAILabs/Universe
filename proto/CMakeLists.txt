include(FindProtobuf)
find_package(Protobuf REQUIRED)

INCLUDE_DIRECTORIES(${Protobuf_INCLUDE_DIR})

set(
  THIRDAI_PROTO_DEFS
  hashing/hashing.proto
  hashtable/hashtable.proto
  bolt/loss.proto
  bolt/model.proto
  bolt/neuron_index.proto
  bolt/ops.proto
  bolt/optimizers.proto
  bolt/parameter.proto
)

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HEADERS ${THIRDAI_PROTO_DEFS})
ADD_LIBRARY(proto ${PROTO_HEADERS} ${PROTO_SRCS})